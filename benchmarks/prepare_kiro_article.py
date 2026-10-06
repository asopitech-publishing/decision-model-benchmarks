"""Extract the Japanese 20-case appendix from the cited Classmethod article.

This local evaluation copy is for research only; cite the original article when
reporting results. The English translations are not published there.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

URL = "https://dev.classmethod.jp/articles/strands-decider-2b-m4-mac-kiro-crew/"
OUT = Path(__file__).resolve().parent / "kiro_article_cases.json"

TIER_DESCRIPTIONS = {
    "simple": "a short, local, mechanical request answerable in one step -- a rename, a lookup, a small edit, a direct factual question",
    "medium": "ordinary work over a few files or steps -- write a function, explain some code, fix a bug whose cause is already named",
    "complex": "work needing a plan, a trade-off or reasoning across a whole system -- design, architecture, a diagnosis with no named cause, a risky refactor",
}
INSTRUCTIONS = (
    "How hard is this request for an AI coding assistant? Answer one of: "
    "simple = a short, local, mechanical request answerable in one step -- a rename, a lookup, "
    "a small edit, a direct factual question; medium = ordinary work over a few files or steps -- "
    "write a function, explain some code, fix a bug whose cause is already named; complex = work "
    "needing a plan, a trade-off or reasoning across a whole system -- design, architecture, "
    "a diagnosis with no named cause, a risky refactor.\n"
    "Being wrong in the two directions does not cost the same. A turn put in a lower tier than it "
    "needs gets less room to work in and a worse answer; a turn put higher only costs more. "
    "So answer simple only when the request is self-contained and you can see the whole of it -- "
    "a request that is a plan, a judgement call, several instructions at once, or that names work "
    "you cannot see, is not simple however short it is."
)


def main() -> None:
    with urllib.request.urlopen(URL, timeout=30) as response:
        soup = BeautifulSoup(response.read(), "html.parser")
    cases = []
    pattern = re.compile(r"^(P\d{2})（expected: (simple|medium|complex) / accept: ([^）]+)）$")
    for paragraph in soup.select("p"):
        match = pattern.match(paragraph.get_text(strip=True))
        if not match:
            continue
        block = paragraph.find_next_sibling("div")
        code = block.find("pre") if block else None
        if code is None:
            raise ValueError(f"Missing state for {match.group(1)}")
        case_id, expected, accepted_text = match.groups()
        accepted = [value.strip() for value in accepted_text.split(",")]
        if expected not in accepted:
            raise ValueError(f"Gold not accepted for {case_id}")
        cases.append({"id": case_id, "state": {"message": code.get_text().strip()},
                      "gold": {"tier": expected}, "accepted": {"tier": accepted}})
    if [case["id"] for case in cases] != [f"P{i:02d}" for i in range(1, 21)]:
        raise ValueError("Expected exactly the article's ordered P01-P20 cases")
    fixture = {"source": URL, "language": "ja", "variant": "B_desc",
               "labels": "Claude Opus 5.5-assigned in the cited article, not human ground truth",
               "questions": {"tier": {"type": "choice", "instructions": INSTRUCTIONS,
                                      "criteria": TIER_DESCRIPTIONS}},
               "cases": cases}
    OUT.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(cases)} cases to {OUT}; lengths: {[len(c['state']['message']) for c in cases]}")


if __name__ == "__main__":
    main()
