"""Offline integrity and quality audit of the published Strands v19/v21 runs."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from compare_four import STRANDS_IDS, _choice, _quality

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/2026-10-06"
CASES = {"kiro": "kiro_article_cases.json", "support": "clef_vs_laya_cases.json"}


def audit() -> dict:
    summary = {}
    for fixture_name, fixture_file in CASES.items():
        fixture = json.loads((ROOT / "benchmarks" / fixture_file).read_text(encoding="utf-8"))
        cases, questions = fixture["cases"], fixture["questions"]
        by_id = {case["id"]: case for case in cases}
        summary[fixture_name] = {}
        for backend in (("mlx", "mps") if fixture_name == "kiro" else ("mlx",)):
            decisions = {}
            for version, model_id in STRANDS_IDS.items():
                prefix = "strands" if version == "v19" else "strands-v21"
                name = f"results-{prefix}-{'mps-' if backend == 'mps' else ''}{fixture_name}-raw-20261006.json"
                data = json.loads((RESULTS / name).read_text(encoding="utf-8"))
                if data["model_id_requested"] != model_id or data["model"] != "strands":
                    raise ValueError(f"wrong requested model in {name}")
                records = data["sequential"]["records"]
                if len(records) != len(cases) or {r["id"] for r in records} != set(by_id):
                    raise ValueError(f"missing or duplicate cases in {name}")
                decisions[version] = {}
                for record in records:
                    case = by_id[record["id"]]
                    request = {"state": case["state"], "questions": questions, "model": model_id}
                    request_bytes = json.dumps(request, ensure_ascii=False).encode("utf-8")
                    body = base64.b64decode(record["raw_response_base64"], validate=True)
                    if (record["status"] != 200 or record["response_model"] != model_id
                            or hashlib.sha256(request_bytes).hexdigest() != record["request_sha256"]
                            or hashlib.sha256(body).hexdigest() != record["raw_response_sha256"]
                            or json.loads(body)["model"] != model_id
                            or json.loads(body)["answers"] != record["answers"]):
                        raise ValueError(f"request, response, or body mismatch in {name}:{record['id']}")
                    decisions[version][record["id"]] = {
                        qid: _choice(record["answers"][qid], question)
                        for qid, question in questions.items()
                    }
                quality = _quality(records, cases, questions)
                if quality != data["quality"]:
                    raise ValueError(f"quality mismatch in {name}")
                summary[fixture_name].setdefault(backend, {})[version] = {
                    "accepted": quality["accepted_correct"], "correct": quality["correct"],
                    "total": quality["total"], "file": f"results/2026-10-06/{name}",
                }
            changes = [f"{case_id}:{qid} {decisions['v19'][case_id][qid]}→{decisions['v21'][case_id][qid]}"
                       for case_id in sorted(by_id) for qid in questions
                       if decisions["v19"][case_id][qid] != decisions["v21"][case_id][qid]]
            summary[fixture_name][backend]["changed_decisions"] = changes
    return summary


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
