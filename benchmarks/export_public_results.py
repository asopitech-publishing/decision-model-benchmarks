"""Export the audited 2026-10-06 benchmark run for a public repository.

The HTTP response body remains byte-for-byte recoverable from raw_response_base64.
Only local fixture paths and response headers are removed from surrounding metadata.
"""

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path


RUN_FILES = (
    "results-clef-kiro-raw-20261006.json",
    "results-clef-support-raw-20261006.json",
    "results-jev-kiro-raw-20261006.json",
    "results-jev-support-raw-20261006.json",
    "results-laya-kiro-raw-20261006.json",
    "results-laya-support-raw-20261006.json",
    "results-strands-kiro-raw-20261006.json",
    "results-strands-mps-kiro-raw-20261006.json",
    "results-strands-support-raw-20261006.json",
    "results-strands-v21-kiro-raw-20261006.json",
    "results-strands-v21-mps-kiro-raw-20261006.json",
    "results-strands-v21-support-raw-20261006.json",
)

SENSITIVE = re.compile(
    r"(?:/Users/|/home/|/private/|/Volumes/|file://|"
    r"[A-Za-z]:\\(?:Users|Documents and Settings)\\|"
    r"\b(?:Bearer\s+\S+|gh[opusr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|"
    r"sk-[A-Za-z0-9_-]{12,}|sk-proj-[A-Za-z0-9_-]+)\b)",
    re.IGNORECASE,
)


def walk(value, location="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{location}[{index}]")
    elif isinstance(value, str):
        yield location, value


def scrub(value):
    if isinstance(value, dict):
        return {key: scrub(child) for key, child in value.items() if key != "response_headers"}
    if isinstance(value, list):
        return [scrub(child) for child in value]
    return value


def verify(raw, public, name):
    for location, value in walk(public):
        if SENSITIVE.search(value):
            raise ValueError(f"sensitive text at {name}:{location}")

    original_bodies = [(location, value) for location, value in walk(raw)
                       if location.endswith(".raw_response_base64")]
    public_bodies = [(location, value) for location, value in walk(public)
                     if location.endswith(".raw_response_base64")]
    if not original_bodies or original_bodies != public_bodies:
        raise ValueError(f"raw response bodies changed or missing in {name}")
    for location, encoded in public_bodies:
        body = base64.b64decode(encoded, validate=True)
        parent = location.rsplit(".", 1)[0]
        record = public
        for part in re.findall(r"[^.\[\]]+|\[\d+\]", parent)[1:]:
            record = record[int(part[1:-1])] if part.startswith("[") else record[part]
        if hashlib.sha256(body).hexdigest() != record["raw_response_sha256"]:
            raise ValueError(f"response hash mismatch at {name}:{location}")
        if body.decode("utf-8", errors="replace") != record["raw_response_text"]:
            raise ValueError(f"response text mismatch at {name}:{location}")
        if SENSITIVE.search(body.decode("utf-8", errors="replace")):
            raise ValueError(f"sensitive text inside response body at {name}:{location}")
    return len(public_bodies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory containing the original results")
    parser.add_argument("destination", type=Path, help="directory for public results")
    args = parser.parse_args()
    exported = []
    for name in RUN_FILES:
        raw = json.loads((args.source / name).read_text(encoding="utf-8"))
        public = scrub(raw)
        fixture = Path(raw["fixture"]).name
        if fixture not in {"kiro_article_cases.json", "clef_vs_laya_cases.json"}:
            raise ValueError(f"unexpected fixture in {name}")
        public["fixture"] = f"benchmarks/{fixture}"
        count = verify(raw, public, name)
        exported.append((name, public, count))
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, public, count in exported:
        (args.destination / name).write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: {count} unchanged HTTP response bodies")


if __name__ == "__main__":
    main()
