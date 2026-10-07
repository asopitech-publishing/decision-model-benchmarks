"""Audit and export three multi-process replica runs for public GitHub."""

import argparse
import json
from pathlib import Path

from export_public_results import SENSITIVE, scrub, verify, walk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    exported = []
    for model in ("laya", "strands", "clef"):
        name = f"{model}-replicas-20261007.json"
        raw = json.loads((args.source / name).read_text(encoding="utf-8"))
        if (raw["model"] != model or raw["model_instances"] != [1, 2, 4]
                or raw["concurrency_levels"] != [1, 2, 4, 8]
                or raw["repeats"] != 3 or raw["server_batch_requests"] != 1
                or raw["fixture"] != "benchmarks/clef_vs_laya_cases.json"
                or len(raw["groups"]) != 3):
            raise ValueError(f"unexpected run configuration in {name}")
        for group, count in zip(raw["groups"], (1, 2, 4), strict=True):
            if (group["replicas"] != count or len(group["warmups"]) != count
                    or len(group["runs"]) != 12
                    or any(record.get("status") != 200
                           or record.get("inference_batch_size") != 1
                           for record in group["warmups"])
                    or {(run["trial"], run["concurrency"]) for run in group["runs"]}
                       != {(trial, concurrency) for trial in (1, 2, 3)
                           for concurrency in (1, 2, 4, 8)}):
                raise ValueError(f"incomplete replica group in {name}")
            for run in group["runs"]:
                if (run["concurrency"] not in {1, 2, 4, 8} or run["ok"] != 24
                        or len(run["records"]) != 24 or any(
                            record.get("status") != 200
                            or record.get("inference_batch_size") != 1
                            or not 0 <= record.get("replica_index", -1) < count
                            for record in run["records"]
                        )):
                    raise ValueError(f"invalid response in {name}")
        public = scrub(raw)
        count = verify(raw, public, name)
        for location, value in walk(public):
            if SENSITIVE.search(value):
                raise ValueError(f"sensitive text at {name}:{location}")
        exported.append((name, public, count))

    args.destination.mkdir(parents=True, exist_ok=True)
    for name, public, count in exported:
        (args.destination / name).write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: {count} unchanged HTTP response bodies audited")


if __name__ == "__main__":
    main()
