"""Audit and export the six single-model batch result files for public GitHub."""

import argparse
import json
from pathlib import Path

from export_public_results import SENSITIVE, scrub, verify, walk


MODELS = ("laya", "strands", "clef")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    exported = []
    for model in MODELS:
        for suffix in ("batch", "http-batch"):
            name = f"{model}-{suffix}-20261007.json"
            raw = json.loads((args.source / name).read_text(encoding="utf-8"))
            if raw["model"] != model:
                raise ValueError(f"model mismatch in {name}")
            if suffix == "batch":
                if raw["model_instances"] != 1 or len(raw["runs"]) != 20:
                    raise ValueError(f"incomplete direct batch run: {name}")
                if any(run["request_count"] != 24 or run["decision_matches"] != 24
                       or run["forward_calls"] != 24 // run["batch_requests"]
                       for run in raw["runs"]):
                    raise ValueError(f"invalid direct batch result: {name}")
                public = raw
            else:
                public = scrub(raw)
                if Path(raw["fixture"]).name != "clef_vs_laya_cases.json":
                    raise ValueError(f"unexpected fixture in {name}")
                public["fixture"] = "benchmarks/clef_vs_laya_cases.json"
                verify(raw, public, name)
                if public["sequential"]["ok"] != 24 or any(
                    run["ok"] != 24 or any(record.get("inference_batch_size") != run["concurrency"]
                                           for record in run["records"])
                    for run in public["parallel"]
                ):
                    raise ValueError(f"HTTP batches incomplete in {name}")
            for location, value in walk(public):
                if SENSITIVE.search(value):
                    raise ValueError(f"sensitive text at {name}:{location}")
            exported.append((name, public))

    args.destination.mkdir(parents=True, exist_ok=True)
    for name, public in exported:
        (args.destination / name).write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: audited")


if __name__ == "__main__":
    main()
