"""Audit and publish paired FP16/q8 batch and replica results without local paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench_model_batch import decision
from compare_four import _quality
from export_public_results import SENSITIVE, scrub, verify, walk


VARIANTS = {
    "fp16": ("float16", "laya-multilingual", 56),
    "q8": ("int8_mixed", "laya-multilingual-q8", 58),
}


def _check_public(data: dict, name: str) -> None:
    for location, value in walk(data):
        if SENSITIVE.search(value):
            raise ValueError(f"sensitive text at {name}:{location}")


def _same_decisions(response: dict, baseline: dict) -> bool:
    return all(decision(response["answers"][qid]) == decision(reference)
               for qid, reference in baseline["answers"].items())


def export(source: Path, destination: Path, fixture_dir: Path,
           verify_only: bool = False) -> list[str]:
    fixture = json.loads((fixture_dir / "clef_vs_laya_cases.json").read_text())
    cases, questions = fixture["cases"], fixture["questions"]
    exported = []
    for variant, (dtype, snapshot, expected_correct) in VARIANTS.items():
        precision = json.loads((destination / f"laya-{variant}-support-head512.json").read_text())
        if precision["quality"]["correct"] != expected_correct:
            raise ValueError(f"{variant}: precision result changed")
        precision_answers = {record["id"]: record["answers"] for record in precision["records"]}

        batch_name = f"laya-{variant}-batch-head512-20261007.json"
        batch = json.loads((source / batch_name).read_text())
        if (batch["model"] != "laya" or batch["model_snapshot"] != snapshot
                or batch["laya_dtype"] != dtype or batch["head_max_len"] != 512
                or batch["model_instances"] != 1 or batch["fixture"] != "benchmarks/clef_vs_laya_cases.json"
                or batch["repeats"] != 5 or len(batch["runs"]) != 20):
            raise ValueError(f"{batch_name}: configuration mismatch")
        baseline = batch["baseline_responses"]
        if set(baseline) != {case["id"] for case in cases}:
            raise ValueError(f"{batch_name}: missing baseline cases")
        if any(baseline[case["id"]]["answers"] != precision_answers[case["id"]] for case in cases):
            raise ValueError(f"{batch_name}: baseline differs from the pinned precision run")
        if {(run["trial"], run["batch_requests"]) for run in batch["runs"]} != {
            (trial, size) for trial in range(1, 6) for size in (1, 2, 4, 8)
        }:
            raise ValueError(f"{batch_name}: missing shape or trial")
        for run in batch["runs"]:
            if (run["request_count"] != 24 or run["forward_calls"] != 24 // run["batch_requests"]
                    or run["decision_matches"] != 24 or len(run["records"]) != 24
                    or {record["case_id"] for record in run["records"]} != set(baseline)):
                raise ValueError(f"{batch_name}: incomplete or incorrect run")
            if any(not _same_decisions(record["response"], baseline[record["case_id"]])
                   for record in run["records"]):
                raise ValueError(f"{batch_name}: decision changed in a batch")
        _check_public(batch, batch_name)
        exported.append((batch_name, batch))

        replica_name = f"laya-{variant}-replicas-head512-20261007.json"
        replicas = json.loads((source / replica_name).read_text())
        if (replicas["model"] != "laya" or replicas["model_snapshot"] != snapshot
                or replicas["laya_dtype"] != dtype or replicas["head_max_len"] != 512
                or replicas["model_instances"] != [1, 2, 4]
                or replicas["concurrency_levels"] != [1, 2, 4, 8]
                or replicas["repeats"] != 3 or replicas["server_batch_requests"] != 1
                or replicas["fixture"] != "benchmarks/clef_vs_laya_cases.json"
                or len(replicas["groups"]) != 3):
            raise ValueError(f"{replica_name}: configuration mismatch")
        for group, count in zip(replicas["groups"], (1, 2, 4), strict=True):
            if (group["replicas"] != count or len(group["warmups"]) != count
                    or len(group["runs"]) != 12 or {(run["trial"], run["concurrency"])
                    for run in group["runs"]} != {(trial, concurrency)
                    for trial in (1, 2, 3) for concurrency in (1, 2, 4, 8)}):
                raise ValueError(f"{replica_name}: incomplete replica group")
            for run in group["runs"]:
                if (run["ok"] != 24 or len(run["records"]) != 24
                        or run["quality"]["correct"] != expected_correct
                        or _quality(run["records"], cases, questions) != run["quality"]):
                    raise ValueError(f"{replica_name}: quality or response count mismatch")
                for record in run["records"]:
                    if (record["status"] != 200 or record.get("inference_batch_size") != 1
                            or not 0 <= record.get("replica_index", -1) < count
                            or record["answers"] != baseline[record["id"]]["answers"]):
                        raise ValueError(f"{replica_name}: invalid model response")
        replica_public = scrub(replicas)
        verify(replicas, replica_public, replica_name)
        _check_public(replica_public, replica_name)
        exported.append((replica_name, replica_public))

        http_source_name = f"laya-{variant}-http-batch-verified-20261007.json"
        http_name = f"laya-{variant}-http-batch-head512-20261007.json"
        http = json.loads((source / http_source_name).read_text())
        if (http["model"] != "laya" or http["model_id_requested"] != snapshot
                or http["laya_precision"] != variant
                or http["server_health"] != {"laya_dtype": dtype, "head_max_len": 512,
                                             "model_instances": 1, "max_batch_requests": 8}
                or Path(http["fixture"]).name != "clef_vs_laya_cases.json"):
            raise ValueError(f"{http_name}: server or checkpoint mismatch")
        rounds = [http["sequential"], *http["parallel"]]
        if [run["concurrency"] for run in rounds] != [1, 2, 4, 8]:
            raise ValueError(f"{http_name}: concurrency levels mismatch")
        for run in rounds:
            if (run["ok"] != 24 or len(run["records"]) != 24
                    or _quality(run["records"], cases, questions)["correct"] != expected_correct):
                raise ValueError(f"{http_name}: incomplete or inaccurate run")
            if any(record["status"] != 200
                   or record.get("inference_batch_size") != run["concurrency"]
                   or not _same_decisions({"answers": record["answers"]}, baseline[record["id"]])
                   for record in run["records"]):
                raise ValueError(f"{http_name}: failed, wrong-sized, or changed response")
        http_public = scrub(http)
        http_public["fixture"] = "benchmarks/clef_vs_laya_cases.json"
        verify(http, http_public, http_name)
        _check_public(http_public, http_name)
        exported.append((http_name, http_public))

    destination.mkdir(parents=True, exist_ok=True)
    if verify_only:
        for name, data in exported:
            if json.loads((destination / name).read_text()) != data:
                raise ValueError(f"{name}: published result differs from audited source")
        return [name for name, _ in exported]
    if any((destination / name).exists() for name, _ in exported):
        raise FileExistsError("a published output already exists; refusing to overwrite")
    for name, data in exported:
        (destination / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return [name for name, _ in exported]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--fixtures", type=Path, default=Path("benchmarks"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for name in export(args.source, args.destination, args.fixtures, args.verify_only):
        print(f"{name}: audited")


if __name__ == "__main__":
    main()
