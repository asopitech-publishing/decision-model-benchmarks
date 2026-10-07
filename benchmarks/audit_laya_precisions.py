"""Audit the published, in-process FP16/q8 Laya results without loading MLX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compare_four import _choice, _quality
from export_public_results import verify


FIXTURES = {
    "kiro": "kiro_article_cases.json",
    "support": "clef_vs_laya_cases.json",
}
CHECKPOINT_SHA256 = {
    "fp16": "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
    "q8": "4efc8723f42950d07fd79d32d032e5ef06bd3e4b3a7fbcc58c4b5a125ed85b86",
}


def audit(results_dir: Path, fixture_dir: Path) -> dict:
    summary = {}
    hashes = {}
    for suite, fixture_name in FIXTURES.items():
        fixture_bytes = (fixture_dir / fixture_name).read_bytes()
        fixture = json.loads(fixture_bytes)
        pair = {}
        for precision in ("fp16", "q8"):
            name = f"laya-{precision}-{suite}-head512.json"
            data = json.loads((results_dir / name).read_text())
            if data["precision"] != precision or data["fixture"] != f"benchmarks/{fixture_name}":
                raise ValueError(f"{name}: precision or fixture mismatch")
            if data["fixture_sha256"] != hashlib.sha256(fixture_bytes).hexdigest():
                raise ValueError(f"{name}: fixture hash mismatch")
            if data["head_max_len"] != 512 or data["max_len"] != 1024:
                raise ValueError(f"{name}: token budget mismatch")
            if data["checkpoint_weights_sha256"] != CHECKPOINT_SHA256[precision]:
                raise ValueError(f"{name}: checkpoint hash mismatch")
            if (precision == "q8") != (data["quantization"] is not None):
                raise ValueError(f"{name}: quantization metadata mismatch")
            if [row["id"] for row in data["records"]] != [case["id"] for case in fixture["cases"]]:
                raise ValueError(f"{name}: missing, duplicate, or out-of-order cases")
            if _quality(data["records"], fixture["cases"], fixture["questions"]) != data["quality"]:
                raise ValueError(f"{name}: quality summary mismatch")
            for row in data["records"]:
                if row["status"] != 200 or row["usage"]["input_tokens"] > data["max_len"]:
                    raise ValueError(f"{name}: failed request or truncated input")
                if row["usage"]["input_tokens"] != sum(
                    detail["input_tokens"] for detail in row["input_audit"].values()
                ):
                    raise ValueError(f"{name}: input token count mismatch")
                if any(detail["instruction_tokens"] + detail["option_tokens_with_markers"]
                       > data["head_max_len"] for detail in row["input_audit"].values()):
                    raise ValueError(f"{name}: head token count exceeds budget")
            verify(data, data, name)  # Includes sensitive-text and raw-byte hash checks.
            previous_hash = hashes.setdefault(precision, data["checkpoint_weights_sha256"])
            if previous_hash != data["checkpoint_weights_sha256"]:
                raise ValueError(f"{name}: checkpoint changed between suites")
            pair[precision] = data
        changes = []
        for fp16, q8 in zip(pair["fp16"]["records"], pair["q8"]["records"]):
            for qid, question in fixture["questions"].items():
                left = _choice(fp16["answers"][qid], question)
                right = _choice(q8["answers"][qid], question)
                if left != right:
                    changes.append({"id": fp16["id"], "question": qid,
                                    "fp16": left, "q8": right})
        summary[suite] = {
            "fp16_exact": pair["fp16"]["quality"]["correct"],
            "q8_exact": pair["q8"]["quality"]["correct"],
            "fp16_accepted": pair["fp16"]["quality"]["accepted_correct"],
            "q8_accepted": pair["q8"]["quality"]["accepted_correct"],
            "total": pair["fp16"]["quality"]["total"],
            "changed_decisions": changes,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/2026-10-07"))
    parser.add_argument("--fixtures", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()
    print(json.dumps(audit(args.results, args.fixtures), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
