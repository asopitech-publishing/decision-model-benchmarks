"""Compare unmodified Laya FP16 and locally quantized q8 on the same fixtures.

Inference runs in-process on MLX. The checkpoint files are never edited; the
question token budget is overridden in memory and every input is audited for
truncation before it is scored. No network service or API credential is used.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from compare_four import _quality


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_untruncated(agent, state: object, questions: dict) -> dict:
    """Fail closed if either instructions, options, or state would be clipped."""
    from laya_mlx.common import render_options, serialize_state

    items, internal = agent.prepare(state, questions)
    state_tokens = len(agent.tok(serialize_state(state), add_special_tokens=False)["input_ids"])
    details = {}
    for (qid, _), item, question in zip(questions.items(), items, internal):
        mask = agent.tok.mask_token
        head = agent.tok(
            f"{question['t']} question: {question['ins'].replace(mask, ' ')}",
            add_special_tokens=False,
        )["input_ids"]
        options = [
            agent.tok(" " + option.replace(mask, " "), add_special_tokens=False)["input_ids"]
            for option in render_options(question)
        ]
        option_lengths = [len(option) for option in options]
        required_head = len(head) + sum(length + 1 for length in option_lengths)
        required_total = required_head + state_tokens + 4  # CLS and three SEP tokens
        if any(length > 48 for length in option_lengths):
            raise ValueError(f"{qid}: option description exceeds the 48-token per-option limit")
        if required_head > agent.cfg["head_max_len"]:
            raise ValueError(f"{qid}: instructions/options exceed head_max_len")
        if required_total > agent.cfg["max_len"] or len(item["ids"]) != required_total:
            raise ValueError(f"{qid}: state or question exceeds max_len")
        details[qid] = {
            "instruction_tokens": len(head),
            "option_tokens_with_markers": required_head - len(head),
            "state_tokens": state_tokens,
            "input_tokens": required_total,
        }
    return details


def run_precision(precision: str, model_path: Path, fixture: dict, fixture_name: str,
                  fixture_sha256: str, head_max_len: int) -> dict:
    from laya_mlx.agent import Agent

    dtype = {"fp16": "float16", "q8": "int8_mixed"}[precision]
    started = time.perf_counter()
    agent = Agent(model_path, dtype=dtype, device="gpu")
    load_s = round(time.perf_counter() - started, 3)
    agent.cfg["head_max_len"] = head_max_len
    cases, questions = fixture["cases"], fixture["questions"]
    if not cases or not questions:
        raise ValueError("fixture needs cases and questions")
    # Audit every case before the first inference; do not publish a partially
    # truncated run as a valid precision comparison.
    audits = {case["id"]: check_untruncated(agent, case["state"], questions)
              for case in cases}
    warmup = agent.system_one(cases[0]["state"], questions)
    records = []
    for case in cases:
        started = time.perf_counter()
        response = agent.system_one(case["state"], questions)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        raw = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        expected_tokens = sum(part["input_tokens"] for part in audits[case["id"]].values())
        if response["usage"]["input_tokens"] != expected_tokens:
            raise ValueError(f"{case['id']}: usage token count does not match audited input")
        records.append({
            "id": case["id"], "status": 200, "latency_ms": latency_ms,
            "input_audit": audits[case["id"]], "answers": response["answers"],
            "usage": response["usage"], "raw_response_text": raw.decode("utf-8"),
            "raw_response_base64": base64.b64encode(raw).decode("ascii"),
            "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        })
    latencies = [record["latency_ms"] for record in records]
    cfg = json.loads((model_path / "rl_agent_config.json").read_text())
    mlx_config_path = model_path / "mlx_config.json"
    mlx_config = json.loads(mlx_config_path.read_text()) if mlx_config_path.is_file() else {}
    return {
        "precision": precision,
        "checkpoint": "convaiinnovations/laya-multilingual" + (" (local q8 conversion)" if precision == "q8" else ""),
        "checkpoint_weights_sha256": sha256_file(model_path / "model.safetensors"),
        "checkpoint_config_head_max_len": cfg["head_max_len"],
        "quantization": mlx_config.get("quantization"),
        "runtime": "laya-mlx in-process MLX GPU",
        "max_len": agent.cfg["max_len"], "head_max_len": head_max_len,
        "fixture": f"benchmarks/{fixture_name}", "fixture_sha256": fixture_sha256,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "warmup_policy": "one local unscored inference",
        "warmup_response": warmup,
        "load_s": load_s,
        "latency_p50_ms": round(statistics.median(latencies), 3),
        "quality": _quality(records, cases, questions),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", required=True, choices=("fp16", "q8"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--head-max-len", type=int, default=512)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a new file")
    if args.head_max_len <= 4:
        parser.error("head-max-len must be greater than 4")
    if args.fixture.name not in {"kiro_article_cases.json", "clef_vs_laya_cases.json"}:
        parser.error("use one of the published fixtures")
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    result = run_precision(args.precision, args.model, fixture, args.fixture.name,
                           hashlib.sha256(fixture_bytes).hexdigest(), args.head_max_len)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8",
              opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"precision": args.precision, "fixture": result["fixture"],
                      "exact": result["quality"]["correct"],
                      "accepted": result["quality"]["accepted_correct"],
                      "total": result["quality"]["total"],
                      "p50_ms": result["latency_p50_ms"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
