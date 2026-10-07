"""Benchmark true cross-request inference batches with one local model instance.

Use a Python environment containing the selected model's runtime. The full
per-request outputs are retained; only a model directory basename is recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def decision(answer):
    kind = answer["type"]
    if kind == "choice":
        return answer["choice"]
    if kind == "noul":
        return answer["noul"] >= 0.5
    return round(answer["score"])


def deltas(reference, actual):
    probability_delta = 0.0
    score_delta = 0.0
    for qid, expected in reference["answers"].items():
        observed = actual["answers"][qid]
        for option, value in expected.get("probabilities", {}).items():
            probability_delta = max(probability_delta, abs(value - observed["probabilities"][option]))
        if expected["type"] == "score":
            score_delta = max(score_delta, abs(expected["score"] - observed["score"]))
        elif expected["type"] == "noul":
            probability_delta = max(probability_delta, abs(expected["noul"] - observed["noul"]))
    return probability_delta, score_delta


def load_backend(kind, model_path, largest_batch, question_count):
    if kind == "laya":
        from laya_mlx.agent import load
        from laya_batch import predict_many
        agent = load(model_path, dtype="int8_mixed", cache_prompts=False,
                     batch_size=largest_batch * question_count)
        return agent, lambda packet: agent.predict(packet["state"], packet["questions"]), \
            predict_many, question_count
    if kind == "strands":
        from strands_decider.infer import load_engine
        from strands_decider.schema import SystemOneRequest
        from strands_batch import predict_many
        engine = load_engine(model_path, device="mlx", use_prefix_cache=False)
        return engine, lambda packet: engine.evaluate(SystemOneRequest.model_validate(packet)).model_dump(), \
            predict_many, question_count
    if kind == "clef":
        from clef_mlx import load
        from clef_batch import predict_many
        model = load(model_path)
        return model, lambda packet: model.systemone({"model": "clef-flash-4bit", **packet}), \
            predict_many, 1
    raise ValueError(f"unsupported model kind: {kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=["laya", "strands", "clef"], required=True)
    parser.add_argument("--model", required=True, help="local model directory or cached model id")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes) or len(set(args.sizes)) != len(args.sizes) or args.repeats < 1:
        parser.error("sizes must be distinct positive integers and repeats must be positive")

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    cases, questions = fixture["cases"], fixture["questions"]
    if not cases or not questions or len({case["id"] for case in cases}) != len(cases):
        parser.error("fixture needs nonempty questions and uniquely identified cases")
    agent, baseline_call, batch_call, rows_per_request = load_backend(
        args.model_kind, args.model, max(args.sizes), len(questions)
    )
    import mlx.core as mx
    packets = {case["id"]: {"state": case["state"], "questions": questions} for case in cases}
    baseline = {case["id"]: baseline_call(packets[case["id"]]) for case in cases}

    # Warm the local model and each batch shape. This makes no cloud/API requests.
    for size in args.sizes:
        for start in range(0, len(cases), size):
            group = cases[start : start + size]
            batch_call(agent, [packets[case["id"]] for case in group],
                       max_rows=size * rows_per_request)

    output = {
        "model": args.model_kind, "model_snapshot": args.model.rstrip("/").split("/")[-1],
        "fixture": f"benchmarks/{args.fixture.name}", "model_instances": 1,
        "batching": "independent requests in one model forward pass",
        "cache_prompts": False, "shape_warmup": "one untimed fixture pass per batch size",
        "repeats": args.repeats, "baseline_responses": baseline, "runs": [],
    }
    for trial in range(args.repeats):
        ordered = list(cases)
        random.Random(20261007 + trial).shuffle(ordered)
        sizes = list(args.sizes)
        random.Random(20261007 + trial * 31).shuffle(sizes)
        for size in sizes:
            groups = [ordered[start : start + size] for start in range(0, len(ordered), size)]
            records, batch_latencies = [], []
            forward_calls = 0
            mx.reset_peak_memory()
            started = time.perf_counter()
            for group in groups:
                batch_started = time.perf_counter()
                responses, calls = batch_call(
                    agent, [packets[case["id"]] for case in group],
                    max_rows=size * rows_per_request,
                )
                batch_latencies.append((time.perf_counter() - batch_started) * 1000)
                forward_calls += calls
                for case, response in zip(group, responses, strict=True):
                    reference = baseline[case["id"]]
                    probability_delta, score_delta = deltas(reference, response)
                    records.append({
                        "case_id": case["id"], "response": response,
                        "exact_response_match": response == reference,
                        "decision_match": all(decision(response["answers"][qid]) ==
                                              decision(reference["answers"][qid]) for qid in questions),
                        "max_probability_delta": probability_delta,
                        "max_score_delta": score_delta,
                    })
            wall_s = time.perf_counter() - started
            if forward_calls != len(groups):
                raise RuntimeError("request group used more than one model forward pass")
            run = {
                "trial": trial + 1, "batch_requests": size, "request_count": len(records),
                "forward_calls": forward_calls, "wall_s": round(wall_s, 4),
                "requests_per_s": round(len(records) / wall_s, 3),
                "batch_latency_p50_ms": round(statistics.median(batch_latencies), 3),
                "batch_latency_p95_ms": round(percentile(batch_latencies, 0.95), 3),
                "metal_peak_memory_bytes": mx.get_peak_memory(),
                "exact_response_matches": sum(record["exact_response_match"] for record in records),
                "decision_matches": sum(record["decision_match"] for record in records),
                "max_probability_delta": max(record["max_probability_delta"] for record in records),
                "max_score_delta": max(record["max_score_delta"] for record in records),
                "records": records,
            }
            output["runs"].append(run)
            print(json.dumps({key: value for key, value in run.items() if key != "records"}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(mode=0o600, exist_ok=True)
    os.chmod(args.out, 0o600)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
