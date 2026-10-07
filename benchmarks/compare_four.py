"""Small, reproducible four-way System One benchmark over the existing Japanese fixture.

Run one model at a time: --model strands|clef|laya|jev. Local servers must already
be running. The Jev credential comes from TYPESAFE_API_KEY or .env.local; it is
never written to the output. Each request asks the same three typed questions.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import random
import stat
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE / "clef_vs_laya_cases.json"
URLS = {
    "strands": "http://127.0.0.1:8012/v1/systemone",
    "clef": "http://127.0.0.1:8013/v1/systemone",
    "laya": "http://127.0.0.1:8014/v1/decision",
    "jev": "https://api.typesafe.ai/v1/systemone",
}
MODEL_IDS = {
    "strands": "strands-decider-2B-hobson-v21",
    "clef": "clef-flash-4bit",
    "laya": "laya-multilingual-q8",
    "jev": "jev-latest",
}
LAYA_IDS = {"fp16": "laya-multilingual", "q8": "laya-multilingual-q8"}
STRANDS_IDS = {version: f"strands-decider-2B-hobson-{version}"
               for version in ("v19", "v21")}


def _key() -> str | None:
    value = os.environ.get("TYPESAFE_API_KEY")
    if value:
        return value
    path = HERE / ".env.local"
    if not path.exists():
        return None
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RuntimeError(f"Set private permissions first: chmod 600 {path}")
    for line in path.read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"\'')
            if value and value != "replace-with-your-typesafe-api-key":
                return value
    return None


def _check_laya_health(precision: str, head_max_len: int) -> dict:
    url = URLS["laya"].rsplit("/v1/", 1)[0] + "/healthz"
    with urllib.request.urlopen(url, timeout=5) as response:
        health = json.load(response)
    expected_dtype = "float16" if precision == "fp16" else "int8_mixed"
    if (health.get("laya_dtype") != expected_dtype
            or health.get("head_max_len") != head_max_len
            or health.get("model_instances") != 1):
        raise RuntimeError("Laya server precision, head budget, or model count mismatch")
    return {key: health[key] for key in
            ("laya_dtype", "head_max_len", "model_instances", "max_batch_requests")}


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


def _request(model: str, case: dict, questions: dict, key: str | None,
             url: str | None = None, model_id: str | None = None) -> dict:
    body = {"state": case["state"], "questions": questions}
    if model != "laya":
        body["model"] = model_id or MODEL_IDS[model]
    headers = {"Content-Type": "application/json"}
    if model == "jev":
        headers["Authorization"] = f"Bearer {key}"
    request_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url or URLS[model], data=request_bytes,
        headers=headers, method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            raw_body = response.read()
            status = response.status
            response_headers = response.headers
    except urllib.error.HTTPError as exc:
        status, raw_body, response_headers = exc.code, exc.read(), exc.headers
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {"id": case["id"], "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": None, "error": type(exc).__name__,
                "request_sha256": hashlib.sha256(request_bytes).hexdigest()}
    raw_text = raw_body.decode("utf-8", errors="replace")
    if key and key.encode("utf-8") in raw_body:
        raise RuntimeError("The API echoed the credential; refusing to save the response")
    try:
        data = json.loads(raw_body)
    except (ValueError, UnicodeDecodeError):
        data = None
    record = {"id": case["id"], "latency_ms": round((time.perf_counter() - started) * 1000, 2),
              "status": status, "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
              "raw_response_text": raw_text, "raw_response_base64": base64.b64encode(raw_body).decode("ascii"),
              "raw_response_sha256": hashlib.sha256(raw_body).hexdigest(),
              "response_headers": {name: value for name, value in response_headers.items()
                                   if name.lower() in {"content-type", "date", "x-request-id", "x-typesafe-request-id"}}}
    if response_headers.get("X-Inference-Batch-Size") is not None:
        record["inference_batch_size"] = int(response_headers["X-Inference-Batch-Size"])
    if status == 200 and isinstance(data, dict):
        record["answers"] = data.get("answers")
        record["response_model"] = data.get("model")
        if model == "strands":
            record["model_id_match"] = data.get("model") == body["model"]
        record["server_latency_ms"] = data.get("latency_ms")
        record["usage"] = data.get("usage")
    return record


def _choice(answer: dict, question: dict):
    kind = question["type"]
    if kind == "choice":
        return answer.get("choice") or max(answer.get("probabilities", {}), key=answer.get("probabilities", {}).get)
    if kind == "noul":
        return float(answer["noul"]) >= 0.5
    if kind == "score":
        levels = question["criteria"]
        probabilities = answer["probabilities"]
        # System One serialises ordinal probability keys as "0", "1", ...;
        # some local adapters instead use the level descriptions.
        return max(range(len(levels)), key=lambda idx: probabilities.get(str(idx),
                   probabilities.get(levels[idx], 0.0)))
    raise ValueError(kind)


def _quality(records: list[dict], cases: list[dict], questions: dict) -> dict:
    by_id = {case["id"]: case for case in cases}
    by_kind = {kind: {"correct": 0, "total": 0} for kind in ("choice", "score", "noul")}
    failures = []
    drift = []
    accepted_correct = 0
    for record in records:
        if record.get("status") != 200 or not isinstance(record.get("answers"), dict):
            failures.append({"id": record["id"], "status": record.get("status"), "error": record.get("error")})
            continue
        case = by_id[record["id"]]
        for qid, question in questions.items():
            kind = question["type"]
            bucket = by_kind[kind]
            bucket["total"] += 1
            try:
                predicted = _choice(record["answers"][qid], question)
            except (KeyError, ValueError, TypeError):
                drift.append({"id": record["id"], "question": qid, "error": "invalid_answer"})
                continue
            expected = case["gold"][qid]
            if predicted == expected:
                bucket["correct"] += 1
            else:
                drift.append({"id": record["id"], "question": qid,
                              "expected": expected, "predicted": predicted})
            if predicted in case.get("accepted", {}).get(qid, [expected]):
                accepted_correct += 1
    return {"by_kind": by_kind, "correct": sum(x["correct"] for x in by_kind.values()),
            "accepted_correct": accepted_correct,
            "total": sum(x["total"] for x in by_kind.values()), "request_failures": failures,
            "mismatches": drift}


def _run(model: str, cases: list[dict], questions: dict, key: str | None, workers: int,
         urls: list[str] | None = None, shuffle_seed: int | None = None,
         model_id: str | None = None) -> dict:
    ordered = list(cases)
    random.Random(shuffle_seed if shuffle_seed is not None else 20261006 + workers).shuffle(ordered)
    if urls is not None and not urls:
        raise ValueError("urls must not be empty")

    def send(index_case):
        index, case = index_case
        if urls is None:
            return _request(model, case, questions, key, model_id=model_id)
        replica_index = index % len(urls)
        record = _request(model, case, questions, key, urls[replica_index], model_id=model_id)
        record["replica_index"] = replica_index
        return record

    started = time.perf_counter()
    if workers == 1:
        records = [send(item) for item in enumerate(ordered)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(send, enumerate(ordered)))
    wall_s = time.perf_counter() - started
    succeeded = [record for record in records if record.get("status") == 200]
    latencies = [record["latency_ms"] for record in succeeded]
    return {"concurrency": workers, "n": len(records), "ok": len(succeeded),
            "statuses": {str(status): sum(r.get("status") == status for r in records)
                         for status in sorted({r.get("status") for r in records}, key=str)},
            "wall_s": round(wall_s, 3), "throughput_ok_rps": round(len(succeeded) / wall_s, 3),
            "client_latency_p50_ms": _percentile(latencies, 0.5),
            "client_latency_p95_ms": _percentile(latencies, 0.95),
            "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=URLS, required=True)
    parser.add_argument("--strands-version", choices=STRANDS_IDS, default=None,
                        help="Strands checkpoint version (default: v21)")
    parser.add_argument("--laya-precision", choices=LAYA_IDS, default=None,
                        help="Laya checkpoint precision (default: q8)")
    parser.add_argument("--expect-laya-head-max-len", type=int, default=None,
                        help="verify the Laya server health endpoint before measurement")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--parallel", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--rescore", action="store_true", help="rescore an existing output without API calls")
    parser.add_argument("--append-parallel", action="store_true",
                        help="append only new parallel rounds to a saved sequential run")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip the extra warmup request for local models (Jev never warms up)")
    args = parser.parse_args()
    if args.strands_version and args.model != "strands":
        parser.error("--strands-version applies only to --model strands")
    if (args.laya_precision is not None or args.expect_laya_head_max_len is not None) and args.model != "laya":
        parser.error("Laya precision and head budget apply only to --model laya")
    laya_precision = args.laya_precision or "q8"
    model_id = (STRANDS_IDS[args.strands_version or "v21"] if args.model == "strands"
                else LAYA_IDS[laya_precision] if args.model == "laya"
                else MODEL_IDS[args.model])
    if args.rescore and args.append_parallel:
        parser.error("--rescore and --append-parallel cannot be combined")
    fixture = json.loads(args.fixture.read_text())
    cases, questions = fixture["cases"], fixture["questions"]
    laya_health = (_check_laya_health(laya_precision, args.expect_laya_head_max_len)
                   if args.model == "laya" and args.expect_laya_head_max_len is not None
                   and not args.rescore else None)
    if args.rescore:
        result = json.loads(args.out.read_text())
        if result["model"] != args.model or result.get("model_id_requested") != model_id:
            parser.error("model or Strands version does not match the existing output")
        sequential, parallel = result["sequential"], result["parallel"]
        result["quality"] = _quality(sequential["records"], cases, questions)
    elif args.append_parallel:
        result = json.loads(args.out.read_text())
        if (result["model"] != args.model or result.get("model_id_requested") != model_id
                or result["fixture"] != str(args.fixture.resolve())):
            parser.error("Existing output does not match model, Strands version, and fixture")
        sequential = result["sequential"]
        if sequential["ok"] != len(cases):
            parser.error("Existing sequential run is incomplete")
        requested = [workers for workers in args.parallel if workers > 1]
        completed = {run["concurrency"] for run in result["parallel"]}
        if len(requested) != len(set(requested)) or any(workers in completed for workers in requested):
            parser.error("Parallel concurrency levels must be new and unique")
        key = _key() if args.model == "jev" else None
        if args.model == "jev" and not key:
            parser.error(f"Set TYPESAFE_API_KEY or create {HERE / '.env.local'} (mode 600)")
        parallel = result["parallel"] + [_run(args.model, cases, questions, key, workers, model_id=model_id)
                                         for workers in requested]
        result["parallel"] = parallel
    else:
        key = _key() if args.model == "jev" else None
        if args.model == "jev" and not key:
            parser.error(f"Set TYPESAFE_API_KEY or create {HERE / '.env.local'} (mode 600)")
        # Jev charges for every input token. Its first real request works without
        # initialization, and no benefit from a separate paid warmup is established.
        warmup = None if args.model == "jev" or args.no_warmup else _request(args.model, cases[0], questions, key, model_id=model_id)
        sequential = _run(args.model, cases, questions, key, 1, model_id=model_id)
        parallel = [_run(args.model, cases, questions, key, workers, model_id=model_id)
                    for workers in args.parallel if workers > 1]
        result = {"model": args.model, "model_id_requested": model_id,
                  "laya_precision": laya_precision if args.model == "laya" else None,
                  "server_health": laya_health,
                  "fixture": str(args.fixture.resolve()), "question_count_per_request": len(questions),
                  "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                  "quality": _quality(sequential["records"], cases, questions),
                  "warmup": warmup, "warmup_policy": "none" if warmup is None else "one",
                  "sequential": sequential, "parallel": parallel}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(mode=0o600, exist_ok=True)
    os.chmod(args.out, 0o600)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model": args.model, "quality": result["quality"]["by_kind"],
                      "accepted_correct": result["quality"]["accepted_correct"],
                      "runs": [{k: run[k] for k in ("concurrency", "ok", "n", "throughput_ok_rps",
                                                    "client_latency_p50_ms", "client_latency_p95_ms", "statuses")}
                               for run in [sequential, *parallel]]}, ensure_ascii=False))
    if not args.rescore and args.model == "strands" and any(
        record.get("model_id_match") is not True
        for record in ([result["warmup"]] if result.get("warmup") is not None else [])
        + [record for run in [sequential, *parallel] for record in run["records"]]
    ):
        raise SystemExit("Strands server response model does not match requested checkpoint; raw results were saved")
    if not args.rescore and ((result.get("warmup") is not None and result["warmup"].get("status") != 200)
                             or sequential["ok"] != len(cases)):
        raise SystemExit("Benchmark incomplete: warmup or sequential requests failed; raw results were saved")


if __name__ == "__main__":
    main()
