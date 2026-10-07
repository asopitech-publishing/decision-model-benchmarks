"""Measure concurrent HTTP latency with 1, 2, or 4 separate local model processes.

Each process owns exactly one model and handles one inference at a time. The
client sends requests directly to replicas in round-robin order; no inference
batching or external API is involved. Full HTTP responses are retained.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from compare_four import STRANDS_IDS, _quality, _request, _run


ROUTES = {"laya": "/v1/decision", "strands": "/v1/systemone", "clef": "/v1/systemone"}


def _free_port(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def _wait_healthy(process: subprocess.Popen, port: int, timeout_s: float,
                  expected_laya_dtype=None, expected_head_max_len=None) -> None:
    url = f"http://127.0.0.1:{port}/healthz"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"replica on port {port} exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                data = json.load(response)
            if (data["model_instances"] == 1 and data["max_batch_requests"] == 1
                    and (expected_laya_dtype is None or data.get("laya_dtype") == expected_laya_dtype)
                    and (expected_head_max_len is None
                         or data.get("head_max_len") == expected_head_max_len)):
                return
            raise RuntimeError("replica model, precision, head budget, or batch size mismatch")
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.25)
    raise TimeoutError(f"replica on port {port} did not become healthy")


def _stop(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    os.chmod(path, 0o600)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=ROUTES, required=True)
    parser.add_argument("--python", type=Path, required=True, help="Python executable with model dependencies")
    parser.add_argument("--model", required=True, help="local model directory or cached model id")
    parser.add_argument("--source", type=Path, help="optional model runtime source root for PYTHONPATH")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port-base", type=int, default=8120)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--laya-dtype", choices=["float16", "int8_mixed"], default="int8_mixed")
    parser.add_argument("--laya-head-max-len", type=int)
    args = parser.parse_args()
    if args.model_kind != "laya" and (args.laya_dtype != "int8_mixed" or args.laya_head_max_len is not None):
        parser.error("Laya precision and head budget apply only to --model-kind laya")
    model_id = None
    if args.model_kind == "strands":
        model_id = args.model.rstrip("/").split("/")[-1]
        if model_id not in STRANDS_IDS.values():
            parser.error("Strands --model must identify a supported v19 or v21 checkpoint")
    if (not args.python.is_file() or args.repeats < 1 or args.startup_timeout <= 0
            or any(value < 1 for value in args.replicas + args.concurrency)
            or len(set(args.replicas)) != len(args.replicas)
            or len(set(args.concurrency)) != len(args.concurrency)
            or args.port_base < 1024 or args.port_base + max(args.replicas) > 65535):
        parser.error("invalid executable, replica count, concurrency, timeout, or port range")
    if args.out.exists():
        parser.error("output already exists; choose a new path")
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    cases, questions = fixture["cases"], fixture["questions"]
    if not cases or not questions or len({case["id"] for case in cases}) != len(cases):
        parser.error("fixture needs nonempty questions and uniquely identified cases")
    if len(questions) > 3:
        parser.error("this run supports at most three questions per request")

    result = {
        "model": args.model_kind,
        "model_snapshot": args.model.rstrip("/").split("/")[-1],
        "model_id_requested": model_id,
        "fixture": f"benchmarks/{args.fixture.name}",
        "model_instances": args.replicas,
        "concurrency_levels": args.concurrency,
        "repeats": args.repeats,
        "routing": "client round robin across loopback-only independent processes",
        "server_batch_requests": 1,
        "cache_prompts": False,
        "laya_dtype": args.laya_dtype if args.model_kind == "laya" else None,
        "head_max_len": args.laya_head_max_len if args.model_kind == "laya" else None,
        "warmup_policy": "one untimed local request per replica",
        "groups": [],
    }
    # Keep the venv/bin/python symlink path; resolving it would escape the venv.
    python = str(args.python.absolute())
    server_file = str(Path(__file__).with_name("batch_server.py"))
    env = {key: value for key, value in os.environ.items()
           if key not in {"TYPESAFE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
    python_paths = [str(Path(__file__).parent)]
    if args.source:
        python_paths.append(str(args.source.resolve()))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    for count in args.replicas:
        ports = [args.port_base + index for index in range(count)]
        if any(not _free_port(port) for port in ports):
            raise RuntimeError("a requested replica port is already in use")
        processes = []
        with tempfile.TemporaryFile(mode="w+t") as log:
            try:
                for port in ports:
                    command = [python, server_file, "--model-kind", args.model_kind,
                               "--model", args.model, "--port", str(port),
                               "--batch-requests", "1", "--batch-wait-ms", "0"]
                    if args.model_kind == "laya":
                        command += ["--laya-dtype", args.laya_dtype]
                        if args.laya_head_max_len is not None:
                            command += ["--laya-head-max-len", str(args.laya_head_max_len)]
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    processes.append(process)
                    _wait_healthy(
                        process, port, args.startup_timeout,
                        expected_laya_dtype=args.laya_dtype if args.model_kind == "laya" else None,
                        expected_head_max_len=args.laya_head_max_len if args.model_kind == "laya" else None,
                    )
                    print(f"{args.model_kind}: replica {len(processes)}/{count} ready", flush=True)
                urls = [f"http://127.0.0.1:{port}{ROUTES[args.model_kind]}" for port in ports]
                warmups = []
                for index, url in enumerate(urls):
                    record = _request(args.model_kind, cases[0], questions, None, url, model_id=model_id)
                    record["replica_index"] = index
                    if (record.get("status") != 200 or record.get("inference_batch_size") != 1
                            or (model_id and record.get("model_id_match") is not True)):
                        raise RuntimeError("local replica warmup failed")
                    warmups.append(record)
                group = {"replicas": count, "warmups": warmups, "runs": []}
                for trial in range(args.repeats):
                    levels = list(args.concurrency)
                    random.Random(20261007 + trial * 31).shuffle(levels)
                    for workers in levels:
                        run = _run(args.model_kind, cases, questions, None, workers, urls=urls,
                                   shuffle_seed=20261007 + trial * 37 + workers, model_id=model_id)
                        if (run["ok"] != len(cases) or any(
                            record.get("inference_batch_size") != 1
                            or (model_id and record.get("model_id_match") is not True)
                            for record in run["records"]
                        )):
                            raise RuntimeError("replica run had failed requests or implicit batching")
                        run["trial"] = trial + 1
                        run["quality"] = _quality(run["records"], cases, questions)
                        group["runs"].append(run)
                        print(json.dumps({"model": args.model_kind, "replicas": count,
                                          "trial": trial + 1, "concurrency": workers,
                                          "ok": run["ok"], "p50_ms": run["client_latency_p50_ms"],
                                          "p95_ms": run["client_latency_p95_ms"],
                                          "rps": run["throughput_ok_rps"]}), flush=True)
                result["groups"].append(group)
                _save(args.out, result)
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-4000:], flush=True)
                raise
            finally:
                _stop(processes)


if __name__ == "__main__":
    main()
