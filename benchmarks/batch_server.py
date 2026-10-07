"""Loopback-only HTTP server that dynamically batches requests for one model.

One worker owns one model instance. Concurrent HTTP requests enter a bounded
queue; the worker groups them and performs one model forward per group.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bench_model_batch import load_backend


@dataclass
class Pending:
    packet: dict
    done: threading.Event = field(default_factory=threading.Event)
    response: dict | None = None
    error: str | None = None
    batch_size: int = 0


class BatchServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address, *, agent, predictor, rows_per_request: int,
                 batch_requests: int, batch_wait_ms: float, max_pending: int = 128,
                 max_questions_per_request: int = 3):
        if (batch_requests < 1 or batch_wait_ms < 0 or max_pending < batch_requests
                or max_questions_per_request < 1):
            raise ValueError("invalid batch or queue settings")
        self.agent = agent
        self.predictor = predictor
        self.rows_per_request = rows_per_request
        self.max_questions_per_request = max_questions_per_request
        self.batch_requests = batch_requests
        self.batch_wait_s = batch_wait_ms / 1000
        self.pending: queue.Queue[Pending] = queue.Queue(maxsize=max_pending)
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self._work, daemon=True)
        super().__init__(address, _handler())
        self.worker.start()

    def _work(self):
        while not self.stopping.is_set():
            try:
                first = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            group = [first]
            deadline = time.perf_counter() + self.batch_wait_s
            while len(group) < self.batch_requests:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    group.append(self.pending.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                responses, calls = self.predictor(
                    self.agent, [item.packet for item in group],
                    max_rows=len(group) * self.rows_per_request,
                )
                if calls != 1 or len(responses) != len(group):
                    raise RuntimeError("the model did not process the group in one forward pass")
                for item, response in zip(group, responses, strict=True):
                    item.response = response
                    item.batch_size = len(group)
            except Exception:
                for item in group:
                    item.error = "inference_failed"
            finally:
                for item in group:
                    item.done.set()
                    self.pending.task_done()

    def server_close(self):
        self.stopping.set()
        self.worker.join(timeout=2)
        super().server_close()


def _handler():
    class Handler(BaseHTTPRequestHandler):
        server: BatchServer

        def _send(self, status, payload, batch_size=None):
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if batch_size is not None:
                self.send_header("X-Inference-Batch-Size", str(batch_size))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/healthz":
                self._send(404, {"error": "not_found"})
                return
            self._send(200, {"status": "ok", "model_loaded": True,
                             "model_instances": 1,
                             "max_batch_requests": self.server.batch_requests,
                             "pending": self.server.pending.qsize()})

        def do_POST(self):
            if self.path not in {"/v1/systemone", "/v1/decision"}:
                self._send(404, {"error": "not_found"})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                self._send(415, {"error": "unsupported_media_type"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 1 or length > 2 * 1024 * 1024:
                self._send(413, {"error": "invalid_body_length"})
                return
            try:
                packet = json.loads(self.rfile.read(length))
                if not isinstance(packet, dict) or "state" not in packet or not packet.get("questions"):
                    raise ValueError("invalid packet")
                if (not isinstance(packet["questions"], dict)
                        or len(packet["questions"]) > self.server.max_questions_per_request):
                    raise ValueError("too many questions")
            except (ValueError, UnicodeDecodeError):
                self._send(400, {"error": "invalid_request"})
                return
            pending = Pending(packet)
            try:
                self.server.pending.put_nowait(pending)
            except queue.Full:
                self._send(503, {"error": "queue_full"})
                return
            if not pending.done.wait(timeout=90):
                self._send(504, {"error": "inference_timeout"})
            elif pending.error:
                self._send(500, {"error": pending.error})
            else:
                self._send(200, pending.response, pending.batch_size)

        def log_message(self, fmt, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=["laya", "strands", "clef"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--batch-requests", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=2)
    parser.add_argument("--max-pending", type=int, default=128)
    parser.add_argument("--max-questions-per-request", type=int, default=3)
    args = parser.parse_args()
    agent, _, predictor, rows_per_request = load_backend(
        args.model_kind, args.model, args.batch_requests, args.max_questions_per_request
    )
    server = BatchServer(
        ("127.0.0.1", args.port), agent=agent, predictor=predictor,
        rows_per_request=rows_per_request, batch_requests=args.batch_requests,
        batch_wait_ms=args.batch_wait_ms, max_pending=args.max_pending,
        max_questions_per_request=args.max_questions_per_request,
    )
    print(json.dumps({"port": args.port, "model": args.model_kind,
                      "model_instances": 1, "max_batch_requests": args.batch_requests}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
