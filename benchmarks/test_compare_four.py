"""Offline checks for raw response capture; never sends a paid API request."""

from __future__ import annotations

import io
import base64
import json
import os
import stat
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import compare_four


class FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, body: bytes):
        super().__init__(body)
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"
        self.headers["X-Request-Id"] = "req-test"
        self.headers["X-Inference-Batch-Size"] = "2"


class RawCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = {"id": "test", "state": "Hello"}
        self.questions = {"ok": {"type": "noul", "instructions": "Is this okay?"}}

    def test_success_preserves_exact_body_and_usage_without_key(self) -> None:
        raw = b'{ "model":"jev-1.13.0", "answers":{"ok":{"noul":0.9}}, "usage":{"input_tokens":123,"output_tokens":1} }'
        with patch.object(compare_four.urllib.request, "urlopen", return_value=FakeResponse(raw)):
            result = compare_four._request("jev", self.case, self.questions, "secret-test-key")
        self.assertEqual(result["raw_response_text"].encode(), raw)
        self.assertEqual(base64.b64decode(result["raw_response_base64"]), raw)
        self.assertEqual(result["usage"]["input_tokens"], 123)
        self.assertEqual(result["response_headers"]["X-Request-Id"], "req-test")
        self.assertEqual(result["inference_batch_size"], 2)
        self.assertNotIn("secret-test-key", json.dumps(result))

    def test_http_error_preserves_body(self) -> None:
        raw = b'{"detail":"rate limited"}'
        error = urllib.error.HTTPError("https://api.typesafe.ai/v1/systemone", 429,
                                       "Too Many Requests", Message(), io.BytesIO(raw))
        with patch.object(compare_four.urllib.request, "urlopen", side_effect=error):
            result = compare_four._request("jev", self.case, self.questions, "secret-test-key")
        self.assertEqual(result["status"], 429)
        self.assertEqual(result["raw_response_text"].encode(), raw)
        self.assertEqual(base64.b64decode(result["raw_response_base64"]), raw)
        self.assertNotIn("answers", result)

    def test_non_utf8_body_is_preserved_byte_for_byte(self) -> None:
        raw = b"\xff\xfe"
        with patch.object(compare_four.urllib.request, "urlopen", return_value=FakeResponse(raw)):
            result = compare_four._request("jev", self.case, self.questions, "secret-test-key")
        self.assertEqual(base64.b64decode(result["raw_response_base64"]), raw)
        self.assertNotIn("answers", result)

    def test_saved_output_includes_raw_response_and_is_private(self) -> None:
        raw = b'{"model":"jev-1.13.0","answers":{"ok":{"noul":0.9}},"usage":{"input_tokens":123}}'
        fixture = {"cases": [{**self.case, "gold": {"ok": True}}], "questions": self.questions}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_path, output_path = root / "fixture.json", root / "results.json"
            fixture_path.write_text(json.dumps(fixture))
            argv = ["compare_four.py", "--model", "jev", "--fixture", str(fixture_path),
                    "--out", str(output_path), "--parallel"]
            with patch.object(sys, "argv", argv), patch.object(compare_four, "_key", return_value="secret-test-key"), \
                    patch.object(compare_four.urllib.request, "urlopen", side_effect=lambda *_args, **_kwargs: FakeResponse(raw)), \
                    redirect_stdout(io.StringIO()):
                compare_four.main()
            saved = json.loads(output_path.read_text())
            self.assertIsNone(saved["warmup"])
            self.assertEqual(base64.b64decode(saved["sequential"]["records"][0]["raw_response_base64"]), raw)
            self.assertEqual(stat.S_IMODE(os.stat(output_path).st_mode), 0o600)
            self.assertNotIn("secret-test-key", output_path.read_text())
            append_argv = ["compare_four.py", "--model", "jev", "--fixture", str(fixture_path),
                           "--out", str(output_path), "--append-parallel", "--parallel", "2"]
            with patch.object(sys, "argv", append_argv), patch.object(compare_four, "_key", return_value="secret-test-key"), \
                    patch.object(compare_four.urllib.request, "urlopen", side_effect=lambda *_args, **_kwargs: FakeResponse(raw)), \
                    redirect_stdout(io.StringIO()):
                compare_four.main()
            appended = json.loads(output_path.read_text())
            self.assertEqual([round_["concurrency"] for round_ in appended["parallel"]], [2])
            self.assertEqual(base64.b64decode(appended["parallel"][0]["records"][0]["raw_response_base64"]), raw)
            self.assertEqual(appended["sequential"]["records"], saved["sequential"]["records"])

    def test_failed_run_keeps_diagnostics_but_exits_nonzero(self) -> None:
        fixture = {"cases": [{**self.case, "gold": {"ok": True}}], "questions": self.questions}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_path, output_path = root / "fixture.json", root / "results.json"
            fixture_path.write_text(json.dumps(fixture))
            argv = ["compare_four.py", "--model", "jev", "--fixture", str(fixture_path),
                    "--out", str(output_path), "--parallel"]
            with patch.object(sys, "argv", argv), patch.object(compare_four, "_key", return_value="secret-test-key"), \
                    patch.object(compare_four.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    compare_four.main()
            saved = json.loads(output_path.read_text())
            self.assertIsNone(saved["warmup"])
            self.assertEqual(saved["sequential"]["ok"], 0)

    def test_jev_default_sends_only_scored_requests(self) -> None:
        raw = b'{"model":"jev-1.13.0","answers":{"ok":{"noul":0.9}},"usage":{"input_tokens":123}}'
        fixture = {"cases": [{**self.case, "gold": {"ok": True}}], "questions": self.questions}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_path, output_path = root / "fixture.json", root / "results.json"
            fixture_path.write_text(json.dumps(fixture))
            argv = ["compare_four.py", "--model", "jev", "--fixture", str(fixture_path),
                    "--out", str(output_path), "--parallel"]
            with patch.object(sys, "argv", argv), patch.object(compare_four, "_key", return_value="secret-test-key"), \
                    patch.object(compare_four.urllib.request, "urlopen",
                                 side_effect=lambda *_args, **_kwargs: FakeResponse(raw)) as sender, \
                    redirect_stdout(io.StringIO()):
                compare_four.main()
            saved = json.loads(output_path.read_text())
            self.assertEqual(sender.call_count, 1)
            self.assertIsNone(saved["warmup"])
            self.assertEqual(saved["warmup_policy"], "none")

    def test_local_default_still_warms_up(self) -> None:
        raw = b'{"model":"strands","answers":{"ok":{"noul":0.9}},"usage":{"input_tokens":123}}'
        fixture = {"cases": [{**self.case, "gold": {"ok": True}}], "questions": self.questions}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_path, output_path = root / "fixture.json", root / "results.json"
            fixture_path.write_text(json.dumps(fixture))
            argv = ["compare_four.py", "--model", "strands", "--fixture", str(fixture_path),
                    "--out", str(output_path), "--parallel"]
            with patch.object(sys, "argv", argv), \
                    patch.object(compare_four.urllib.request, "urlopen",
                                 side_effect=lambda *_args, **_kwargs: FakeResponse(raw)) as sender, \
                    redirect_stdout(io.StringIO()):
                compare_four.main()
            saved = json.loads(output_path.read_text())
            self.assertEqual(sender.call_count, 2)
            self.assertEqual(saved["warmup_policy"], "one")
            self.assertEqual(base64.b64decode(saved["warmup"]["raw_response_base64"]), raw)

    def test_replica_run_routes_to_distinct_endpoints(self) -> None:
        cases = [{"id": str(index), "state": "hello"} for index in range(4)]
        urls = ["http://127.0.0.1:9001/v1/systemone", "http://127.0.0.1:9002/v1/systemone"]

        def fake_request(_model, case, _questions, _key, url=None):
            return {"id": case["id"], "status": 200, "latency_ms": 1.0,
                    "inference_batch_size": 1, "endpoint": url}

        with patch.object(compare_four, "_request", side_effect=fake_request):
            run = compare_four._run("strands", cases, self.questions, None, 2, urls=urls)
        self.assertEqual(run["ok"], 4)
        self.assertEqual([record["replica_index"] for record in run["records"]], [0, 1, 0, 1])
        self.assertEqual([record["endpoint"] for record in run["records"]], [urls[0], urls[1], urls[0], urls[1]])


if __name__ == "__main__":
    unittest.main()
