"""The HTTP batch server must perform one forward for concurrent requests."""

import concurrent.futures
import json
import threading
import unittest
import urllib.request

from batch_server import BatchServer


class BatchServerTests(unittest.TestCase):
    def test_two_requests_share_one_model_call(self):
        calls = []

        def predictor(_agent, packets, *, max_rows):
            calls.append((len(packets), max_rows))
            return [{"model": "fake", "answers": {"q": {"type": "noul", "noul": 1}},
                     "usage": {"input_tokens": 1, "output_tokens": 0}} for _ in packets], 1

        server = BatchServer(
            ("127.0.0.1", 0), agent=object(), predictor=predictor,
            rows_per_request=1, batch_requests=2, batch_wait_ms=100,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        barrier = threading.Barrier(2)

        def send(index):
            barrier.wait()
            payload = json.dumps({"state": str(index), "questions": {"q": {}}}).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/decision", data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.headers["X-Inference-Batch-Size"], json.load(response)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(send, range(2)))
            self.assertEqual(calls, [(2, 2)])
            self.assertEqual([status for status, _, _ in results], [200, 200])
            self.assertEqual([size for _, size, _ in results], ["2", "2"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
