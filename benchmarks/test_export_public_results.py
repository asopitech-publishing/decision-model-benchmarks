"""Offline checks for the public-results privacy and integrity gate."""

import base64
import hashlib
import unittest

import export_public_results as exporter


def sample_record(body=b'{"ok":true}'):
    return {
        "raw_response_base64": base64.b64encode(body).decode("ascii"),
        "raw_response_sha256": hashlib.sha256(body).hexdigest(),
        "raw_response_text": body.decode(),
        "response_headers": {"x-typesafe-request-id": "private-id"},
    }


class PublicExportTests(unittest.TestCase):
    def test_headers_removed_and_body_preserved(self):
        raw = {"warmup": sample_record()}
        public = exporter.scrub(raw)
        self.assertNotIn("response_headers", public["warmup"])
        self.assertEqual(exporter.verify(raw, public, "sample.json"), 1)

    def test_local_path_rejected(self):
        raw = {"warmup": sample_record()}
        public = exporter.scrub(raw)
        public["fixture"] = "/" + "Users/person/private/file.json"
        with self.assertRaises(ValueError):
            exporter.verify(raw, public, "sample.json")

    def test_secret_inside_encoded_body_rejected(self):
        raw = {"warmup": sample_record(b'{"token":"Bearer ' + b'example-secret-value"}')}
        public = exporter.scrub(raw)
        with self.assertRaises(ValueError):
            exporter.verify(raw, public, "sample.json")


if __name__ == "__main__":
    unittest.main()
