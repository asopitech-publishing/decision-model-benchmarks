"""Offline regression checks for version provenance and score decisions."""

import unittest

from audit_strands_versions import audit
from bench_model_batch import decision


class StrandsVersionTests(unittest.TestCase):
    def test_published_version_comparison_has_distinct_checkpoints(self):
        result = audit()
        self.assertEqual(result["kiro"]["mlx"]["v19"]["accepted"], 20)
        self.assertEqual(result["kiro"]["mlx"]["v21"]["accepted"], 19)
        self.assertEqual(result["support"]["mlx"]["v19"]["correct"], 66)
        self.assertEqual(result["support"]["mlx"]["v21"]["correct"], 63)
        self.assertEqual(len(result["support"]["mlx"]["changed_decisions"]), 3)

    def test_score_decision_uses_modal_level_not_rounded_expected_score(self):
        answer = {"type": "score", "score": 0.51,
                  "probabilities": {"0": 0.49, "1": 0.48, "2": 0.03}}
        self.assertEqual(decision(answer), "0")


if __name__ == "__main__":
    unittest.main()
