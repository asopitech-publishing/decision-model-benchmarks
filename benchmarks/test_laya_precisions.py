"""Offline regression checks for the published FP16/q8 comparison."""

import unittest
from pathlib import Path

from audit_laya_precisions import audit


ROOT = Path(__file__).resolve().parents[1]


class LayaPrecisionResultsTests(unittest.TestCase):
    def test_raw_results_and_scoring_audit(self):
        summary = audit(ROOT / "results/2026-10-07", ROOT / "benchmarks")
        self.assertEqual((summary["kiro"]["fp16_accepted"],
                          summary["kiro"]["q8_accepted"]), (9, 9))
        self.assertEqual(summary["kiro"]["changed_decisions"], [])
        self.assertEqual((summary["support"]["fp16_exact"],
                          summary["support"]["q8_exact"]), (56, 58))
        self.assertEqual(len(summary["support"]["changed_decisions"]), 3)


if __name__ == "__main__":
    unittest.main()
