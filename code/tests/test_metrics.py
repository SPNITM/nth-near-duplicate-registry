import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nth_registry.metrics import (
    calibrate_max_threshold,
    fixed_gate_operating_curve,
    fixed_gate_score,
    summarize_registry_candidates,
)


class MetricTests(unittest.TestCase):
    def test_calibration_respects_discrete_ties(self):
        negatives = np.array([1] * 2 + [2] * 998)
        threshold = calibrate_max_threshold(negatives, target_fpr=0.001)
        self.assertLess(threshold, 1)

    def test_fixed_gate_cannot_recover_rejected_positive(self):
        nth = np.array([2, 20, 3, 30])
        emb = np.array([0.2, 0.1, 0.8, 0.01])
        score = fixed_gate_score(nth, emb, radius=16)
        self.assertLess(score[1], score[0])
        self.assertLess(score[3], score[2])

    def test_fixed_gate_curve_stops_at_gate_ceiling(self):
        labels = np.array([1, 1, 0, 0], dtype=np.uint8)
        nth = np.array([2, 20, 3, 30])
        emb = np.array([0.2, 0.1, 0.8, 0.01])
        curve = fixed_gate_operating_curve(labels, nth, emb, radius=16)
        self.assertEqual(curve["maximum_recall"], 0.5)
        self.assertEqual(curve["maximum_fpr"], 0.5)
        self.assertEqual(curve["tpr"][-1], 0.5)
        self.assertNotEqual((curve["fpr"][-1], curve["tpr"][-1]), (1.0, 1.0))
        self.assertLessEqual(curve["ap"], curve["maximum_recall"])

    def test_registry_summary_separates_false_candidates(self):
        distances = np.array([[0, 20, 30], [12, 3, 8]])
        summary = summarize_registry_candidates(distances, np.array([0, 1]), radius=8)
        self.assertEqual(summary.mean_shortlist, 1.5)
        self.assertEqual(summary.true_candidate_recall, 1.0)
        self.assertEqual(summary.any_false_candidate_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
