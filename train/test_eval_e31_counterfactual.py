"""Small deterministic tests for E31's statistical reductions."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_e31_counterfactual import (
    balanced_accuracy,
    binary_auc,
    within_model_scores,
)


class E31EvaluationTest(unittest.TestCase):
    def test_auc_handles_ties(self):
        self.assertAlmostEqual(binary_auc([False, False, True, True], [0, 1, 1, 2]), 0.875)

    def test_balanced_accuracy(self):
        self.assertAlmostEqual(
            balanced_accuracy([False, False, True, True], [False, True, True, True]), 0.75
        )

    def test_within_model_score_removes_scale_and_orients_low(self):
        frame = pd.DataFrame(
            {
                "model_id": ["a", "a", "b", "b"],
                "signal": [1.0, 2.0, 100.0, 200.0],
            }
        )
        np.testing.assert_allclose(within_model_scores(frame, "signal"), [-0.5, -1.0, -0.5, -1.0])


if __name__ == "__main__":
    unittest.main()
