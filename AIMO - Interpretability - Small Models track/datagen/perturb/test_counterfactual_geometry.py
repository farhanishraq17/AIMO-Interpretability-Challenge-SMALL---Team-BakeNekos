"""Unit tests for E31 counterfactual geometry helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from counterfactual_geometry import (  # noqa: E402
    counterfactual_signals,
    cosine_distance_curve,
    jensen_shannon_from_logits,
)


class CounterfactualGeometryTest(unittest.TestCase):
    def test_cosine_distance_curve(self):
        anchor = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        variant = np.asarray([[1, 0], [0, -1]], dtype=np.float32)
        np.testing.assert_allclose(cosine_distance_curve(anchor, variant), [0, 2])

    def test_js_is_symmetric_and_zero_for_identity(self):
        a = np.asarray([3.0, 1.0, -2.0])
        b = np.asarray([-1.0, 2.0, 0.0])
        before = a.copy()
        self.assertAlmostEqual(jensen_shannon_from_logits(a, a), 0.0, places=12)
        np.testing.assert_array_equal(a, before)
        self.assertAlmostEqual(
            jensen_shannon_from_logits(a, b), jensen_shannon_from_logits(b, a), places=12
        )
        self.assertGreater(jensen_shannon_from_logits(a, b), 0.0)

    def test_reducer_uses_non_embedding_layers(self):
        anchor = np.tile(np.asarray([[1.0, 0.0]]), (5, 1))
        v1 = anchor.copy()
        v2 = anchor.copy()
        # The embedding layer is ignored. Transformer layers become progressively less aligned.
        v1[1:] = np.asarray([[1, 0], [1, 1], [0, 1], [-1, 0]], dtype=float)
        v2[1:] = v1[1:]
        result = counterfactual_signals(anchor, [v1, v2], np.asarray([1, 0]), [np.asarray([1, 0])] * 2)
        self.assertAlmostEqual(result["cf_cosine_tail_max"], 2.0)
        self.assertGreater(result["cf_cosine_slope"], 0.0)
        self.assertAlmostEqual(result["cf_js_next_mean"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
