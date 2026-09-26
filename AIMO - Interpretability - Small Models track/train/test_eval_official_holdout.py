"""Self-checks for the cross-model holdout, where the arithmetic is easy to get wrong.

The measured E21 numbers sit at chance, and a scoring bug looks exactly like that, so the
one thing worth testing is that a planted signal does come back.

Run:  uv run python -m unittest train.test_eval_official_holdout -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_official_holdout as ev  # noqa: E402


def frame(n: int, offset: float, seed: int) -> pd.DataFrame:
    """Half robust, half spurious; entropy separates them, shifted per model by `offset`."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2 == 0
    columns = {c: rng.normal(size=n) for c in ev.UNCERTAINTY_COLUMNS}
    columns["next_token_entropy"] = offset + np.where(y, -2.0, 2.0) + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"is_robust": y, "dataset_id": ev.OFFICIAL_DATASET, **columns})


class CrossModelTest(unittest.TestCase):
    def test_planted_signal_survives_a_held_out_model(self) -> None:
        frames = {f"m{i}/model": frame(20, offset=10.0 * i, seed=i) for i in range(3)}
        rows = [r for r in ev.cross_model(frames) if "base rate" not in r]
        self.assertEqual(len(rows), 3)
        for row in rows:
            balanced_accuracy = float(row.split("|")[4])
            # Per-model standardisation is what removes the offset; without it this is chance.
            self.assertGreater(balanced_accuracy, 0.9, row)

    def test_single_row_model_still_scores(self) -> None:
        """One row has an undefined standard deviation; the guard must keep it finite."""
        frames = {"a/model": frame(20, 0.0, 1), "b/model": frame(20, 5.0, 2).head(1)}
        rows = [r for r in ev.cross_model(frames) if "base rate" not in r]
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(np.isfinite(float(row.split("|")[4])), row)


if __name__ == "__main__":
    unittest.main()
