"""Self-checks for the E22 harness: template isolation and the scoring arithmetic.

Leave-one-template-out is the whole point of the harness (Finding 34), so the one thing worth
pinning down is that a template's siblings never train its own prediction.

Run:  uv run python -m unittest discover -s train -p "test_eval_problem_level.py" -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_problem_level as ev  # noqa: E402


class HarnessTest(unittest.TestCase):
    def test_loto_never_trains_on_the_held_out_template(self) -> None:
        problems = pd.DataFrame({"template": [0, 0, 1, 1, 2, 2], "label": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]})

        def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
            self.assertTrue(set(train["template"]).isdisjoint(set(test["template"])))
            return np.full(len(test), 0.5)

        self.assertTrue(np.all(ev.loto(problems, fit_predict) == 0.5))

    def test_score_recovers_a_planted_signal_and_rejects_noise(self) -> None:
        rng = np.random.default_rng(0)
        y = np.arange(40) % 2 == 0
        groups = np.arange(40) // 2
        planted = ev.score(y, y + rng.normal(scale=0.3, size=40), groups, rng, n_boot=300, n_perm=300)
        self.assertGreater(planted["low"], 0.5)
        self.assertLess(planted["p"], 0.01)
        noise = ev.score(y, rng.normal(size=40), groups, rng, n_boot=300, n_perm=300)
        self.assertLess(noise["low"], 0.5)


if __name__ == "__main__":
    unittest.main()
