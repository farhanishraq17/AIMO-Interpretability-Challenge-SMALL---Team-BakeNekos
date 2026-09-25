"""Synthetic contract tests for E35's frozen external-label construction."""

from __future__ import annotations

import unittest

import pandas as pd

from eval_e35_external import BUDGET, derive_labels


def row(problem: str, model: str, variant: str, correct: bool) -> dict:
    return {
        "model": model,
        "token_budget": BUDGET,
        "problem_id": problem,
        "question": f"question {problem} {variant}",
        "variant": variant,
        "correct_strict": correct,
    }


class E35LabelTest(unittest.TestCase):
    def test_math_clear_cut_and_ambiguous(self) -> None:
        rows = []
        patterns = {
            "robust": [True, True, True, False],
            "spurious": [False, False, False, True],
            "ambiguous": [True, True, False, False],
        }
        for problem, outcomes in patterns.items():
            for index, outcome in enumerate(outcomes):
                model = f"m{index}"
                rows.extend(
                    [
                        row(problem, model, "base", True),
                        row(problem, model, "simple", outcome),
                        row(problem, model, "hard", outcome),
                    ]
                )
        frame = pd.DataFrame(rows)
        # Original wording must be identical across model configurations.
        frame.loc[frame["variant"] == "base", "question"] = frame.loc[
            frame["variant"] == "base", "problem_id"
        ].map(lambda value: f"question {value}")
        labels = derive_labels(frame, "math_perturb").set_index("problem_id")
        self.assertEqual(set(labels.index), {"robust", "spurious"})
        self.assertTrue(bool(labels.loc["robust", "is_robust"]))
        self.assertFalse(bool(labels.loc["spurious", "is_robust"]))

    def test_putnam_requires_original_and_variation(self) -> None:
        rows = []
        for model in ("m0", "m1"):
            rows.extend([row("p", model, "original", True), row("p", model, "variation", True)])
        frame = pd.DataFrame(rows)
        frame.loc[frame["variant"] == "original", "question"] = "same original"
        labels = derive_labels(frame, "putnam_axiom")
        self.assertEqual(len(labels), 1)
        self.assertTrue(bool(labels.iloc[0]["is_robust"]))


if __name__ == "__main__":
    unittest.main()
