"""Checks for E34a's frozen feasibility gate."""

from __future__ import annotations

import unittest

import pandas as pd

from train.eval_e34_yield import evaluate


class E34YieldGateTest(unittest.TestCase):
    @staticmethod
    def frames(n24: int = 8, n26: int = 5):
        records = []
        rows = []
        for source, n in (("aime-2024", n24), ("aime-2026", n26)):
            for index in range(n):
                key = f"{source}-{index}"
                records.append({
                    "problem_key": key,
                    "dataset_id": source,
                    "is_eligible": True,
                    "is_clear_cut": True,
                    "is_robust": index % 2 == 0,
                })
                for category in ("latex_to_plaintext", "premise_reorder"):
                    rows.append({"problem_key": key, "perturbation_key": category})
        return pd.DataFrame(records), pd.DataFrame(rows)

    def test_passes_exact_frozen_minima(self):
        instances, rows = self.frames()
        result = evaluate(instances, rows)
        self.assertTrue(result["passed"])
        self.assertEqual(result["clear_cut_per_source"], {"aime-2024": 8, "aime-2026": 5})

    def test_rejects_single_class_holdout(self):
        instances, rows = self.frames()
        instances.loc[instances["dataset_id"] == "aime-2026", "is_robust"] = True
        result = evaluate(instances, rows)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["both_classes_in_each_source"])

    def test_rejects_missing_second_family(self):
        instances, rows = self.frames()
        rows = rows[rows["perturbation_key"] == "latex_to_plaintext"]
        result = evaluate(instances, rows)
        self.assertFalse(result["checks"]["each_category_has_at_least_4_rows"])


if __name__ == "__main__":
    unittest.main()
