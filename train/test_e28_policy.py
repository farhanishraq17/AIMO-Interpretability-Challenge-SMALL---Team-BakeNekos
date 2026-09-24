from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e28_solution", ROOT / "submission" / "e28-familiarity" / "solution.py")
assert SPEC and SPEC.loader
SOLUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLUTION)


class E28PolicyTest(unittest.TestCase):
    def test_familiar_half_preserves_order_and_uses_ceil_half(self) -> None:
        self.assertEqual(SOLUTION.familiar_half([3.0, 1.0, 4.0, 2.0, 0.0]), [False, True, False, True, True])

    def test_rank_ensemble_is_scale_free(self) -> None:
        records = [
            {"mean_nll": 1.0, "suffix_nll": 30.0, "min_k_10": 1.0, "min_k_20": 0.0},
            {"mean_nll": 2.0, "suffix_nll": 10.0, "min_k_10": 1.0, "min_k_20": 0.0},
            {"mean_nll": 3.0, "suffix_nll": 20.0, "min_k_10": 1.0, "min_k_20": 0.0},
        ]
        old = SOLUTION.SCORE_MODE
        try:
            SOLUTION.SCORE_MODE = "mean_suffix"
            self.assertEqual(SOLUTION.combine_scores(records), [1.0, 0.5, 1.5])
        finally:
            SOLUTION.SCORE_MODE = old

    def test_non_split_model_uses_native_bool_constants(self) -> None:
        old = SOLUTION.SPLIT_MODELS
        try:
            SOLUTION.SPLIT_MODELS = "Qwen/Qwen3.5-4B"
            prediction = SOLUTION.are_robust("Skywork/Skywork-OR1-Math-7B", ["a", "b"])
            self.assertEqual(prediction, [False, False])
            self.assertTrue(all(type(value) is bool for value in prediction))
        finally:
            SOLUTION.SPLIT_MODELS = old

    def test_unknown_model_falls_back_without_loading(self) -> None:
        prediction = SOLUTION.are_robust("closed/model:low", ["a", "b"])
        self.assertEqual(prediction, [False, False])

    def test_deepseek_only_policy_scores_deepseek_and_keeps_other_constants(self) -> None:
        old_split = SOLUTION.SPLIT_MODELS
        old_score = SOLUTION.SCORE_MODE
        try:
            SOLUTION.SPLIT_MODELS = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
            SOLUTION.SCORE_MODE = "min_k_gap"
            records = [
                {"mean_nll": 0.0, "suffix_nll": 0.0, "min_k_10": 3.0, "min_k_20": 1.0},
                {"mean_nll": 0.0, "suffix_nll": 0.0, "min_k_10": 1.5, "min_k_20": 1.0},
                {"mean_nll": 0.0, "suffix_nll": 0.0, "min_k_10": 2.0, "min_k_20": 1.0},
            ]
            with patch.object(SOLUTION, "problem_scores", return_value=records) as scorer:
                deepseek = SOLUTION.are_robust(SOLUTION.SPLIT_MODELS, ["a", "b", "c"])
                qwen = SOLUTION.are_robust("Qwen/Qwen3.5-4B", ["a", "b", "c"])
            self.assertEqual(deepseek, [False, True, True])
            self.assertEqual(qwen, [True, True, True])
            scorer.assert_called_once()
        finally:
            SOLUTION.SPLIT_MODELS = old_split
            SOLUTION.SCORE_MODE = old_score


if __name__ == "__main__":
    unittest.main()
