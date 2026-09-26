"""Unit tests for the E14 test-time agreement helpers (`datagen/extract_agreement.py`).

Run:  python -m unittest discover -s datagen/perturb -v
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_agreement import final_answer, rewrites, signals, typos  # noqa: E402

PROBLEM = "Let $x^2 = 16$ and suppose everything positive holds. Compute the value of $x + 1$."


class AgreementTest(unittest.TestCase):
    def test_final_answer(self):
        self.assertEqual(final_answer("so \\boxed{1,024}"), "1024")
        self.assertEqual(final_answer("<think>\\boxed{3}</think> thus \\boxed{\\dfrac{1}{ 2}}"), "\\frac{1}{2}")
        self.assertIsNone(final_answer("no answer"))

    def test_typos_leave_math_alone(self):
        out = typos(PROBLEM, random.Random(0))
        self.assertNotEqual(out, PROBLEM)
        self.assertIn("$x^2 = 16$", out)
        self.assertIsNone(typos("Find $x$.", random.Random(0)))

    def test_rewrites_are_deterministic_and_different(self):
        first = rewrites(PROBLEM)
        self.assertEqual(first, rewrites(PROBLEM))
        self.assertIn("distract", first)
        self.assertTrue(all(text != PROBLEM for text in first.values()))

    def test_signals(self):
        out = {
            "original": {"answers": ["5", "5"], "tokens": [10, 30], "hit": [False, False]},
            "typos": {"answers": ["5", "4"], "tokens": [20, 20], "hit": [False, True]},
            "distract": {"answers": [None, "5"], "tokens": [40, 40], "hit": [False, False]},
        }
        s = signals(out)
        self.assertEqual(s["orig_self_agree"], 1.0)
        self.assertEqual(s["rewrite_agree"], 0.5)  # an unanswered rewrite counts as a change
        self.assertEqual(s["rewrite_min_agree"], 0.5)
        self.assertEqual(s["rewrite_token_ratio"], 1.5)
        self.assertAlmostEqual(s["hit_length_frac"], 1 / 6)
        no_anchor = signals({**out, "original": {"answers": [None, None], "tokens": [1, 1], "hit": [True, True]}})
        self.assertIsNone(no_anchor["rewrite_agree"])


if __name__ == "__main__":
    unittest.main()
