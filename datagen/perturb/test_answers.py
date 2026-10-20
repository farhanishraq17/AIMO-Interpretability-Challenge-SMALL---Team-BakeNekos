"""Unit tests for final-answer extraction.

Run:  python -m unittest discover -s datagen/perturb -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answers import extract_answer, last_boxed, normalize_integer  # noqa: E402


class LastBoxedTest(unittest.TestCase):
    def test_nested_braces_and_last_occurrence(self):
        self.assertEqual(last_boxed("\\boxed{1} then \\boxed{\\frac{3}{4}}"), "\\frac{3}{4}")
        self.assertEqual(last_boxed("answer \\fbox{12}"), "12")

    def test_missing_or_unbalanced(self):
        self.assertIsNone(last_boxed("no box"))
        self.assertIsNone(last_boxed("\\boxed{12"))


class NormalizeIntegerTest(unittest.TestCase):
    def test_accepted(self):
        for raw, expected in [
            ("42", 42), ("$1{,}024$", 1024), ("1,024", 1024), ("\\text{7}", 7),
            ("n = 7", 7), ("0.0", 0), ("-3", -3), ("30^\\circ", 30), ("45.", 45),
        ]:
            self.assertEqual(normalize_integer(raw), expected, raw)

    def test_rejected(self):
        for raw in ("\\frac{3}{4}", "2.5", "x+1", "", None, "(1, 2)"):
            self.assertIsNone(normalize_integer(raw), raw)


class ExtractAnswerTest(unittest.TestCase):
    def test_prefers_box_after_reasoning(self):
        text = "<think>try \\boxed{5} ... no, recompute</think>The answer is \\boxed{204}."
        self.assertEqual(extract_answer(text), 204)

    def test_box_inside_reasoning_when_no_conclusion(self):
        self.assertEqual(extract_answer("<think>so \\boxed{17}</think>"), 17)

    def test_fallback_formats(self):
        self.assertEqual(extract_answer("work...\nANSWER: 33"), 33)
        self.assertEqual(extract_answer("Thus the final answer is 113."), 113)

    def test_no_answer(self):
        self.assertIsNone(extract_answer("I could not finish the"))
        self.assertIsNone(extract_answer(""))


if __name__ == "__main__":
    unittest.main()
