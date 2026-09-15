"""Unit tests for the perturbation transforms and pool answer parsing.

Run:  python -m unittest discover -s datagen/perturb -v
"""

from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pool import parse_integer  # noqa: E402
from transforms import (  # noqa: E402
    CATEGORIES,
    answer_modulus,
    latex_to_plaintext,
    premise_reorder,
    split_sentences,
)


def rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


class ParseIntegerTest(unittest.TestCase):
    def test_accepted_formats(self):
        self.assertEqual(parse_integer("33"), 33)
        self.assertEqual(parse_integer("142.0"), 142)
        self.assertEqual(parse_integer("['$2$']"), 2)
        self.assertEqual(parse_integer("['5,000']"), 5000)
        self.assertEqual(parse_integer("-7"), -7)

    def test_rejected_formats(self):
        self.assertIsNone(parse_integer("['$\\frac{1}{2}$']"))
        self.assertIsNone(parse_integer("['1', '2']"))
        self.assertIsNone(parse_integer("(1, 2)"))
        self.assertIsNone(parse_integer("2.5"))
        self.assertIsNone(parse_integer(None))


class AnswerModulusTest(unittest.TestCase):
    def test_expected_answer_is_exact_residue(self):
        for answer in (13, 33, 96, 97, 500, 999):
            variant = answer_modulus("Find N.", answer, rng(answer))
            modulus = variant.params["modulus"]
            self.assertLessEqual(modulus, answer)
            self.assertEqual(variant.expected_answer, answer % modulus)
            self.assertIn(f"divided by {modulus}", variant.problem)
            self.assertFalse(variant.needs_verification)

    def test_large_answers_use_large_moduli(self):
        self.assertGreaterEqual(answer_modulus("Find N.", 500, rng()).params["modulus"], 97)

    def test_small_or_negative_answers_skipped(self):
        self.assertIsNone(answer_modulus("Find N.", 5, rng()))
        self.assertIsNone(answer_modulus("Find N.", -40, rng()))


class LatexToPlaintextTest(unittest.TestCase):
    def convert(self, text: str) -> str | None:
        variant = latex_to_plaintext(text, 1, rng())
        return None if variant is None else variant.problem

    def test_fractions_roots_powers(self):
        self.assertEqual(self.convert("Compute $\\frac{1}{2}x + \\sqrt{3}$."), "Compute (1/2)x + sqrt(3).")
        self.assertEqual(self.convert("Find $2^{10}$."), "Find 2^10.")
        # A non-atomic denominator keeps its own parentheses, so precedence cannot be misread.
        self.assertEqual(self.convert("Find $\\frac{x+1}{2^{n}}$."), "Find ((x+1)/(2^n)).")

    def test_sets_degrees_relations(self):
        self.assertEqual(self.convert("Let $S=\\{1,2\\}$."), "Let S={1,2}.")
        self.assertEqual(self.convert("An angle of $30^\\circ$."), "An angle of 30 degrees.")
        self.assertEqual(self.convert("If $a \\leq b$ then"), "If a <= b then")

    def test_unsupported_or_unchanged_returns_none(self):
        self.assertIsNone(self.convert("Find $\\sum_{i=1}^n i$."))
        self.assertIsNone(self.convert("$\\begin{cases} x \\end{cases}$"))
        self.assertIsNone(self.convert("Let $n \\in \\mathbb{Z}$."))
        self.assertIsNone(self.convert("No math here."))
        self.assertIsNone(self.convert("Uses an unknown $\\foo$ command."))

    def test_answer_unchanged(self):
        self.assertEqual(latex_to_plaintext("Find $2^{3}$.", 8, rng()).expected_answer, 8)


class PremiseReorderTest(unittest.TestCase):
    PROBLEM = (
        "A box contains $3$ red balls. A bag contains $5$ blue balls. "
        "A jar contains $7$ green balls. How many balls are there in total?"
    )

    def test_same_premises_question_last(self):
        variant = premise_reorder(self.PROBLEM, 15, rng(1))
        original = split_sentences(self.PROBLEM)
        reordered = split_sentences(variant.problem)
        self.assertEqual(Counter(original), Counter(reordered))
        self.assertEqual(reordered[-1], original[-1])
        self.assertNotEqual(reordered, original)
        self.assertEqual(variant.expected_answer, 15)
        self.assertTrue(variant.needs_verification)

    def test_question_lead_in_stays_with_question(self):
        text = (
            "A box contains $3$ red balls. A bag contains $5$ blue balls. A jar contains $7$ green balls. "
            "Suppose one more ball is added to the jar. How many balls are there in total?"
        )
        for seed in range(5):
            reordered = split_sentences(premise_reorder(text, 16, rng(seed)).problem)
            self.assertEqual(reordered[-2:], split_sentences(text)[-2:])

    def test_figure_code_skipped(self):
        text = "A box has $3$ balls. A bag has $5$ balls. A jar has $7$. [asy] draw((0,0)--(1,1)); [/asy] Find the total."
        self.assertIsNone(premise_reorder(text, 15, rng()))
        self.assertIsNone(latex_to_plaintext("Find $2^{3}$. [asy] draw((0,0)--(1,1)); [/asy]", 8, rng()))

    def test_does_not_split_inside_math(self):
        self.assertEqual(len(split_sentences("Let $x = 1. 5$ be given. Find x.")), 2)

    def test_anaphoric_sentences_stay_put(self):
        text = "A box has $3$ balls. Then it gains $2$. This doubles. How many are there?"
        self.assertIsNone(premise_reorder(text, 10, rng()))

    def test_requires_question_cue_and_single_paragraph(self):
        self.assertIsNone(premise_reorder("A is 1. B is 2. C is 3. Done.", 6, rng()))
        self.assertIsNone(premise_reorder("A is 1.\n\nB is 2. C is 3. Find A+B+C.", 6, rng()))


    def test_answer_form_sentence_stays_with_question(self):
        text = (
            "Let $a$ be the number of red balls. Let $b$ be the number of blue balls. A jar holds $a+b$ balls. "
            "The ratio can be written as $\\frac{m}{n}$, where $m$ and $n$ are relatively prime. Find $m+n$."
        )
        for seed in range(5):
            variant = premise_reorder(text, 7, rng(seed))
            self.assertIsNotNone(variant)
            self.assertEqual(split_sentences(variant.problem)[-2:], split_sentences(text)[-2:])


class RepairEscapesTest(unittest.TestCase):
    def test_restores_unescaped_and_double_escaped_commands(self):
        from pool import repair_escapes

        self.assertEqual(repair_escapes("$s+\x0crac{1}{2}$")[0], "$s+\\frac{1}{2}$")
        self.assertEqual(repair_escapes("$a\times b$")[0], "$a\\times b$")
        self.assertEqual(repair_escapes("$\\\\frac{p}{q}$")[0], "$\\frac{p}{q}$")

    def test_leaves_clean_text_and_real_line_breaks(self):
        from pool import repair_escapes

        self.assertEqual(repair_escapes("no damage"), ("no damage", 0))
        self.assertEqual(repair_escapes("line one\\\\ line two")[0], "line one\\\\ line two")


class RegistryTest(unittest.TestCase):
    def test_every_category_tagged(self):
        for name, category in CATEGORIES.items():
            self.assertIn(category.compliance, {"distinct", "confirm"}, name)
            self.assertTrue(category.rationale)


if __name__ == "__main__":
    unittest.main()
