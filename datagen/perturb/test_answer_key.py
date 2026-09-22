"""The answer key accepts only unanimous, parsed answers; anything less must stay unresolved."""

from __future__ import annotations

import unittest

from answer_key import consensus


class ConsensusTest(unittest.TestCase):
    def test_unanimous_answer_is_kept(self) -> None:
        self.assertEqual(consensus([204] * 8), 204)

    def test_any_disagreement_or_missing_answer_rejects(self) -> None:
        self.assertIsNone(consensus([204] * 7 + [205]))
        self.assertIsNone(consensus([204] * 7 + [None]))
        self.assertIsNone(consensus([None] * 8))


if __name__ == "__main__":
    unittest.main()
