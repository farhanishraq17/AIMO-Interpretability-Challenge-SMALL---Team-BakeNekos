"""E38a prefix-audit tests.

Written for `unittest`, not `pytest`: the project venv is pinned to the competition runtime
(`COMPETITION.md` section 11) and pytest is not one of those packages, so importing it makes the
whole module fail to load under `python -m unittest discover`. The same thing happened to
`test_eval_e36_budget_path.py` and was fixed the same way.
"""

import unittest

import pandas as pd
from audit_e38_prefix import audit_cell


def frame():
    return pd.DataFrame([{"problem_id": "p", "variant": "base", "token_budget": budget,
                          "trace": trace, "temperature": 0, "seed": 42, "answer_forced": i < 3}
                         for i, (budget, trace) in enumerate(zip((2048, 4096, 8192, 16384),
                                                                ("a", "ab", "abc", "abcd")))])


class AuditCell(unittest.TestCase):
    def test_exact_prefixes_and_a_real_divergence(self):
        data = frame()
        self.assertEqual(audit_cell(data, "base")["mismatches"], 0)
        data.loc[1, "trace"] = "a b"  # Whitespace differences must not be hidden.
        result = audit_cell(data, "base")
        self.assertEqual(result["mismatches"], 1)
        self.assertEqual(result["budgets"]["4096"]["forced_mismatches"], 1)

    def test_missing_and_duplicate_budgets_rejected(self):
        data = frame()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit_cell(data.iloc[:-1], "base")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_cell(pd.concat([data, data.iloc[:1]]), "base")

    def test_non_deterministic_settings_rejected(self):
        data = frame()
        data.loc[0, "temperature"] = 1
        with self.assertRaisesRegex(ValueError, "sampling"):
            audit_cell(data, "base")


if __name__ == "__main__":
    unittest.main()
