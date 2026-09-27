import pandas as pd
import pytest
from audit_e38_prefix import audit_cell


def frame():
    return pd.DataFrame([{"problem_id": "p", "variant": "base", "token_budget": budget,
                          "trace": trace, "temperature": 0, "seed": 42, "answer_forced": i < 3}
                         for i, (budget, trace) in enumerate(zip((2048, 4096, 8192, 16384),
                                                                ("a", "ab", "abc", "abcd")))])


def test_exact_prefixes_and_a_real_divergence():
    data = frame()
    assert audit_cell(data, "base")["mismatches"] == 0
    data.loc[1, "trace"] = "a b"  # Whitespace differences must not be hidden.
    result = audit_cell(data, "base")
    assert result["mismatches"] == 1
    assert result["budgets"]["4096"]["forced_mismatches"] == 1


def test_missing_and_duplicate_budgets_rejected():
    data = frame()
    with pytest.raises(ValueError, match="incomplete"):
        audit_cell(data.iloc[:-1], "base")
    with pytest.raises(ValueError, match="duplicate"):
        audit_cell(pd.concat([data, data.iloc[:1]]), "base")


def test_non_deterministic_settings_rejected():
    data = frame()
    data.loc[0, "temperature"] = 1
    with pytest.raises(ValueError, match="sampling"):
        audit_cell(data, "base")
