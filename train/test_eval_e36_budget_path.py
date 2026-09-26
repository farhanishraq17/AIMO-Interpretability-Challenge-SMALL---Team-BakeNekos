import math

import pandas as pd

from eval_e36_budget_path import build_examples, normalize_answer, repeated_line_fraction


def _row(problem_id, model, budget, variant, correct, answer, forced=False, trace="work\nanswer"):
    return {
        "problem_id": problem_id,
        "model": model,
        "token_budget": budget,
        "variant": variant,
        "correct_strict": correct,
        "predicted_answer": answer,
        "answer_forced": forced,
        "trace": trace,
    }


def test_normalize_answer():
    assert normalize_answer(" [ $X, Y$ ] ") == "xy"


def test_repeated_line_fraction():
    # `math.isclose` rather than `pytest.approx`: the pinned venv has no pytest, and the repo
    # runs its tests with `unittest discover`.
    assert math.isclose(repeated_line_fraction("a\na\nb"), 1 / 3, rel_tol=1e-9)


def test_build_math_examples_keeps_only_eligible_original():
    rows = []
    for budget in (2048, 4096, 8192, 16384):
        rows.append(_row("p#base", "m", budget, "base", budget == 16384, "[1]"))
        rows.append(_row("p#simple", "m", budget, "simple", True, "[1]"))
        rows.append(_row("p#hard", "m", budget, "hard", True, "[1]"))
    result = build_examples(pd.DataFrame(rows), "math_perturb")
    assert len(result) == 1
    assert bool(result.iloc[0]["is_robust"])
    assert result.iloc[0]["answer_modal_fraction"] == 1.0
