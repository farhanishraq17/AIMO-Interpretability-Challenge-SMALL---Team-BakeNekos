import pandas as pd

from eval_e37_gsm_confirmation import build_gsm_examples


VARIANTS = (
    "numerical substitution",
    "digit expansion",
    "integer-decimal-fraction conversion",
    "adding operation",
    "reversing operation",
    "problem understanding",
    "distraction insertion",
    "critical thinking",
)


def test_build_gsm_examples_groups_consecutive_ids_and_drops_critical():
    rows = []
    for budget in (2048, 4096, 8192, 16384):
        for offset, variant in enumerate(VARIANTS):
            rows.append({
                "problem_id": str(8 + offset),
                "model": "m",
                "token_budget": budget,
                "variant": variant,
                "correct_strict": variant != "critical thinking",
                "predicted_answer": "[1]",
                "answer_forced": False,
                "trace": "short solution",
            })
    result = build_gsm_examples(pd.DataFrame(rows))
    assert len(result) == 1
    assert result.iloc[0]["problem_id"] == "1"
    assert bool(result.iloc[0]["is_robust"])
