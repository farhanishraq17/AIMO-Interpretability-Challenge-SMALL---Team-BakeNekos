"""The E61 submission must score problems on the scale its thresholds came from.

E61's thresholds are quantiles of `mean_nll` from `datagen/extract_familiarity.familiarity` (E55). The
submission recomputes the statistic in `submission/e61-familiarity/solution.py` without that module. This
runs both on the tiny cached test model and asserts the values match exactly, then checks that a call
returns native bools and that an unknown model returns the constants without loading anything.

Run:  HF_HUB_OFFLINE=1 python submission/test_e61_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
sys.path.insert(0, str(ROOT / "submission" / "e61-familiarity"))
sys.path.insert(0, str(ROOT / "datagen"))


def test_parity() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import solution
    from extract_familiarity import familiarity

    problems = pd.read_parquet(ROOT / "data" / "e55" / "instances.parquet").drop_duplicates("problem_key").head(3)["problem"]
    tokenizer, model = solution._load(TINY)
    scores = []
    for problem in problems:
        ours = solution.mean_nll(tokenizer, model, problem)
        assert ours == familiarity(model, tokenizer, problem)["mean_nll"], problem[:60]
        scores.append(ours)

    solution.THRESHOLDS = {TINY: sorted(scores)[1]}
    calls = [solution.are_robust(TINY, [p]) for p in problems]
    assert all(len(c) == 1 and type(c[0]) is bool for c in calls), calls
    assert sum(c[0] for c in calls) == 2, calls  # <= the middle score: two of three robust
    assert solution.are_robust("Qwen/Qwen3.5-4B", ["x", "y"]) == [True, True]  # no threshold: the constant
    assert solution.are_robust("someone/else", ["x"]) == [False]


if __name__ == "__main__":
    test_parity()
    print("ok: the submission's mean NLL equals extract_familiarity.py's; calls return native bools")
