"""The E52 grader on hand-written endings; fixed with the protocol, before any record was read.

Run:  python -m pytest -q train/test_eval_e52.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_e52 import CAP, grade  # noqa: E402


def test_grade() -> None:
    assert grade("...so the answer is \\boxed{20}.", 900) == "answered"
    assert grade("</think>The conditions contradict each other, so there is no solution.", 900) == "detected"
    assert grade("</think>\\boxed{\\text{No solution}}", 900) == "detected"
    assert grade("Such a triangle does not exist.", 900) == "detected"
    assert grade("</think>The contradiction means we ignore it. \\boxed{20}", 900) == "answered"
    assert grade("</think>\\boxed{\\frac{3}{2}}", 900) == "other"
    assert grade("still working on the case", CAP) == "truncated"
    assert grade("no solution", CAP) == "truncated"  # unfinished, whatever its text says


if __name__ == "__main__":
    test_grade()
    print("ok")
