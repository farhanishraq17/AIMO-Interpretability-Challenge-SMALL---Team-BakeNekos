"""E58's variant construction and row labels on hand-made records; fixed with the protocol, before any record.

Run:  python -m pytest -q train/test_eval_e58.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "datagen" / "perturb"))

from e58_variants import properties, satisfied  # noqa: E402
from eval_e58 import label_row  # noqa: E402

CAP = 1000


def record(category: str | None, answers: list, tails: list[str] | None = None, tokens: list[int] | None = None) -> dict:
    return {"category": category, "expected_answer": 7, "answers": answers, "n_samples": len(answers),
            "completion_tokens": tokens or [100] * len(answers), "sample_tails": tails or ["\\boxed{7}"] * len(answers)}


def test_variants() -> None:
    for answer in [0, 1, 2, 7, 50, 504, 26244]:
        for family, phrase in properties(answer).values():
            assert satisfied(phrase, answer) == (family == "consistent"), (answer, phrase)


def test_labels() -> None:
    base = record(None, [7] * 4)
    detected = ["</think>The conditions contradict each other, so there is no solution."] * 4
    consistent = [record("consistent_parity", [7] * 4), record("consistent_bound", [7] * 4)]
    robust = label_row(base, [record("contradict_parity", [None] * 4, detected),
                              record("contradict_bound", [None] * 4, detected), *consistent], CAP)
    assert (robust["status"], robust["label"], robust["max_drop"]) == ("labelled", "robust", 0.0)
    # Answering through the contradiction with the original answer is the spurious behaviour.
    spurious = label_row(base, [record("contradict_parity", [7] * 4), record("contradict_bound", [7] * 4), *consistent], CAP)
    assert (spurious["label"], spurious["deciding"], spurious["kept_original"]) == ("spurious", "no_solution", 8)
    # 7 of 8 detected is a drop of 0.125: band, so unlabelled.
    band = label_row(base, [record("contradict_parity", [None] * 4, detected),
                            record("contradict_bound", [None, None, None, 7], detected[:3] + ["\\boxed{7}"]), *consistent], CAP)
    assert band["status"] == "band"
    # A truncated base sample makes the row ineligible, whatever it says.
    assert label_row(record(None, [7] * 4, tokens=[100, 100, 100, CAP]), [], CAP)["status"] == "ineligible"
    assert label_row(None, [], CAP)["status"] == "not_sampled"
    assert label_row(base, consistent, CAP)["status"] == "incomplete"


if __name__ == "__main__":
    test_variants()
    test_labels()
    print("ok")
