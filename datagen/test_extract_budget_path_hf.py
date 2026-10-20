"""Checks for E49's batching and per-row stopping, without loading a model.

The subtle case is the one the tiny smoke model never produces: a row that stops naturally and is
then padded, when the pad id is also an EOS id. Its trace must end at the first stop token, and a
row that runs to its budget must be cut at the budget before any stop token is searched for.

Run:  python datagen/test_extract_budget_path_hf.py      (or pytest)
"""

from __future__ import annotations

import torch

from extract_budget_path_hf import PROBES, RowBudget, Sizer, batches, cost, fit_memory, generate, positions

EOS = 2


class FakeModel:
    """Appends a fixed continuation, as `generate` would after per-row stopping and padding."""

    device = torch.device("cpu")

    def __init__(self, continuation: list[list[int]]):
        self.continuation = torch.tensor(continuation)

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, self.continuation], dim=1)


JOBS = [(f"p{i}", budget, [1] * (10 + i)) for i in range(3) for budget in (8, 16, 32, 64)]


def test_batches_keep_every_job_within_the_cap():
    groups = batches(JOBS, Sizer(200).fits)
    assert sorted(job for group in groups for job in group) == sorted(JOBS)
    assert all(cost(group) <= 200 or len(group) == 1 for group in groups)
    assert [job[1] for group in groups for job in group] == sorted((job[1] for job in JOBS), reverse=True)
    assert len(batches(JOBS, Sizer().fits)) == 1


def test_fit_memory_recovers_a_linear_device():
    # Two devices with known overhead, per-sequence and per-position bytes; the probes' peaks follow
    # the model exactly, so the fit must return the coefficients unchanged.
    truth, width = [(1000.0, 300.0, 7.0), (0.0, 50.0, 2.0)], 40
    peaks = {(n, t): [o + n * (k + s * (width + t)) for o, k, s in truth] for n, t in PROBES}
    for fitted, expected in zip(fit_memory(peaks, width), truth):
        assert all(abs(a - b) < 1e-6 for a, b in zip(fitted, expected)), (fitted, expected)


def test_sizer_keeps_predicted_peak_under_the_margin_and_tightens_after_oom():
    sizer = Sizer(memory=[(0.0, 0.0, 1.0)], available=[1_000])  # one byte per cached position
    groups = batches(JOBS, sizer.fits)
    assert len(groups) > 1 and all(sizer.predict(g)[0] <= 0.85 * 1_000 or len(g) == 1 for g in groups)
    assert sizer.predict(groups[0]) == [len(groups[0]) * positions(groups[0])]
    before = max(len(g) for g in groups)
    sizer.shrink(groups[0])
    assert sizer.safety == 0.425 and max(len(g) for g in batches(JOBS, sizer.fits)) < before


def test_row_budget_finishes_rows_independently():
    stop = RowBudget(width=5, budgets=[2, 4])
    assert stop(torch.zeros(2, 7, dtype=torch.long), None).tolist() == [True, False]
    assert stop(torch.zeros(2, 9, dtype=torch.long), None).tolist() == [True, True]


def test_generate_cuts_at_the_budget_then_at_the_first_stop():
    model = FakeModel([
        [5, 6, 7, EOS, EOS, EOS],  # natural stop after three tokens, then EOS-valued padding
        [5, 6, 7, 8, EOS, EOS],    # ran to its budget of 4; the padding after it is not a stop
    ])
    (natural, budgeted), stats = generate(model, [[1, 1], [1]], [6, 4], pad_id=EOS, stop_ids=[EOS])
    assert natural == ([5, 6, 7], False)
    assert budgeted == ([5, 6, 7, 8], True)
    assert stats["generated_tokens"] == 7 and stats["width"] == 2 and stats["steps"] == 6


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok  {name}")
