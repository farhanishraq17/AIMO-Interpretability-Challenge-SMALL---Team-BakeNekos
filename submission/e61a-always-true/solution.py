"""E61a: answer `True` for every small-track pair, and load nothing.

This is the floor the rest of E61 is measured against, and it exists because the shipped
`e27f` constants are worse than it on the only labelled olympiad data we have. On the 43
official `train-main-v2` small-track rows (Finding 87, the table grew from 34 on 15 Oct):

    policy            accuracy   balanced
    e27f constants       0.465      0.450
    always-True          0.721      0.500
    always-False         0.279      0.500

`e27f` is below *both* constants on balanced accuracy and 11 rows behind this one on accuracy.

Two measurements say the per-model split `e27f` encodes should not exist at all:

* **Finding 38** - the label is a property of the problem, not the model. Small-track models
  agree on 20 of 20 shared problems and 83 of 85 across every configuration, against 0.608 at
  chance. `e27f`'s entire content is a per-model distinction.
* **E43, re-read on the grown table** - every model is majority robust: Qwen 8/12, DeepSeek
  7/10, Skywork 9/12, Olmo 7/9. Constants derived from this source are `True` four times, so
  there is no split to make. Skywork, which `e27f` hardcodes to `False`, is now the most
  robust-leaning of the four on 12 rows rather than the 3 the earlier docs recorded.

Why a constant rather than a model: a policy answering `True` on a share `w` of pairs and
`False` on the rest scores `w*p + (1-w)*(1-p)` under a shared robust prior `p` - a weighted
average of `p` and `1-p`, never above `max(p, 1-p)`. Always-`True` scores `p` itself. The
split can only win if the per-model priors differ in the direction `e27f` assumes, which is
what both measurements above argue against.

Deliberately loads no model. The ranking that used to sit on these constants cannot act under
per-problem calls (Finding 79), so a forward pass would buy nothing and only add a failure
mode; ties break on runtime, and this is the floor of that too.

Run: built by `submission/make_e61.py`; see `protocols/E61_PROTOCOL.md` gate A.
"""

from __future__ import annotations

# The four Competition Phase small-track models, passed verbatim with no aliases
# (docs/COMPETITION.md). Every one of them is majority robust in the official labels.
ROBUST_MODELS = {
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}


def are_robust(model_id, problems, reasoning_effort="default"):
    """One native `bool` per problem, in the order given.

    Accepts both call styles the harness uses: positional `(model_id, problems)` and the
    keyword form with `reasoning_effort`, which is merged upstream at `de79405`. Small-track
    cases are always `"default"`; `train-main-v2` also carries `"high"`, so the value is
    accepted and ignored rather than checked.

    A model we do not cover returns `False` rather than raising, as the contract requires.
    """
    n = len(list(problems))
    return [model_id in ROBUST_MODELS] * n
