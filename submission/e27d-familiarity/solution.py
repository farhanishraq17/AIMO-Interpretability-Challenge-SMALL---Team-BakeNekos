"""E27d/e: split each batch by how familiar the problem is to the model scoring it.

Each batch is one `model_id`. That model is loaded from the worker cache, every problem gets
one forward pass over its bare text, and the batch is split at its median mean-token NLL. A
median split assumes nothing about the class prior, which the train split and the validation
set disagree on (Finding 39). If anything fails, the batch falls back to E11 v2's constants so
it stays valid.

`ROBUST_HALF` fixes which half is called robust, and the two readings have different evidence:

* `"familiar"` - the MATH direction. `problem_nll` ranks robust *below* spurious on DeepSeek's
  142 Hendrycks MATH instances (AUC 0.68 reversed, 0.58 within length quintiles: Findings 22,
  47), and E27d's 4/14 on the leaderboard says the validation cases agree, since the opposite
  split scored far below every constant (Finding 49).
* `"unfamiliar"` - the memorization reading, from E23's 33 official olympiad problems, where
  familiar problems lean spurious (AUC 0.69, CI to 0.49). That is what E27d shipped and what
  the leaderboard rejected.
"""

from __future__ import annotations

import gc
import sys

ROBUST_HALF = "unfamiliar"  # make_e27.py ships one ZIP per reading
# Which models get the split. "all", or a comma-separated list; the rest use the constants in
# FALLBACK_ROBUST, which are the best constants the probes allow (9/14, Finding 45). One ZIP per
# model decomposes the split's leaderboard gain by model without ever probing a single case.
SPLIT_MODELS = "all"
# How many of each batch to call robust, as ceil(n/2) + TILT[model]. The probes measured each
# model's robust-minus-spurious balance on the validation set: +1 for Qwen and DeepSeek, -1 for
# Skywork and Olmo (Finding 45). With an odd batch, ceil(n/2) is already the +1 count, so only
# the two spurious-leaning models need a tilt. An empty mapping is the plain median split.
TILT = {}
FALLBACK_ROBUST = {"Qwen/Qwen3.5-4B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
MAX_TOKENS = 2048


def problem_nll(model_id: str, problems: list[str]) -> list[float]:
    """Mean token NLL of each bare problem under `model_id`, as `datagen/extract_familiarity.py`."""
    import torch
    from transformers import AutoModelForCausalLM

    from tokenizer_check import load_checked_tokenizer

    tokenizer, _ = load_checked_tokenizer(model_id, local_files_only=True)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if cuda else torch.float32,
        device_map="auto" if cuda else None,
        local_files_only=True,
    ).eval()
    try:
        scores = []
        with torch.inference_mode():
            for problem in problems:
                ids = tokenizer(problem, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"]
                ids = ids.to(model.device)
                scores.append(float(model(input_ids=ids, labels=ids, use_cache=False).loss))
        return scores
    finally:
        del model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()


def split_half(scores: list[float], model_id: str = "") -> list[bool]:
    """Call the `ROBUST_HALF` end of the batch robust; ties break by input order."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))  # ascending NLL: familiar first
    if ROBUST_HALF == "unfamiliar":
        order.reverse()
    n_robust = (len(scores) + 1) // 2 + TILT.get(model_id, 0)
    robust = set(order[: max(0, min(len(scores), n_robust))])
    return [i in robust for i in range(len(scores))]


def uses_split(model_id: str) -> bool:
    return SPLIT_MODELS == "all" or model_id in SPLIT_MODELS.split(",")


def are_robust(model_id, problems, reasoning_effort="default"):
    problems = list(problems)
    if not uses_split(model_id):
        return [model_id in FALLBACK_ROBUST] * len(problems)
    try:
        return split_half(problem_nll(model_id, problems), model_id)
    # SystemExit too: `tokenizer_check` raises it for a lossy tokenizer, and it would end the run.
    except (Exception, SystemExit) as exc:  # never let a batch go invalid
        print(f"e27d: {type(exc).__name__}: {exc}; falling back to constants", file=sys.stderr)
        return [model_id in FALLBACK_ROBUST] * len(problems)
