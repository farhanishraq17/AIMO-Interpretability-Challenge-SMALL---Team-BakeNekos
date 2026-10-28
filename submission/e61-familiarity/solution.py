"""E61: familiarity as an absolute per-problem threshold, for the one-problem calls of 15 Oct.

`e27k` called the more familiar half of each model's batch robust (lower mean token NLL over the bare
problem). Calls now carry one problem (Finding 79), so a rank cannot act. Here the same statistic is
compared with a per-model threshold fixed on unlabelled problems (`protocols/E61_PROTOCOL.md`):
robust iff the problem is at least as familiar as the threshold.

The NLL is `datagen/extract_familiarity.familiarity`'s `mean_nll`, arithmetic unchanged, so a call's
score sits on the scale of the reference values the thresholds were taken from. The checkpoint is
loaded once and reused across calls; switching model evicts the previous one. Any failure, an unknown
model or a non-finite score returns the per-model constant (`e27f`), so every call stays valid.
`submission/make_e61.py` fills THRESHOLDS and STRICT, one ZIP per rule.
"""

from __future__ import annotations

import gc
import math
import sys

THRESHOLDS: dict[str, float] = {}
STRICT = False  # E61a compares with <, as E54's S1 did; E61b with <=
CONSTANTS = {"Qwen/Qwen3.5-4B": True, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": True}
MAX_TOKENS = 2048
_cache: dict = {}


def _load(model_id: str):
    if _cache.get("id") == model_id:
        return _cache["tokenizer"], _cache["model"]
    import torch
    from transformers import AutoModelForCausalLM

    from tokenizer_check import load_checked_tokenizer

    _cache.clear()
    gc.collect()
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
    tokenizer, _ = load_checked_tokenizer(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16 if cuda else torch.float32,
        device_map="auto" if cuda else None, local_files_only=True,
    ).eval()
    _cache.update(id=model_id, tokenizer=tokenizer, model=model)
    return tokenizer, model


def mean_nll(tokenizer, model, problem: str) -> float:
    import torch

    ids = tokenizer(problem, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = ids[0, 1:].to(log_probs.device)
    return float(-log_probs.gather(1, targets[:, None])[:, 0].mean())


def are_robust(model_id, problems, reasoning_effort="default"):
    problems = list(problems)
    constant = model_id in CONSTANTS
    if model_id not in THRESHOLDS:
        return [constant] * len(problems)
    try:
        tokenizer, model = _load(model_id)
        threshold, out = THRESHOLDS[model_id], []
        for problem in problems:
            score = mean_nll(tokenizer, model, problem)
            if not math.isfinite(score):
                out.append(constant)
            else:
                out.append(bool(score < threshold if STRICT else score <= threshold))
        return out
    # SystemExit too: tokenizer_check raises it for a lossy tokenizer, and it would end the run.
    except (Exception, SystemExit) as exc:
        print(f"e61: {type(exc).__name__}: {exc}; falling back to constants", file=sys.stderr)
        return [constant] * len(problems)
