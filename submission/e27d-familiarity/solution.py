"""E27d: robust iff the problem is unfamiliar to the model scoring it (memorization hypothesis).

A spurious case is one the model solves as written but not once perturbed, which is what a
memorized problem looks like. On the 33 official olympiad problems, E23 found familiar problems
lean spurious: the model's mean token NLL over the bare problem text ranks robust above
spurious at AUC 0.69 (PROGRESS Finding 47). That direction is the one fixed here, before any
leaderboard score is seen.

Each batch is one `model_id`. That model is loaded from the worker cache, every problem gets
one forward pass, and the less familiar half of the batch (higher NLL) is predicted robust. A
median split assumes nothing about the class prior, which the train split and the validation
set disagree on (Finding 39). If anything fails, the batch falls back to E11 v2's constants so
it stays valid.
"""

from __future__ import annotations

import gc
import sys

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


def upper_half(scores: list[float]) -> list[bool]:
    """True for the higher-scoring half, ties broken by input order; exactly floor(n/2) are False."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    robust = set(order[len(scores) // 2:])
    return [i in robust for i in range(len(scores))]


def are_robust(model_id, problems, reasoning_effort="default"):
    problems = list(problems)
    try:
        return upper_half(problem_nll(model_id, problems))
    # SystemExit too: `tokenizer_check` raises it for a lossy tokenizer, and it would end the run.
    except (Exception, SystemExit) as exc:  # never let a batch go invalid
        print(f"e27d: {type(exc).__name__}: {exc}; falling back to constants", file=sys.stderr)
        return [model_id in FALLBACK_ROBUST] * len(problems)
