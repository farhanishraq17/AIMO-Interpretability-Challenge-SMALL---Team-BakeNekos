"""E28: robust familiarity-rank candidates derived from the E27 result.

The evaluator calls this function once per model with that model's complete problem batch.
Each problem gets one teacher-forced forward pass over its bare statement.  Lower scores mean
the statement is more familiar; the familiar half of each batch is called robust.

`SCORE_MODE` is replaced by `make_e28.py` to build the pre-registered variants.  Any model or
tokenizer failure falls back to the best per-model constants, preserving full coverage.
"""

from __future__ import annotations

import gc
import sys

SCORE_MODE = "suffix_nll"
SPLIT_MODELS = "all"
FALLBACK_ROBUST = {"Qwen/Qwen3.5-4B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
SUPPORTED_MODELS = {
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}
MAX_TOKENS = 2048


def percentile_ranks(values: list[float]) -> list[float]:
    """Stable zero-based ranks; only their ordering is used."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return ranks


def combine_scores(records: list[dict[str, float]]) -> list[float]:
    if SCORE_MODE in {"mean_nll", "suffix_nll", "min_k_20"}:
        return [record[SCORE_MODE] for record in records]
    if SCORE_MODE == "min_k_gap":
        return [record["min_k_10"] - record["min_k_20"] for record in records]
    if SCORE_MODE == "mean_suffix":
        mean_rank = percentile_ranks([record["mean_nll"] for record in records])
        suffix_rank = percentile_ranks([record["suffix_nll"] for record in records])
        return [(mean_rank[i] + suffix_rank[i]) / 2 for i in range(len(records))]
    raise ValueError(f"unsupported SCORE_MODE: {SCORE_MODE}")


def problem_scores(model_id: str, problems: list[str]) -> list[dict[str, float]]:
    """Compute mean, suffix, and Min-K 20% token NLL in one forward pass per problem."""
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM

    from tokenizer_check import load_checked_tokenizer

    if not problems:
        return []
    tokenizer, _ = load_checked_tokenizer(model_id, local_files_only=True)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if cuda else torch.float32,
        device_map="auto" if cuda else None,
        local_files_only=True,
    ).eval()
    try:
        records = []
        with torch.inference_mode():
            for problem in problems:
                ids = tokenizer(
                    problem, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
                )["input_ids"].to(model.device)
                if ids.shape[1] < 2:
                    records.append({
                        "mean_nll": float("inf"), "suffix_nll": float("inf"),
                        "min_k_10": float("inf"), "min_k_20": float("inf"),
                    })
                    continue
                logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
                targets = ids[0, 1:].to(logits.device)
                nll = functional.cross_entropy(logits, targets, reduction="none")
                half = nll.numel() // 2
                k10 = max(1, int(round(0.10 * nll.numel())))
                k20 = max(1, int(round(0.20 * nll.numel())))
                records.append({
                    "mean_nll": float(nll.mean()),
                    "suffix_nll": float(nll[half:].mean()),
                    "min_k_10": float(nll.topk(k10, largest=True).values.mean()),
                    "min_k_20": float(nll.topk(k20, largest=True).values.mean()),
                })
        return records
    finally:
        del model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()


def familiar_half(scores: list[float]) -> list[bool]:
    """Call the lower-scoring ceil(n/2) problems robust, preserving input order."""
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    robust = set(order[: (len(scores) + 1) // 2])
    return [index in robust for index in range(len(scores))]


def uses_split(model_id: str) -> bool:
    return model_id in SUPPORTED_MODELS and (SPLIT_MODELS == "all" or model_id in SPLIT_MODELS.split(","))


def are_robust(model_id, problems, reasoning_effort="default"):
    problems = list(problems)
    if not uses_split(model_id):
        return [model_id in FALLBACK_ROBUST] * len(problems)
    try:
        return familiar_half(combine_scores(problem_scores(model_id, problems)))
    except (Exception, SystemExit) as exc:
        print(f"e28: {type(exc).__name__}: {exc}; falling back to constants", file=sys.stderr)
        return [model_id in FALLBACK_ROBUST] * len(problems)
