"""E64: predicted difficulty from one forward pass; easier problems are called robust.

Each call carries one problem (Finding 79). The model being scored reads it once, with the probe baseline's system
prompt and chat template, and its hidden states are mean-pooled over the prompt at every layer, exactly as E55 pooled
them (the E57 template, parity 0.0 on four checkpoints, Finding 82). A per-layer standardised ridge, fitted under
`protocols/E64_PROTOCOL.md` to Qwen's log trace length on 497 olympiad problems, reads difficulty from each layer; the
layer predictions are averaged, and the problem is robust iff predicted difficulty is at or below that model's
cutoff. `submission/make_e64.py` ships the arrays and cutoffs from `train/eval_e64.py`.

The checkpoint is loaded once and reused across calls; switching model evicts the previous one. Any failure - no
cached weights, a lossy tokenizer, a model without arrays - returns the per-model constants (`e27f`), so every call
stays valid.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHORT = {"Qwen/Qwen3.5-4B": "qwen", "Skywork/Skywork-OR1-Math-7B": "skywork",
         "allenai/Olmo-3-7B-Think": "olmo", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek"}
CONSTANTS = {"Qwen/Qwen3.5-4B": True, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": True}
MAX_LENGTH = 2048  # extract_features.py's default, the baseline's max_prompt_length
_cache: dict = {}


def build_prompt(tokenizer, problem: str, system_prompt: str) -> str:
    """`extract_features.build_prompt`, unchanged."""
    if system_prompt and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": problem}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{system_prompt}\n\n{problem}" if system_prompt else problem


def _load(model_id: str):
    if _cache.get("id") == model_id:
        return _cache["tokenizer"], _cache["model"]
    import torch
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(HERE))
    from tokenizer_check import load_checked_tokenizer

    _cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cuda = torch.cuda.is_available()
    tokenizer, _ = load_checked_tokenizer(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=(torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32,
        device_map="auto" if cuda else None, local_files_only=True,
    ).eval()
    _cache.update(id=model_id, tokenizer=tokenizer, model=model)
    return tokenizer, model


def mean_states(tokenizer, model, problem: str, system_prompt: str) -> np.ndarray:
    """(n_layers, hidden) mean-pooled prompt states, cast to float16 as E55 stored them."""
    import torch

    inputs = tokenizer(build_prompt(tokenizer, problem, system_prompt), return_tensors="pt",
                       truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    mask = inputs["attention_mask"][0].bool()
    layers = [layer[0][mask.to(layer.device)].float().mean(dim=0).cpu().numpy() for layer in output.hidden_states]
    return np.stack(layers).astype(np.float16).astype(np.float32)


def difficulty(arrays, short: str, states: np.ndarray) -> float:
    """Mean over layers of the standardised ridge prediction of log trace length."""
    z = (states - arrays[f"{short}__mean"]) / arrays[f"{short}__scale"]
    return float(np.mean((z * arrays[f"{short}__coef"]).sum(axis=1) + arrays[f"{short}__intercept"]))


def are_robust(model_id, problems, reasoning_effort="default"):
    problems = list(problems)
    fallback = [CONSTANTS.get(model_id, False)] * len(problems)
    short = SHORT.get(model_id)
    try:
        arrays = np.load(HERE / "e64_difficulty.npz")
        if short is None or f"{short}__coef" not in arrays.files:
            return fallback
        system_prompt = (HERE / "system_prompt.txt").read_text(encoding="utf-8")
        tokenizer, model = _load(model_id)
        cutoff = float(arrays[f"{short}__cutoff"])
        return [bool(difficulty(arrays, short, mean_states(tokenizer, model, p, system_prompt)) <= cutoff) for p in problems]
    # SystemExit too: tokenizer_check raises it for a lossy tokenizer, and it would end the run.
    except (Exception, SystemExit) as exc:
        print(f"e64: {type(exc).__name__}: {exc}; falling back to constants", file=sys.stderr)
        return fallback
