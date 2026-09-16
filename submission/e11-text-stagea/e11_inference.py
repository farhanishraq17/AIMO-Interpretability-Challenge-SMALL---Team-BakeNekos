"""E11 inference: text + prompt-hidden-state fusion (Findings 11 and 16).

For `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` each problem gets two scores:

* **text:** a char tf-idf logistic regression over the problem statement;
* **stagea:** the model is run once over the probe baseline's chat prompt; every layer's
  prompt tokens are pooled (last, mean, max), each of the 111 (pooling, layer) vectors is
  scored by its own linear probe, and the probabilities are averaged.

Both scores become standardised logits, are averaged, and are compared with a threshold
learned on out-of-fold training scores. Everything is fit by `train/build_e11_artifact.py`.

Decisions that matter at evaluation time:

* **Tokenizer.** Under transformers 5.13, `AutoTokenizer` loads this checkpoint as a
  `LlamaTokenizer` that strips all whitespace (Finding 14). The fast tokenizer is loaded
  instead, and it must round-trip a sample before the model is used.
* **Other models.** No labels exist for Qwen/Qwen3.5-4B, Skywork/Skywork-OR1-Math-7B or
  allenai/Olmo-3-7B-Think, and every public model with enough labels is mostly spurious, so
  any model other than DeepSeek gets `False` for every problem. The Competition Phase passes
  model ids verbatim, so no alias resolution is needed.
* **Failure policy.** If the language model cannot be loaded or run, the batch falls back to
  the text block with its own out-of-fold threshold; if the text block itself fails, every
  prediction is `False`. Either way the batch stays valid.
"""

from __future__ import annotations

import functools
import gc
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "e11_deepseek.joblib"
ROUND_TRIP_SAMPLE = "Every morning Aya goes for a $9$-kilometer-long walk.\nFind the answer \\boxed{204}."
EPS = 1e-6


@functools.cache
def load_artifact() -> dict:
    return joblib.load(ARTIFACT_PATH)


def standardised(block: str, probabilities: np.ndarray, artifact: dict) -> np.ndarray:
    z = logit(np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1 - EPS))
    return (z - artifact["block_logit_mean"][block]) / artifact["block_logit_std"][block]


def stagea_probability(pooled: dict[str, np.ndarray], artifact: dict) -> float:
    """Mean probability over every (pooling, layer) probe for one problem's pooled states."""
    probabilities = []
    for index, pooling in enumerate(artifact["poolings"]):
        x = np.asarray(pooled[pooling], dtype=np.float64)  # (n_layers, hidden_dim)
        z = (x - artifact["probe_mean"][index]) / artifact["probe_scale"][index]
        margins = np.einsum("ld,ld->l", z, artifact["probe_coef"][index].astype(np.float64))
        probabilities.append(expit(margins + artifact["probe_intercept"][index]))
    return float(np.mean(np.concatenate(probabilities)))


def fused_decisions(text_prob: np.ndarray, stagea_prob: np.ndarray, artifact: dict) -> np.ndarray:
    """Decision scores; >= 0 means robust."""
    fused = (standardised("text", text_prob, artifact) + standardised("stagea", stagea_prob, artifact)) / 2
    return fused - artifact["threshold"]


def text_only_decisions(text_prob: np.ndarray, artifact: dict) -> np.ndarray:
    return standardised("text", text_prob, artifact) - artifact["text_threshold"]


def load_language_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, local_files_only=True)
    except (OSError, TypeError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    ids = tokenizer(ROUND_TRIP_SAMPLE, add_special_tokens=False)["input_ids"]
    if tokenizer.decode(ids) != ROUND_TRIP_SAMPLE:
        raise RuntimeError(f"{type(tokenizer).__name__} does not round-trip text")

    cuda = torch.cuda.is_available()
    if cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if cuda else None,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def pooled_prompt_states(tokenizer, model, problem: str, artifact: dict) -> dict[str, np.ndarray]:
    """Replicates datagen/extract_features.py: same prompt, truncation, poolings, float16 storage."""
    import torch

    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": artifact["system_prompt"]}, {"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=artifact["max_prompt_tokens"])
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

    mask = inputs["attention_mask"][0].bool()
    pooled: dict[str, list[np.ndarray]] = {"last": [], "mean": [], "max": []}
    for layer in output.hidden_states:
        tokens = layer[0][mask.to(layer.device)].float()
        pooled["last"].append(tokens[-1].cpu().numpy())
        pooled["mean"].append(tokens.mean(dim=0).cpu().numpy())
        pooled["max"].append(tokens.max(dim=0).values.cpu().numpy())
    del output
    # Training features were stored as float16; round-trip through it for identical inputs.
    return {name: np.stack(values).astype(np.float16).astype(np.float32) for name, values in pooled.items()}


def predict_robustness(model_id: str, problems: list[str]) -> list[bool]:
    artifact = load_artifact()
    if model_id != artifact["model_id"]:
        return [False] * len(problems)

    try:
        text_prob = artifact["text_model"].predict_proba(problems)[:, 1]
    except Exception as exc:  # e.g. an sklearn version mismatch; never let the batch go invalid
        print(f"e11: text block failed ({type(exc).__name__}: {exc}); predicting False", file=sys.stderr)
        return [False] * len(problems)

    import torch

    tokenizer = model = None
    try:
        tokenizer, model = load_language_model(model_id)
        stagea_prob = np.array(
            [stagea_probability(pooled_prompt_states(tokenizer, model, p, artifact), artifact) for p in problems]
        )
        scores = fused_decisions(text_prob, stagea_prob, artifact)
    except Exception as exc:  # keep the batch valid if the model path fails
        print(f"e11: model path failed ({type(exc).__name__}: {exc}); using the text block alone", file=sys.stderr)
        scores = text_only_decisions(text_prob, artifact)
    finally:
        tokenizer = model = None
        gc.collect()
        torch.cuda.empty_cache()
    return [bool(score >= 0.0) for score in scores]
