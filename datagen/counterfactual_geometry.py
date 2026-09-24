"""Pure NumPy reductions for E31 counterfactual activation geometry."""

from __future__ import annotations

import numpy as np

SIGNALS = (
    "cf_cosine_tail_mean",
    "cf_cosine_tail_max",
    "cf_cosine_auc",
    "cf_cosine_slope",
    "cf_js_next_mean",
)


def cosine_distance_curve(anchor: np.ndarray, variant: np.ndarray) -> np.ndarray:
    """Layerwise cosine distance for equally shaped ``(layers, hidden)`` arrays."""
    if anchor.shape != variant.shape or anchor.ndim != 2:
        raise ValueError(f"activation shapes differ: {anchor.shape} versus {variant.shape}")
    numerator = np.einsum("ij,ij->i", anchor, variant, dtype=np.float64)
    denominator = np.linalg.norm(anchor, axis=1) * np.linalg.norm(variant, axis=1)
    cosine = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def jensen_shannon_from_logits(anchor: np.ndarray, variant: np.ndarray) -> float:
    """Stable natural-log Jensen-Shannon divergence between two logit vectors."""
    # ``copy=True`` prevents the stabilization below from mutating float64 caller arrays.
    a = np.array(anchor, dtype=np.float64, copy=True)
    b = np.array(variant, dtype=np.float64, copy=True)
    a -= np.max(a)
    b -= np.max(b)
    p = np.exp(a)
    q = np.exp(b)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    tiny = np.finfo(np.float64).tiny
    return float(
        0.5 * np.sum(p * (np.log(np.maximum(p, tiny)) - np.log(np.maximum(midpoint, tiny))))
        + 0.5 * np.sum(q * (np.log(np.maximum(q, tiny)) - np.log(np.maximum(midpoint, tiny))))
    )


def counterfactual_signals(
    anchor_states: np.ndarray,
    variant_states: list[np.ndarray],
    anchor_logits: np.ndarray,
    variant_logits: list[np.ndarray],
) -> dict[str, float]:
    """Reduce original/rewrite forward passes to the five pre-registered signals."""
    if not variant_states or len(variant_states) != len(variant_logits):
        raise ValueError("at least one aligned state/logit variant is required")
    curves = np.stack([cosine_distance_curve(anchor_states, state) for state in variant_states])
    curves = curves[:, 1:] if curves.shape[1] > 1 else curves
    quarter = max(1, int(np.ceil(curves.shape[1] / 4)))
    first = curves[:, :quarter]
    tail = curves[:, -quarter:]
    js = [jensen_shannon_from_logits(anchor_logits, logits) for logits in variant_logits]
    result = {
        "cf_cosine_tail_mean": float(tail.mean()),
        "cf_cosine_tail_max": float(tail.max()),
        "cf_cosine_auc": float(curves.mean()),
        "cf_cosine_slope": float(tail.mean() - first.mean()),
        "cf_js_next_mean": float(np.mean(js)),
    }
    if not all(np.isfinite(result[name]) for name in SIGNALS):
        raise ValueError(f"non-finite counterfactual signal: {result}")
    return result
