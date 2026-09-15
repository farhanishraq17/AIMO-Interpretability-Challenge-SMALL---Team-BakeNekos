"""Shared RESULTS.md helpers for the training scripts.

`train_text_baseline.py` rewrites RESULTS.md from scratch and `check_dataset_confound.py`
appends to it. Later stages instead upsert a named `## ` section, so rerunning one stage
replaces its own numbers without touching anyone else's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def score(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, balanced accuracy, and ROC-AUC for out-of-fold probabilities."""
    pred = proba >= threshold
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(set(y.tolist())) > 1 else float("nan"),
        "n_predicted_robust": int(pred.sum()),
    }


def paired_difference(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, metric, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """metric(y, a) - metric(y, b) with a 95% paired bootstrap CI over instances.

    At ~138 instances a single AUC carries roughly +/-0.09 of noise, so differences
    between methods are only reported with this interval.
    """
    rng = np.random.default_rng(seed)
    diffs = []
    for idx in rng.integers(0, len(y), size=(n_boot, len(y))):
        if y[idx].all() or not y[idx].any():
            continue
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(metric(y, a) - metric(y, b)), float(low), float(high)


def paired_auc_difference(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """AUC(a) - AUC(b) with a 95% paired bootstrap CI over instances."""
    return paired_difference(y, a, b, roc_auc_score, n_boot, seed)


def metrics_row(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| {name} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} "
        f"| {metrics['roc_auc']:.4f} | {metrics['n_predicted_robust']} |"
    )


METRICS_HEADER = [
    "| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |",
    "| --- | --- | --- | --- | --- |",
]


def upsert_section(path: Path, heading: str, body: str) -> None:
    """Replace the `## {heading}` section (up to the next `## `), or append it."""
    marker = f"## {heading}"
    text = path.read_text(encoding="utf-8") if path.is_file() else "# Results\n"
    lines = text.splitlines()
    block = [marker, "", body.strip("\n"), ""]
    if marker in lines:
        start = lines.index(marker)
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines)
        )
        lines = lines[:start] + block + lines[end:]
    else:
        lines = lines + [""] + block
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    print(f"updated section '{heading}' in {path}")
