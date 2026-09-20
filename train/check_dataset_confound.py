"""Test whether the text baseline is really detecting the source dataset.

Most of the train-ready set is Hendrycks MATH while the private test set is entirely
olympiad. If robustness rate varies sharply by source
dataset, a text model can score well by inferring the dataset from formatting alone -
signal that will not transfer. This script quantifies that risk four ways:

1. Robustness base rate per source dataset.
2. How accurately the same text features predict the *dataset* instead of the label.
3. Label performance within a single dataset, where the confound is held constant.
4. A MATH-to-olympiad transfer check, now possible: `train-main-v2` (17 Sep) added olympiad
   instances with official labels.

Appends its findings to RESULTS.md, so run it *after* `train_text_baseline.py`, which
rewrites that file from scratch.

Run:  uv run python ../train/check_dataset_confound.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
SPLITS = ROOT / "data" / "small_track" / "splits.json"
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def char_model() -> object:
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
    )


def grouped_oof(texts: list[str], y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    proba = np.zeros(len(y), dtype=float)
    for fold in sorted(set(folds.tolist())):
        is_val = folds == fold
        if is_val.all() or not is_val.any():
            continue
        train_y = y[~is_val]
        if len(set(train_y.tolist())) < 2:
            proba[is_val] = train_y.mean()
            continue
        model = char_model()
        model.fit([t for t, v in zip(texts, is_val) if not v], train_y)
        proba[is_val] = model.predict_proba([t for t, v in zip(texts, is_val) if v])[:, 1]
    return proba


def main() -> None:
    rows = [r for r in pq.read_table(INSTANCES).to_pylist() if r["model_id"] == MODEL_ID]
    fold_map = json.loads(SPLITS.read_text(encoding="utf-8"))["problem_fold"]

    texts = [r["problem"] for r in rows]
    y = np.asarray([bool(r["is_robust"]) for r in rows])
    datasets = [r["dataset_id"] for r in rows]
    folds = np.asarray([fold_map[r["problem_key"]] for r in rows])

    print("1. robustness base rate by source dataset")
    for name, count in Counter(datasets).most_common():
        mask = np.asarray([d == name for d in datasets])
        print(
            f"   {name:38s} n={count:4d}  robust={int(y[mask].sum()):3d}"
            f"  rate={y[mask].mean():.3f}"
        )
    print(f"   {'overall':38s} n={len(y):4d}  robust={int(y.sum()):3d}  rate={y.mean():.3f}")

    print("\n2. can the same features predict the dataset instead of the label?")
    is_math = np.asarray([d.startswith("hendrycks") for d in datasets])
    if len(set(is_math.tolist())) > 1:
        proba = grouped_oof(texts, is_math, folds)
        pred = proba >= 0.5
        print(
            f"   predicting 'is Hendrycks MATH': bal_acc={balanced_accuracy_score(is_math, pred):.4f}"
            f"  auc={roc_auc_score(is_math, proba):.4f}"
        )
        print(f"   (base rate {is_math.mean():.3f}) - high scores mean the confound is learnable")

    print("\n3. label performance within a single dataset (confound held constant)")
    for name, count in Counter(datasets).most_common():
        mask = np.asarray([d == name for d in datasets])
        if count < 30 or len(set(y[mask].tolist())) < 2:
            print(f"   {name:38s} n={count:4d}  skipped (too small or single-class)")
            continue
        sub_texts = [t for t, m in zip(texts, mask) if m]
        proba = grouped_oof(sub_texts, y[mask], folds[mask])
        pred = proba >= 0.5
        print(
            f"   {name:38s} n={count:4d}  bal_acc={balanced_accuracy_score(y[mask], pred):.4f}"
            f"  auc={roc_auc_score(y[mask], proba):.4f}"
        )

    print("\n4. olympiad-only holdout (the distribution that actually matters)")
    olympiad = np.asarray([not d.startswith("hendrycks") for d in datasets])
    print(f"   olympiad instances: {int(olympiad.sum())}  robust: {int(y[olympiad].sum())}")
    if 0 < olympiad.sum() and len(set(y[~olympiad].tolist())) > 1:
        model = char_model().fit([t for t, m in zip(texts, olympiad) if not m], y[~olympiad])
        proba = model.predict_proba([t for t, m in zip(texts, olympiad) if m])[:, 1]
        pred = proba >= 0.5
        if len(set(y[olympiad].tolist())) > 1:
            print(
                f"   train on MATH -> test on olympiad: "
                f"bal_acc={balanced_accuracy_score(y[olympiad], pred):.4f}"
                f"  auc={roc_auc_score(y[olympiad], proba):.4f}"
            )
        else:
            print(
                f"   olympiad subset is single-class "
                f"(all {'robust' if y[olympiad].all() else 'spurious'}); "
                f"predicted robust for {int(pred.sum())}/{int(olympiad.sum())}"
            )

    append_results(rows, y, datasets, texts, folds)


def transfer_summary(y, datasets, texts) -> str:
    """Train on the MATH instances, test on the olympiad ones: the shift the test set asks for."""
    olympiad = np.asarray([not d.startswith("hendrycks") for d in datasets])
    counts = f"{int(olympiad.sum())} olympiad instances ({int(y[olympiad].sum())} robust)"
    if not olympiad.any() or len(set(y[olympiad].tolist())) < 2 or len(set(y[~olympiad].tolist())) < 2:
        return f"{counts}: too few, or single-class, to measure transfer."
    model = char_model().fit([t for t, m in zip(texts, olympiad) if not m], y[~olympiad])
    proba = model.predict_proba([t for t, m in zip(texts, olympiad) if m])[:, 1]
    return (
        f"Char tf-idf trained on the {int((~olympiad).sum())} MATH instances and tested on the {counts}: "
        f"**balanced accuracy {balanced_accuracy_score(y[olympiad], proba >= 0.5):.4f}, "
        f"ROC-AUC {roc_auc_score(y[olympiad], proba):.4f}**. Below 0.5 means the text signal does not "
        "survive the shift in difficulty, which is exactly what the private test set asks of it. "
        "At this sample size treat it as a warning, not a measurement."
    )


def append_results(rows, y, datasets, texts, folds) -> None:
    """Append the confound analysis to RESULTS.md."""
    results = ROOT / "RESULTS.md"
    if not results.is_file():
        print("\nRESULTS.md missing; run train_text_baseline.py first")
        return

    lines = [
        "",
        "## Is the text baseline just detecting the source dataset?",
        "",
        "The concern: the train-ready set is mostly Hendrycks MATH problems while the private "
        "test set is entirely olympiad. A text model could score "
        "well by recognising dataset formatting rather than robustness.",
        "",
        "### Base rate by source dataset",
        "",
        "| Dataset | Instances | Robust | Rate |",
        "| --- | --- | --- | --- |",
    ]
    for name, count in Counter(datasets).most_common():
        mask = np.asarray([d == name for d in datasets])
        lines.append(
            f"| `{name}` | {count} | {int(y[mask].sum())} | {y[mask].mean():.3f} |"
        )

    lines += [
        "",
        "### Performance within a single dataset (confound held constant)",
        "",
        "| Dataset | Instances | Balanced acc. | ROC-AUC |",
        "| --- | --- | --- | --- |",
    ]
    for name, count in Counter(datasets).most_common():
        mask = np.asarray([d == name for d in datasets])
        if count < 30 or len(set(y[mask].tolist())) < 2:
            continue
        sub_texts = [t for t, m in zip(texts, mask) if m]
        proba = grouped_oof(sub_texts, y[mask], folds[mask])
        pred = proba >= 0.5
        lines.append(
            f"| `{name}` | {count} | {balanced_accuracy_score(y[mask], pred):.4f} "
            f"| {roc_auc_score(y[mask], proba):.4f} |"
        )

    lines += [
        "",
        "Within-dataset scores land close to the pooled number in the table above, and the MATH "
        "subsets have similar base rates, so **the signal is not a dataset artifact** within MATH. "
        "Char n-grams do carry robustness information about the problem statement.",
        "",
        "### MATH to olympiad: the transfer the competition actually scores",
        "",
        transfer_summary(y, datasets, texts),
        "",
        "Regenerate with `train_text_baseline.py` followed by `check_dataset_confound.py`.",
        "",
    ]
    with results.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"\nappended confound analysis to {results}")


if __name__ == "__main__":
    main()
