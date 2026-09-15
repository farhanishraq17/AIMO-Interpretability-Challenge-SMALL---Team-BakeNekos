"""Text-only robustness baseline, no GPU required.

Establishes the number any model-internals method has to beat. Every feature here comes
from the problem statement alone, because that is all inference provides: cases expose
`{id, model_id, problem}` and nothing else (PLAN.md §9, Finding 3).

`dataset_id` is deliberately excluded. It is present in our training table and strongly
correlated with difficulty, but it is *not* available at inference, so using it would
inflate CV and teach us nothing transferable.

Evaluation uses the grouped folds from `data/small_track/splits.json`, so no problem
appears in both train and validation - 31 of 156 problems recur across models, and
ungrouped splitting would leak.

Run:  uv run python ../train/train_text_baseline.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
SPLITS = ROOT / "data" / "small_track" / "splits.json"

KEYWORDS = [
    "find",
    "prove",
    "compute",
    "determine",
    "how many",
    "what is",
    "maximum",
    "minimum",
    "probability",
    "integer",
    "positive",
    "remainder",
    "sum",
    "product",
    "least",
    "greatest",
]

LATEX_TOKENS = [
    r"\frac",
    r"\sqrt",
    r"\sum",
    r"\int",
    r"\prod",
    r"\log",
    r"\sin",
    r"\cos",
    r"\pmod",
    r"\binom",
    r"\cdot",
    r"\angle",
    r"\triangle",
]

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def handcrafted(text: str) -> list[float]:
    """Surface features a human could compute from the statement alone."""
    words = text.split()
    numbers = [float(n) for n in NUMBER_RE.findall(text)]
    sentences = [s for s in re.split(r"[.!?]\s", text) if s.strip()]
    lower = text.lower()

    features = [
        len(text),
        len(words),
        len(sentences),
        np.mean([len(w) for w in words]) if words else 0.0,
        len(set(w.lower() for w in words)) / len(words) if words else 0.0,
        text.count("$"),
        text.count("^"),
        text.count("_"),
        text.count("{"),
        text.count("?"),
        text.count("\\"),
        len(numbers),
        max(numbers) if numbers else 0.0,
        np.log1p(max(numbers)) if numbers else 0.0,
        np.mean(numbers) if numbers else 0.0,
    ]
    features += [float(lower.count(token)) for token in KEYWORDS]
    features += [float(text.count(token)) for token in LATEX_TOKENS]
    return features


def feature_names() -> list[str]:
    base = [
        "n_chars",
        "n_words",
        "n_sentences",
        "avg_word_len",
        "type_token_ratio",
        "n_dollar",
        "n_caret",
        "n_underscore",
        "n_brace",
        "n_question",
        "n_backslash",
        "n_numbers",
        "max_number",
        "log_max_number",
        "mean_number",
    ]
    return base + [f"kw_{k.replace(' ', '_')}" for k in KEYWORDS] + [
        f"tex_{t.strip(chr(92))}" for t in LATEX_TOKENS
    ]


def oof_predictions(build, X_train_arg, y, folds, texts) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities and hard predictions over the grouped folds."""
    proba = np.zeros(len(y), dtype=float)
    pred = np.zeros(len(y), dtype=bool)
    for fold in sorted(set(folds)):
        is_val = folds == fold
        if is_val.all() or not is_val.any():
            continue
        model = build()
        if isinstance(X_train_arg, np.ndarray):
            model.fit(X_train_arg[~is_val], y[~is_val])
            scores = model.predict_proba(X_train_arg[is_val])[:, 1]
            hard = model.predict(X_train_arg[is_val])
        else:
            train_texts = [t for t, v in zip(texts, is_val) if not v]
            val_texts = [t for t, v in zip(texts, is_val) if v]
            model.fit(train_texts, y[~is_val])
            scores = model.predict_proba(val_texts)[:, 1]
            hard = model.predict(val_texts)
        proba[is_val] = scores
        pred[is_val] = hard.astype(bool)
    return proba, pred


def evaluate(name: str, y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_robust": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan"),
        "n_predicted_robust": int(pred.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=ROOT / "RESULTS.md")
    args = parser.parse_args()

    rows = pq.read_table(INSTANCES).to_pylist()
    rows = [r for r in rows if r["model_id"] == args.model_id]
    fold_map = json.loads(SPLITS.read_text(encoding="utf-8"))["problem_fold"]

    texts = [r["problem"] for r in rows]
    y = np.asarray([bool(r["is_robust"]) for r in rows])
    folds = np.asarray([fold_map[r["problem_key"]] for r in rows])
    X = np.asarray([handcrafted(t) for t in texts], dtype=np.float64)

    print(f"model: {args.model_id}")
    print(f"instances: {len(y)}  robust: {int(y.sum())}  spurious: {int((~y).sum())}")
    print(f"majority-class rate: {max(y.mean(), 1 - y.mean()):.4f}")
    print(f"handcrafted features: {X.shape[1]}")
    print(f"folds: {dict(zip(*np.unique(folds, return_counts=True)))}\n")

    seed = args.seed
    candidates = {
        "always-spurious (majority)": (
            lambda: DummyClassifier(strategy="most_frequent"),
            X,
        ),
        "logreg on handcrafted": (
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed
                ),
            ),
            X,
        ),
        "hist-gradient-boosting on handcrafted": (
            lambda: HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.06, max_depth=3, random_state=seed
            ),
            X,
        ),
        "logreg on tf-idf words": (
            lambda: make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed
                ),
            ),
            texts,
        ),
        "logreg on tf-idf chars": (
            lambda: make_pipeline(
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed
                ),
            ),
            texts,
        ),
    }

    results = []
    for name, (build, data) in candidates.items():
        proba, pred = oof_predictions(build, data, y, folds, texts)
        metrics = evaluate(name, y, proba, pred)
        results.append(metrics)
        print(
            f"{name:40s} acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"auc={metrics['roc_auc']:.4f} f1={metrics['f1_robust']:.4f}"
        )

    # Which surface features carry signal at all, fit on everything for inspection.
    inspector = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    ).fit(X, y)
    coefs = inspector[-1].coef_[0]
    order = np.argsort(-np.abs(coefs))[:10]
    names = feature_names()
    print("\nlargest |coef| surface features (full-fit, direction toward robust):")
    for index in order:
        print(f"  {names[index]:24s} {coefs[index]:+.3f}")

    write_results(args, y, results, [(names[i], float(coefs[i])) for i in order])


def write_results(args, y, results: list[dict], top_features) -> None:
    best = max(results[1:], key=lambda r: r["balanced_accuracy"])
    majority = results[0]
    lines = [
        "# Results",
        "",
        "## Text-only baseline (no GPU)",
        "",
        f"Model: `{args.model_id}`  ",
        f"Instances: {len(y)} ({int(y.sum())} robust / {int((~y).sum())} spurious)  ",
        "Evaluation: out-of-fold over the 5 problem-grouped folds in "
        "`data/small_track/splits.json`  ",
        "Features: problem text only, since inference exposes just `{id, model_id, problem}`",
        "",
        "| Method | Accuracy | Balanced acc. | ROC-AUC | F1 (robust) | Predicted robust |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} "
            f"| {row['roc_auc']:.4f} | {row['f1_robust']:.4f} | {row['n_predicted_robust']} |"
        )

    lines += [
        "",
        "### Reading these numbers",
        "",
        f"Our training split is {1 - y.mean():.1%} spurious, so predicting *always spurious* "
        f"already scores {majority['accuracy']:.4f} accuracy here. The competition's public "
        "validation sample is only 62.5% spurious, so raw accuracy on this table is inflated "
        "relative to the leaderboard and **balanced accuracy is the number to track**.",
        "",
        f"Best text-only method: **{best['model']}** at "
        f"{best['balanced_accuracy']:.4f} balanced accuracy "
        f"({best['roc_auc']:.4f} AUC).",
        "",
        "This is the floor that model-internals work has to clear. Any probe or SAE feature "
        "set that cannot beat it is not reading robustness - it is reading problem surface "
        "form, which text features already capture for free.",
        "",
        "### Surface features with the largest weight",
        "",
        "| Feature | Coefficient (toward robust) |",
        "| --- | --- |",
    ]
    for name, coef in top_features:
        lines.append(f"| `{name}` | {coef:+.3f} |")
    lines += [
        "",
        "Fit on all instances for inspection only, not used for the scores above.",
        "",
        "Regenerate with `uv run python ../train/train_text_baseline.py`, then "
        "`uv run python ../train/check_dataset_confound.py` to append the confound analysis.",
        "",
    ]
    args.results.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {args.results}")


if __name__ == "__main__":
    main()
