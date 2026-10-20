"""Evaluate E35's pre-registered external train/holdout experiment.

This script deliberately knows nothing about AIMO labels. It derives clear-cut problem labels
from Qwen3-Math-Eval, trains the single frozen text ensemble on MATH-Perturb, and scores the
source-disjoint Putnam-AXIOM holdout once.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


BUDGET = 16_384
SEED = 35


def _one_value(part: pd.DataFrame, variant: str) -> bool:
    rows = part.loc[part["variant"] == variant, "correct_strict"]
    if len(rows) != 1:
        raise ValueError(f"expected one {variant!r} row, found {len(rows)}")
    return bool(rows.iloc[0])


def derive_labels(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return one clear-cut label per problem under the frozen E35 rules."""
    required = {"model", "token_budget", "problem_id", "question", "variant", "correct_strict"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = frame.loc[frame["token_budget"] == BUDGET].copy()
    if frame.empty:
        raise ValueError(f"no {BUDGET}-token rows")
    # The archive encodes the paired variant in problem_id. Canonicalising only removes that
    # documented suffix/prefix; it does not inspect correctness or change the frozen pairing.
    if source == "math_perturb":
        suffix = "#" + frame["variant"].astype(str)
        valid = [str(value).endswith(tag) for value, tag in zip(frame["problem_id"], suffix)]
        if not all(valid):
            raise ValueError("unexpected MATH-Perturb problem_id format")
        frame["problem_id"] = [str(value)[: -len(tag)] for value, tag in zip(frame["problem_id"], suffix)]
    elif source == "putnam_axiom":
        def putnam_id(value: object, variant: object) -> str:
            pattern = r"^putnam_orig_(.+::.+)$" if variant == "original" else r"^putnam_var_(.+::.+)_\d+$"
            match = re.fullmatch(pattern, str(value))
            if match is None:
                raise ValueError(f"unexpected Putnam problem_id format: {value!r}")
            return "putnam_" + match.group(1)

        frame["problem_id"] = [putnam_id(value, variant) for value, variant in zip(frame["problem_id"], frame["variant"])]

    by_config = []
    for (problem_id, model), part in frame.groupby(["problem_id", "model"], sort=True):
        if source == "math_perturb":
            eligible = _one_value(part, "base")
            robust = eligible and _one_value(part, "simple") and _one_value(part, "hard")
            original = part.loc[part["variant"] == "base", "question"]
        elif source == "putnam_axiom":
            eligible = _one_value(part, "original")
            variations = part.loc[part["variant"] == "variation", "correct_strict"]
            if variations.empty:
                raise ValueError(f"{problem_id}/{model}: no variation row")
            robust = eligible and bool(variations.astype(bool).all())
            original = part.loc[part["variant"] == "original", "question"]
        else:
            raise ValueError(source)
        if len(original) != 1:
            raise ValueError(f"{problem_id}/{model}: expected one original question")
        by_config.append(
            {
                "problem_id": str(problem_id),
                "model": str(model),
                "problem": str(original.iloc[0]),
                "eligible": eligible,
                "robust": robust,
            }
        )

    configs = pd.DataFrame(by_config)
    rows = []
    for problem_id, part in configs.groupby("problem_id", sort=True):
        originals = part["problem"].drop_duplicates()
        if len(originals) != 1:
            raise ValueError(f"{problem_id}: original text differs across models")
        eligible = part.loc[part["eligible"]]
        n_eligible = len(eligible)
        robust_rate = float(eligible["robust"].mean()) if n_eligible else np.nan
        label = True if n_eligible >= 2 and robust_rate >= 0.75 else (
            False if n_eligible >= 2 and robust_rate <= 0.25 else None
        )
        rows.append(
            {
                "problem_id": str(problem_id),
                "problem": originals.iloc[0],
                "eligible_models": n_eligible,
                "robust_rate": robust_rate,
                "is_robust": label,
            }
        )
    return pd.DataFrame(rows).loc[lambda x: x["is_robust"].notna()].astype({"is_robust": bool})


def _vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    common = dict(min_df=2, max_features=20_000, sublinear_tf=True, dtype=np.float64)
    return (
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **common),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), **common),
    )


def fit_score(train: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    scores = []
    y = train["is_robust"].to_numpy(bool)
    for vectorizer in _vectorizers():
        x_train = vectorizer.fit_transform(train["problem"])
        x_holdout = vectorizer.transform(holdout["problem"])
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, random_state=SEED)
        model.fit(x_train, y)
        scores.append(model.predict_proba(x_holdout)[:, 1])
    return np.mean(scores, axis=0)


def half_rank(ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Stable higher-half decision; ceil(n/2) rows are robust."""
    n = len(scores)
    order = np.lexsort((ids.astype(str), scores))
    pred = np.zeros(n, dtype=bool)
    pred[order[n // 2 :]] = True
    return pred


def auc_bootstrap(y: np.ndarray, score: np.ndarray, replicates: int = 5_000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    values = np.empty(replicates, dtype=float)
    for i in range(replicates):
        sample = np.concatenate((rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)))
        values[i] = roc_auc_score(y[sample], score[sample])
    return tuple(np.quantile(values, [0.025, 0.975]))


def read_shards(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet shards in {directory}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def evaluate(math: pd.DataFrame, putnam: pd.DataFrame, bootstrap: int = 5_000) -> dict:
    train = derive_labels(math, "math_perturb")
    holdout = derive_labels(putnam, "putnam_axiom")
    score = fit_score(train, holdout)
    y = holdout["is_robust"].to_numpy(bool)
    pred = half_rank(holdout["problem_id"].to_numpy(str), score)
    auc = float(roc_auc_score(y, score))
    ci = auc_bootstrap(y, score, bootstrap)
    balanced = float(balanced_accuracy_score(y, pred))
    counts = lambda frame: {
        "n": len(frame),
        "robust": int(frame["is_robust"].sum()),
        "spurious": int((~frame["is_robust"]).sum()),
    }
    train_counts, holdout_counts = counts(train), counts(holdout)
    gates = {
        "train_at_least_30_each": bool(min(train_counts["robust"], train_counts["spurious"]) >= 30),
        "holdout_at_least_20_each": bool(min(holdout_counts["robust"], holdout_counts["spurious"]) >= 20),
        "auc_at_least_0_65": bool(auc >= 0.65),
        "auc_ci_lower_above_0_50": bool(ci[0] > 0.50),
        "balanced_accuracy_at_least_0_60": bool(balanced >= 0.60),
    }
    return {
        "train": train_counts,
        "holdout": holdout_counts,
        "holdout_auc": auc,
        "holdout_auc_ci95": list(ci),
        "holdout_balanced_accuracy": balanced,
        "holdout_predicted_robust": int(pred.sum()),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-dir", type=Path, required=True)
    parser.add_argument("--putnam-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(read_shards(args.math_dir), read_shards(args.putnam_dir), args.bootstrap)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
