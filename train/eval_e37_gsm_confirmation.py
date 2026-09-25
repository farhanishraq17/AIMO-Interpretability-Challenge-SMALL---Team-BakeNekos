"""Evaluate E37 on untouched GSM-Plus outcomes with E36's frozen score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from eval_e36_budget_path import (
    BUDGETS,
    LABEL_BUDGET,
    build_examples,
    feature_columns,
    grouped_auc_ci,
    make_model,
    read_shards,
    trace_features,
)


ANCHOR = "problem understanding"
EXCLUDED = "critical thinking"


def build_gsm_examples(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model", "token_budget", "problem_id", "variant", "predicted_answer",
        "correct_strict", "answer_forced", "trace",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = frame.loc[frame["token_budget"].isin(BUDGETS)].copy()
    try:
        numeric_id = frame["problem_id"].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("GSM-Plus problem IDs must be integers") from exc
    frame["canonical_id"] = (numeric_id // 8).astype(str)
    rows = []
    for (problem_id, model), part in frame.groupby(["canonical_id", "model"], sort=True):
        at_label = part.loc[part["token_budget"] == LABEL_BUDGET]
        variants = at_label.loc[at_label["variant"] != EXCLUDED]
        if len(variants) != 7 or variants["variant"].nunique() != 7:
            raise ValueError(f"{problem_id}/{model}: expected seven distinct scorable variants")
        anchor_label = variants.loc[variants["variant"] == ANCHOR, "correct_strict"]
        if len(anchor_label) != 1:
            raise ValueError(f"{problem_id}/{model}: expected one anchor")
        if not bool(anchor_label.iloc[0]):
            continue
        correct = int(variants["correct_strict"].astype(bool).sum())
        if correct == 6:
            continue
        robust = correct == 7
        anchor = part.loc[part["variant"] == ANCHOR]
        rows.append({
            "problem_id": problem_id,
            "model": str(model),
            "is_robust": robust,
            **trace_features(anchor),
        })
    return pd.DataFrame(rows)


def half_rank_by_model(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    predictions = np.zeros(len(frame), dtype=bool)
    for _, part in frame.groupby("model", sort=True):
        index = part.index.to_numpy()
        order = np.lexsort((part["problem_id"].to_numpy(str), scores[index]))
        predictions[index[order[len(order) // 2 :]]] = True
    return predictions


def evaluate(math_frame: pd.DataFrame, gsm_frame: pd.DataFrame, bootstrap: int = 5_000) -> dict:
    train = build_examples(math_frame, "math_perturb")
    holdout = build_gsm_examples(gsm_frame)
    columns = feature_columns(train)
    if columns != feature_columns(holdout):
        raise ValueError("feature columns differ between MATH-Perturb and GSM-Plus")
    model = make_model()
    model.fit(train[columns], train["is_robust"].to_numpy(bool))
    scores = model.predict_proba(holdout[columns])[:, 1]
    y = holdout["is_robust"].to_numpy(bool)
    predictions = half_rank_by_model(holdout, scores)
    auc = float(roc_auc_score(y, scores))
    ci = grouped_auc_ci(holdout, scores, bootstrap)
    balanced = float(balanced_accuracy_score(y, predictions))
    by_model = {}
    for model_name, part in holdout.groupby("model", sort=True):
        index = part.index.to_numpy()
        local_y = part["is_robust"].to_numpy(bool)
        by_model[str(model_name)] = None if np.unique(local_y).size < 2 else float(
            roc_auc_score(local_y, scores[index])
        )
    counts = {
        "n": int(len(holdout)),
        "problems": int(holdout["problem_id"].nunique()),
        "robust": int(y.sum()),
        "spurious": int((~y).sum()),
    }
    valid_model_aucs = [value for value in by_model.values() if value is not None]
    gates = {
        "at_least_50_each": min(counts["robust"], counts["spurious"]) >= 50,
        "auc_at_least_0_65": auc >= 0.65,
        "auc_ci_lower_above_0_50": ci[0] > 0.50,
        "balanced_accuracy_at_least_0_60": balanced >= 0.60,
        "all_model_aucs_at_least_0_50": bool(valid_model_aucs) and min(valid_model_aucs) >= 0.50,
        "three_model_aucs_at_least_0_58": sum(value >= 0.58 for value in valid_model_aucs) >= 3,
    }
    return {
        "holdout": counts,
        "holdout_auc": auc,
        "holdout_auc_ci95_problem_bootstrap": list(ci),
        "holdout_half_rank_balanced_accuracy": balanced,
        "model_aucs": by_model,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-dir", type=Path, required=True)
    parser.add_argument("--gsm-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(read_shards(args.math_dir), read_shards(args.gsm_dir), args.bootstrap)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
