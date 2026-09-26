"""Evaluate the pre-registered E36 budget-path reasoning diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BUDGETS = (2_048, 4_096, 8_192, 16_384)
LABEL_BUDGET = 16_384
SEED = 36
MARKERS = ("wait", "however", "alternatively", "instead", "check", "verify", "reconsider", "mistake")


def read_shards(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet shards in {directory}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def canonical_problem_id(value: object, variant: object, source: str) -> str:
    value, variant = str(value), str(variant)
    if source == "math_perturb":
        suffix = f"#{variant}"
        if not value.endswith(suffix):
            raise ValueError(f"unexpected MATH-Perturb problem_id: {value!r}")
        return value[: -len(suffix)]
    if source == "putnam_axiom":
        pattern = r"^putnam_orig_(.+::.+)$" if variant == "original" else r"^putnam_var_(.+::.+)_\d+$"
        match = re.fullmatch(pattern, value)
        if match is None:
            raise ValueError(f"unexpected Putnam problem_id: {value!r}")
        return "putnam_" + match.group(1)
    raise ValueError(source)


def normalize_answer(value: object) -> str:
    text = "" if value is None else str(value).lower()
    return re.sub(r"[\s$\[\]{},]", "", text)


def repeated_line_fraction(trace: str) -> float:
    lines = [re.sub(r"\s+", " ", line.strip().lower()) for line in str(trace).splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def trace_features(part: pd.DataFrame) -> dict[str, float]:
    rows = {}
    for budget in BUDGETS:
        selected = part.loc[part["token_budget"] == budget]
        if len(selected) != 1:
            raise ValueError(f"expected one original row at budget {budget}, found {len(selected)}")
        rows[budget] = selected.iloc[0]

    answers = [normalize_answer(rows[b]["predicted_answer"]) for b in BUDGETS]
    nonempty = [answer for answer in answers if answer]
    counts = Counter(nonempty)
    modal_fraction = max(counts.values()) / len(nonempty) if nonempty else 0.0
    unique_fraction = len(counts) / len(nonempty) if nonempty else 1.0
    adjacent = [answers[i] == answers[i + 1] and bool(answers[i]) for i in range(len(answers) - 1)]
    final_matches = [answer == answers[-1] and bool(answer) for answer in answers[:-1]]
    natural = [not bool(rows[b]["answer_forced"]) for b in BUDGETS]
    first_natural = next((math.log2(budget) for budget, flag in zip(BUDGETS, natural) if flag), 15.0)
    words = [len(str(rows[b]["trace"]).split()) for b in BUDGETS]
    final_trace = str(rows[BUDGETS[-1]]["trace"]).lower()
    marker_count = sum(len(re.findall(rf"\b{re.escape(marker)}\b", final_trace)) for marker in MARKERS)
    final_words = max(words[-1], 1)

    result: dict[str, float] = {}
    for budget, flag, word_count in zip(BUDGETS, natural, words):
        result[f"natural_{budget}"] = float(flag)
        result[f"log_words_{budget}"] = math.log1p(word_count)
    result.update(
        natural_fraction=float(np.mean(natural)),
        first_natural_log2=float(first_natural),
        answer_nonempty_fraction=float(np.mean([bool(answer) for answer in answers])),
        answer_modal_fraction=float(modal_fraction),
        answer_unique_fraction=float(unique_fraction),
        answer_adjacent_agreement=float(np.mean(adjacent)),
        answer_final_agreement=float(np.mean(final_matches)),
        trace_growth_8k_16k=float((words[-1] + 1) / (words[-2] + 1)),
        correction_markers_per_1k=float(1_000 * marker_count / final_words),
        repeated_line_fraction_16k=float(repeated_line_fraction(final_trace)),
    )
    return result


def build_examples(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {
        "model", "token_budget", "problem_id", "variant", "predicted_answer",
        "correct_strict", "answer_forced", "trace",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = frame.loc[frame["token_budget"].isin(BUDGETS)].copy()
    frame["canonical_id"] = [
        canonical_problem_id(value, variant, source)
        for value, variant in zip(frame["problem_id"], frame["variant"])
    ]
    original_name = "base" if source == "math_perturb" else "original"
    rows = []
    for (problem_id, model), part in frame.groupby(["canonical_id", "model"], sort=True):
        at_label = part.loc[part["token_budget"] == LABEL_BUDGET]
        original_label = at_label.loc[at_label["variant"] == original_name, "correct_strict"]
        if len(original_label) != 1:
            raise ValueError(f"{problem_id}/{model}: expected one label-budget original")
        if not bool(original_label.iloc[0]):
            continue
        if source == "math_perturb":
            variants = at_label.set_index("variant")["correct_strict"]
            if not {"simple", "hard"}.issubset(variants.index):
                raise ValueError(f"{problem_id}/{model}: missing MATH variants")
            robust = bool(variants["simple"] and variants["hard"])
        else:
            variants = at_label.loc[at_label["variant"] == "variation", "correct_strict"]
            if variants.empty:
                raise ValueError(f"{problem_id}/{model}: missing Putnam variations")
            robust = bool(variants.astype(bool).all())
        original = part.loc[part["variant"] == original_name]
        rows.append({
            "problem_id": problem_id,
            "model": str(model),
            "is_robust": robust,
            **trace_features(original),
        })
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"problem_id", "model", "is_robust"}]


def make_model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, random_state=SEED),
    )


def choose_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)
    values = np.array([balanced_accuracy_score(y, scores >= threshold) for threshold in candidates])
    return float(candidates[np.flatnonzero(values == values.max())[-1]])


def fit_predict(train: pd.DataFrame, holdout: pd.DataFrame) -> tuple[np.ndarray, float]:
    columns = feature_columns(train)
    if columns != feature_columns(holdout):
        raise ValueError("feature columns differ between sources")
    x, y = train[columns], train["is_robust"].to_numpy(bool)
    groups = train["problem_id"].to_numpy(str)
    oof = np.full(len(train), np.nan)
    for fit_index, score_index in GroupKFold(5).split(x, y, groups):
        model = make_model()
        model.fit(x.iloc[fit_index], y[fit_index])
        oof[score_index] = model.predict_proba(x.iloc[score_index])[:, 1]
    threshold = choose_threshold(y, oof)
    model = make_model()
    model.fit(x, y)
    return model.predict_proba(holdout[columns])[:, 1], threshold


def grouped_auc_ci(frame: pd.DataFrame, scores: np.ndarray, replicates: int) -> tuple[float, float]:
    ids = frame["problem_id"].drop_duplicates().to_numpy(str)
    indices = {problem_id: np.flatnonzero(frame["problem_id"].to_numpy(str) == problem_id) for problem_id in ids}
    rng = np.random.default_rng(SEED)
    values = []
    y = frame["is_robust"].to_numpy(bool)
    for _ in range(replicates):
        sampled = rng.choice(ids, len(ids), replace=True)
        selected = np.concatenate([indices[problem_id] for problem_id in sampled])
        if np.unique(y[selected]).size == 2:
            values.append(roc_auc_score(y[selected], scores[selected]))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def evaluate(math_frame: pd.DataFrame, putnam_frame: pd.DataFrame, bootstrap: int = 5_000) -> dict:
    train = build_examples(math_frame, "math_perturb")
    holdout = build_examples(putnam_frame, "putnam_axiom")
    scores, threshold = fit_predict(train, holdout)
    y = holdout["is_robust"].to_numpy(bool)
    auc = float(roc_auc_score(y, scores))
    ci = grouped_auc_ci(holdout, scores, bootstrap)
    balanced = float(balanced_accuracy_score(y, scores >= threshold))
    by_model = {}
    for model, part in holdout.groupby("model", sort=True):
        index = part.index.to_numpy()
        local_y = part["is_robust"].to_numpy(bool)
        by_model[str(model)] = None if np.unique(local_y).size < 2 else float(roc_auc_score(local_y, scores[index]))
    counts = lambda frame: {
        "n": int(len(frame)),
        "problems": int(frame["problem_id"].nunique()),
        "robust": int(frame["is_robust"].sum()),
        "spurious": int((~frame["is_robust"]).sum()),
    }
    train_counts, holdout_counts = counts(train), counts(holdout)
    valid_model_aucs = [value for value in by_model.values() if value is not None]
    gates = {
        "train_at_least_100_each": min(train_counts["robust"], train_counts["spurious"]) >= 100,
        "holdout_at_least_40_each": min(holdout_counts["robust"], holdout_counts["spurious"]) >= 40,
        "auc_at_least_0_65": auc >= 0.65,
        "auc_ci_lower_above_0_50": ci[0] > 0.50,
        "balanced_accuracy_at_least_0_60": balanced >= 0.60,
        "all_model_aucs_at_least_0_50": bool(valid_model_aucs) and min(valid_model_aucs) >= 0.50,
        "three_model_aucs_at_least_0_58": sum(value >= 0.58 for value in valid_model_aucs) >= 3,
    }
    diagnostics = {}
    for column in feature_columns(train):
        values = holdout[column].to_numpy(float)
        diagnostics[column] = None if np.unique(values).size < 2 else float(roc_auc_score(y, values))
    return {
        "train": train_counts,
        "holdout": holdout_counts,
        "feature_count": len(feature_columns(train)),
        "oof_threshold": threshold,
        "holdout_auc": auc,
        "holdout_auc_ci95_problem_bootstrap": list(ci),
        "holdout_balanced_accuracy": balanced,
        "model_aucs": by_model,
        "diagnostic_feature_aucs_unoriented": diagnostics,
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
