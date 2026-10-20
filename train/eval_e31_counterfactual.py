"""Evaluate E31's pre-registered counterfactual-geometry signals.

The family is fixed in ``datagen/counterfactual_geometry.py``.  Signals are ranked within
model to remove architecture scale, negated because the registered prediction is lower
sensitivity for robust rows, and pooled.  Inference clusters the bootstrap by near-duplicate template and
uses within-model label permutations with a max-T correction across all five signals.

Gate for considering a Codabench submission (fixed before downloading E31 outputs):

* selected pooled AUC >= .70 and problem-bootstrap 95% CI lower bound > .50;
* one-sided family-wise max-T p <= .05;
* worst per-model AUC among models with both classes >= .65;
* length-residualized AUC >= .65 and leave-one-model-out threshold BA >= .65.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_train_main_v2 import cluster_templates  # noqa: E402
SIGNALS = (
    "cf_cosine_tail_mean",
    "cf_cosine_tail_max",
    "cf_cosine_auc",
    "cf_cosine_slope",
    "cf_js_next_mean",
)
DEFAULT_GLOBS = (
    "runs/e31-counterfactual-a/features/e31-counterfactual-*/shard_*/records.jsonl",
    "runs/e31-counterfactual-b/features/e31-counterfactual-*/shard_*/records.jsonl",
)


def binary_auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_accuracy(labels, predictions) -> float:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    return float(0.5 * (predictions[labels].mean() + (~predictions[~labels]).mean()))


def within_model_scores(frame: pd.DataFrame, signal: str) -> np.ndarray:
    score = np.empty(len(frame), dtype=float)
    for _, indices in frame.groupby("model_id", sort=True).groups.items():
        loc = frame.index.get_indexer(indices)
        values = frame.loc[indices, signal].to_numpy(float)
        score[loc] = -rankdata(values, method="average") / len(values)
    return score


def clustered_ci(frame: pd.DataFrame, score: np.ndarray, rng, draws: int = 5000) -> tuple[float, float]:
    keys = np.asarray(sorted(frame.template.unique()))
    aucs: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        positions = np.concatenate(
            [np.flatnonzero(frame.template.to_numpy() == key) for key in sampled]
        )
        value = binary_auc(frame.is_robust.to_numpy()[positions], score[positions])
        if np.isfinite(value):
            aucs.append(value)
    return tuple(float(x) for x in np.quantile(aucs, [0.025, 0.975]))


def max_t_pvalues(frame: pd.DataFrame, score_matrix: np.ndarray, observed: np.ndarray, rng, draws=10000):
    labels = frame.is_robust.to_numpy(bool)
    groups = [frame.index.get_indexer(index) for _, index in frame.groupby("model_id").groups.items()]
    null_max = np.empty(draws, dtype=float)
    for draw in range(draws):
        permuted = labels.copy()
        for positions in groups:
            permuted[positions] = rng.permutation(permuted[positions])
        null_max[draw] = max(binary_auc(permuted, score_matrix[:, j]) for j in range(score_matrix.shape[1]))
    return np.asarray([(1 + np.sum(null_max >= value)) / (draws + 1) for value in observed])


def best_threshold(labels, scores) -> float:
    unique = np.unique(scores)
    candidates = np.r_[unique[0] - 1, (unique[:-1] + unique[1:]) / 2, unique[-1] + 1]
    values = [balanced_accuracy(labels, scores >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(values))])


def leave_one_model_threshold_ba(frame: pd.DataFrame, score: np.ndarray) -> float:
    predictions = np.zeros(len(frame), dtype=bool)
    labels = frame.is_robust.to_numpy(bool)
    for model in sorted(frame.model_id.unique()):
        test = frame.model_id.to_numpy() == model
        threshold = best_threshold(labels[~test], score[~test])
        predictions[test] = score[test] >= threshold
    return balanced_accuracy(labels, predictions)


def length_residual_auc(frame: pd.DataFrame, score: np.ndarray) -> tuple[float, float]:
    delta = (frame.max_rewrite_tokens - frame.n_prompt_tokens).to_numpy(float)
    adjusted = score.copy()
    for _, indices in frame.groupby("model_id").groups.items():
        positions = frame.index.get_indexer(indices)
        x = rankdata(delta[positions]) / len(positions)
        design = np.column_stack([np.ones(len(x)), x])
        adjusted[positions] -= design @ np.linalg.lstsq(design, score[positions], rcond=None)[0]
    return binary_auc(frame.is_robust, adjusted), float(spearmanr(score, delta).statistic)


def load_records(patterns: tuple[str, ...]) -> pd.DataFrame:
    paths = sorted({path for pattern in patterns for path in glob.glob(str(ROOT / pattern))})
    if not paths:
        raise SystemExit("no E31 records found; download both Kaggle outputs first")
    rows = [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows).sort_values(["model_id", "problem_key"]).reset_index(drop=True)
    if len(frame) != 34 or frame[["model_id", "problem_key"]].duplicated().any():
        raise SystemExit(f"expected 34 unique official model/problem rows, found {len(frame)}")
    if frame.truncated.any() or frame[list(SIGNALS)].isna().any().any():
        raise SystemExit("E31 output contains truncation or missing signals")

    official = pq.read_table(
        ROOT / "data/small_track/instances_train.parquet",
        columns=["model_id", "problem_key", "problem", "dataset_id", "is_robust"],
    ).to_pandas()
    official = official[official.dataset_id == "train-main-v2"]
    check = frame.merge(
        official,
        on=["model_id", "problem_key"],
        suffixes=("", "_official"),
        validate="one_to_one",
    )
    if len(check) != 34 or not np.array_equal(check.is_robust, check.is_robust_official):
        raise SystemExit("downloaded E31 labels do not match the frozen official table")
    check = check.drop(columns=["is_robust_official", "dataset_id_official"])
    normalized = check.problem.map(lambda value: " ".join(value.split()))
    templates = cluster_templates(sorted(normalized.unique()))
    check["template"] = normalized.map(templates)
    return check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()
    frame = load_records(DEFAULT_GLOBS)
    score_matrix = np.column_stack([within_model_scores(frame, signal) for signal in SIGNALS])
    labels = frame.is_robust.to_numpy(bool)
    aucs = np.asarray([binary_auc(labels, score_matrix[:, j]) for j in range(len(SIGNALS))])
    rng = np.random.default_rng(20260924)
    pvalues = max_t_pvalues(frame, score_matrix, aucs, rng)
    order = np.argsort(-aucs)
    rows = []
    for index in order:
        score = score_matrix[:, index]
        ci = clustered_ci(frame, score, rng)
        residual_auc, length_rho = length_residual_auc(frame, score)
        per_model = {
            model.split("/")[-1]: binary_auc(part.is_robust, score[part.index])
            for model, part in frame.groupby("model_id", sort=True)
        }
        rows.append(
            {
                "signal": SIGNALS[index],
                "auc": aucs[index],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "max_t_p": pvalues[index],
                "residual_auc": residual_auc,
                "length_rho": length_rho,
                "lomo_ba": leave_one_model_threshold_ba(frame, score),
                "per_model": per_model,
            }
        )

    winner = rows[0]
    finite_model_aucs = [value for value in winner["per_model"].values() if np.isfinite(value)]
    passed = (
        winner["auc"] >= 0.70
        and winner["ci_low"] > 0.50
        and winner["max_t_p"] <= 0.05
        and min(finite_model_aucs) >= 0.65
        and winner["residual_auc"] >= 0.65
        and winner["lomo_ba"] >= 0.65
    )
    print(f"E31: {len(frame)} rows, {frame.problem_key.nunique()} problems, {frame.template.nunique()} templates")
    for row in rows:
        models = ", ".join(f"{key}={value:.3f}" for key, value in row["per_model"].items())
        print(
            f"{row['signal']:24s} AUC={row['auc']:.3f} CI=[{row['ci_low']:.3f},{row['ci_high']:.3f}] "
            f"maxT-p={row['max_t_p']:.4f} residual={row['residual_auc']:.3f} "
            f"LOMO-BA={row['lomo_ba']:.3f} rho_len={row['length_rho']:.3f} | {models}"
        )
    print("GATE", "PASS" if passed else "FAIL", "for", winner["signal"])

    if args.write_results:
        result_path = ROOT / "RESULTS.md"
        text = result_path.read_text(encoding="utf-8").rstrip()
        marker = "## E31 — counterfactual activation sensitivity"
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
        table = [
            marker,
            "",
            "Pre-registered answer-preserving rewrite experiment on all 34 official small-track rows "
            f"({frame.problem_key.nunique()} problems, {frame.template.nunique()} near-duplicate templates). Signals are "
            "ranked within model; robust is predicted to have lower sensitivity. Confidence intervals "
            "bootstrap near-duplicate templates and p-values use within-model permutations with max-T correction across all five signals.",
            "",
            "| signal | pooled AUC | template-bootstrap 95% CI | max-T p | length-residual AUC | LOMO threshold BA |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            table.append(
                f"| `{row['signal']}` | {row['auc']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] | "
                f"{row['max_t_p']:.4f} | {row['residual_auc']:.3f} | {row['lomo_ba']:.3f} |"
            )
        table += [
            "",
            "Selected signal per-model AUCs: "
            + ", ".join(f"{key} {value:.3f}" for key, value in winner["per_model"].items())
            + ".",
            "",
            ("**Decision: E31 passes the pre-registered gate; package only after a GPU contract check.**" if passed else
             "**Decision: E31 fails the pre-registered gate and is not submitted to Codabench.**"),
            "",
        ]
        result_path.write_text(text + "\n\n" + "\n".join(table), encoding="utf-8")


if __name__ == "__main__":
    main()
