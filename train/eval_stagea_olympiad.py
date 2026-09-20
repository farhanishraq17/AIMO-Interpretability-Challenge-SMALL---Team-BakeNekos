"""E21: do prompt hidden states carry olympiad robustness signal when the text block does not?

The text block sits at AUC 0.617 against the organizers' olympiad labels with a CI spanning 0.5,
and retraining it on olympiad labels rather than MATH changes nothing (Findings 37, 39). The
prompt-hidden-state block is the only part of the shipped model never measured out of
population: E1b scored it on Hendrycks MATH, where it is the block that earns its place.

The split is fixed by construction, not by a fold assignment: probes are fitted on the 26
answer-modulus self-labels and scored on the 12 officially labelled `train-main-v2` rows, which
`build_stagea_olympiad_set.py` guarantees are disjoint by `problem_key`.

The ensemble follows E11 exactly -- one standardised L2 probe per (pooling, layer), averaging
probabilities over all 99 of them -- because Finding 11 showed that picking a layer overfits
badly at this sample size. At n=26 even the ensemble is fragile, so every number is reported
with a bootstrap CI over the 12 evaluation rows and compared against the text block on the
same rows.

Writes the `E21: Stage A on olympiad labels` section of RESULTS.md.

Run:  uv run python ../train/eval_stagea_olympiad.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from results_io import upsert_section  # noqa: E402

FEATURES = ROOT / "runs" / "e21-stagea-qwen35-olympiad" / "features" / "stagea-qwen35-olympiad"
ARTIFACT = ROOT / "submission" / "e11-text-stagea" / "artifacts" / "e11_deepseek.joblib"
# Carries the statements for both groups; the self-label table holds only the training half.
INSTANCES = ROOT / "data" / "stagea_olympiad" / "instances.parquet"
RESULTS = ROOT / "RESULTS.md"

TRAIN_GROUP = "qwen35-self-labels"
EVAL_GROUP = "train-main-v2"
PROBE_C = 0.01  # the value E11 settled on; not re-tuned here, there is no data to tune it with


def ensemble_probabilities(
    train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray
) -> np.ndarray:
    """Mean probability over one probe per (pooling, layer). Shapes: (n, n_layers, hidden)."""
    probabilities = []
    for layer in range(train_x.shape[1]):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=PROBE_C, class_weight="balanced", max_iter=5000),
        )
        probe.fit(train_x[:, layer, :].astype(np.float64), train_y)
        probabilities.append(probe.predict_proba(eval_x[:, layer, :].astype(np.float64))[:, 1])
    return np.mean(probabilities, axis=0)


def bootstrap_auc(y: np.ndarray, p: np.ndarray, rng, n_boot: int) -> tuple[float, float]:
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1:
            draws.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, default=FEATURES)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--n-refits", type=int, default=40,
                        help="shuffled-training-label refits; each is 99 logistic fits, so this is slow")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = np.load(args.features_dir / "features.npz")
    records = pd.read_parquet(args.features_dir / "records.parquet")
    poolings = sorted(name.removeprefix("X_") for name in payload.files if name.startswith("X_"))

    is_train = (records["dataset_id"] == TRAIN_GROUP).to_numpy()
    is_eval = (records["dataset_id"] == EVAL_GROUP).to_numpy()
    y_train = records["is_robust"].to_numpy()[is_train].astype(bool)
    y_eval = records["is_robust"].to_numpy()[is_eval].astype(bool)
    if set(records["problem_key"][is_train]) & set(records["problem_key"][is_eval]):
        raise SystemExit("train and eval groups overlap; the evaluation would leak")

    rng = np.random.default_rng(args.seed)
    per_pooling: dict[str, np.ndarray] = {}
    for pooling in poolings:
        x = payload[f"X_{pooling}"]
        per_pooling[pooling] = ensemble_probabilities(x[is_train], y_train, x[is_eval])
    stagea = np.mean(list(per_pooling.values()), axis=0)

    # The text block on the same 12 rows, as the reference every block is measured against.
    artifact = joblib.load(ARTIFACT)
    problems = pd.read_parquet(INSTANCES)
    key_to_problem = dict(zip(problems["problem_key"], problems["problem"]))
    missing = [k for k in records["problem_key"][is_eval] if k not in key_to_problem]
    if missing:
        raise SystemExit(f"{len(missing)} evaluation problems are not in {INSTANCES}")
    eval_problems = [key_to_problem[k] for k in records["problem_key"][is_eval]]
    text = artifact["text_model"].predict_proba(eval_problems)[:, 1]

    lines = [
        "> **Superseded by Finding 31.** The probes below are fitted on self-generated "
        "`answer_modulus` labels, which E20 showed do not measure official robustness "
        "(agreement 0.29, kappa -0.135). This section records what E21 measured; it is not a "
        "result to build on, and its borderline strict-control p-value is consistent with the "
        "labels carrying no signal.",
        "",
        f"Probes fitted on {int(is_train.sum())} answer-modulus self-labels, scored on "
        f"{int(is_eval.sum())} officially labelled `train-main-v2` rows for `Qwen/Qwen3.5-4B`, "
        "disjoint by `problem_key`. Regenerated by `train/eval_stagea_olympiad.py`.",
        "",
        f"Evaluation rows are {y_eval.mean():.3f} robust. The hidden validation set is close to "
        "balanced (Finding 27), so read ROC-AUC, not accuracy.",
        "",
        "| Block | ROC-AUC | 95% CI |",
        "| --- | --- | --- |",
    ]
    for pooling in poolings:
        low, high = bootstrap_auc(y_eval, per_pooling[pooling], rng, args.n_boot)
        lines.append(
            f"| Stage A, `{pooling}` pooling | {roc_auc_score(y_eval, per_pooling[pooling]):.3f} "
            f"| [{low:.3f}, {high:.3f}] |"
        )
    low, high = bootstrap_auc(y_eval, stagea, rng, args.n_boot)
    lines.append(f"| **Stage A, all poolings** | **{roc_auc_score(y_eval, stagea):.3f}** | [{low:.3f}, {high:.3f}] |")
    if text is not None:
        low, high = bootstrap_auc(y_eval, text, rng, args.n_boot)
        lines.append(f"| Text block (reference) | {roc_auc_score(y_eval, text):.3f} | [{low:.3f}, {high:.3f}] |")

        diffs = []
        for _ in range(args.n_boot):
            idx = rng.integers(0, len(y_eval), len(y_eval))
            if len(np.unique(y_eval[idx])) > 1:
                diffs.append(roc_auc_score(y_eval[idx], stagea[idx]) - roc_auc_score(y_eval[idx], text[idx]))
        low, high = np.percentile(diffs, [2.5, 97.5])
        verdict = "no difference detected" if low < 0 < high else "difference established"
        lines += [
            "",
            f"Paired bootstrap, Stage A minus text: **{np.mean(diffs):+.3f}**, "
            f"95% CI [{low:+.3f}, {high:+.3f}] -- {verdict}.",
        ]
    # Bootstrap CIs are unreliable at n=12 -- few distinct resamples, and they say nothing about
    # whether the probe learned anything. Two permutation controls do the real work, and the
    # second is the strict one: it refits on shuffled training labels, so its null keeps the
    # probe's capacity to fit noise at this sample size.
    observed = roc_auc_score(y_eval, stagea)
    eval_null = [roc_auc_score(rng.permutation(y_eval), stagea) for _ in range(args.n_permutations)]
    eval_p = (sum(value >= observed for value in eval_null) + 1) / (args.n_permutations + 1)

    train_null = np.array([
        roc_auc_score(
            y_eval,
            np.mean(
                [
                    ensemble_probabilities(payload[f"X_{p}"][is_train], rng.permutation(y_train),
                                           payload[f"X_{p}"][is_eval])
                    for p in poolings
                ],
                axis=0,
            ),
        )
        for _ in range(args.n_refits)
    ])
    train_p = (int((train_null >= observed).sum()) + 1) / (len(train_null) + 1)

    lines += [
        "",
        "### Controls",
        "",
        "| Null | Null mean | 95th pct | p |",
        "| --- | --- | --- | --- |",
        f"| Permuted **evaluation** labels ({args.n_permutations} draws) | {np.mean(eval_null):.3f} "
        f"| {np.percentile(eval_null, 95):.3f} | **{eval_p:.3f}** |",
        f"| Shuffled **training** labels, refitted ({args.n_refits} draws) | {train_null.mean():.3f} "
        f"| {np.percentile(train_null, 95):.3f} | **{train_p:.3f}** |",
        "",
        f"The second control is the one to believe: a probe fitted on random labels still reaches "
        f"{np.percentile(train_null, 95):.3f} five percent of the time at this sample size, so the "
        f"observed {observed:.3f} sits inside its tail. **Suggestive, not established.**",
        "",
        f"At n={int(is_eval.sum())} this can falsify a block, not confirm one. Per-pooling rows are "
        "reported for spread only -- the all-pooling ensemble is the pre-specified estimator, and "
        "picking the best pooling on 12 rows repeats the selection effect of Finding 11.",
    ]
    upsert_section(RESULTS, "E21: Stage A on olympiad labels", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
