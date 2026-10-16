"""E57: train the per-problem probe on full-budget self-labels, and apply the pre-registered ship rule.

Evaluation is per-problem (Finding 79), so the submission needs an absolute decision for one problem.
This trains Finding 11's layer ensemble on E55's one-pass features (Finding 81) against the
collaborator's self-labels, under `protocols/E57_PROTOCOL.md`, which fixed the model, the nested
threshold rule, the grouping, the baselines and the ship rule before any self-label existed.

* Groups: problems sharing at least 80% of word 3-grams are one group (near-duplicates across sources).
* The 36 `train-main-v2` problems never enter training; their official labels are read once, last.
* Per model: per-layer standardised L2 logistic (C = 0.01, balanced classes) on mean-pooled states,
  averaged over layers; threshold = balanced-accuracy maximiser on inner out-of-fold probabilities.
* Held-out decisions pooled over models, against the per-model training majority and `e27f`'s constants.

`--synthetic` draws random labels for every pair and reads no real label: a plumbing test whose
held-out balanced accuracy must sit near 0.5 (protocol §6).

Writes the `E57: per-problem probe on self-labels` section of RESULTS.md (real labels only).

Run:  python train/train_e57_probe.py --labels <self_labels.parquet>
      python train/train_e57_probe.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from results_io import upsert_section  # noqa: E402

SHORT = {"Qwen/Qwen3.5-4B": "qwen", "Skywork/Skywork-OR1-Math-7B": "skywork",
         "allenai/Olmo-3-7B-Think": "olmo", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek"}
CONSTANTS = {"Qwen/Qwen3.5-4B": True, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": True,
             "Skywork/Skywork-OR1-Math-7B": False, "allenai/Olmo-3-7B-Think": False}
INSTANCES = ROOT / "data" / "e55" / "instances.parquet"
FEATURES = "runs/e55-features-{short}-*/e55-features-{short}/stagea-{short}/features.npz"
SEED, FOLDS, INNER, BOOT, C = 57, 5, 3, 2_000, 0.01


def groups(problems: pd.DataFrame) -> dict[str, str]:
    """Union of problems whose word 3-gram sets overlap by at least 80% (Jaccard)."""
    grams = {}
    for row in problems.itertuples():
        words = re.findall(r"[a-z0-9]+", row.problem.lower())
        grams[row.problem_key] = {tuple(words[i:i + 3]) for i in range(max(len(words) - 2, 1))}
    keys, parent = list(grams), {k: k for k in grams}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            union = len(grams[a] | grams[b])
            if union and len(grams[a] & grams[b]) / union >= 0.8:
                parent[find(a)] = find(b)
    return {k: find(k) for k in keys}


def load_features(model: str) -> tuple[np.ndarray, dict[str, int]]:
    paths = sorted(ROOT.glob(FEATURES.format(short=SHORT[model])))
    assert len(paths) == 1, f"{model}: expected one E55 feature file, found {paths}"
    bundle = np.load(paths[0], allow_pickle=True)
    return bundle["X_mean"], {str(k): i for i, k in enumerate(bundle["problem_keys"])}


def fit_probe(X: np.ndarray, y: np.ndarray) -> list:
    return [make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2_000))
            .fit(X[:, layer].astype(np.float32), y) for layer in range(X.shape[1])]


def predict_probe(models: list, X: np.ndarray) -> np.ndarray:
    return np.mean([m.predict_proba(X[:, layer].astype(np.float32))[:, 1] for layer, m in enumerate(models)], axis=0)


def ba_threshold(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.unique(p)
    scores = [balanced_accuracy_score(y, p >= t) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def nested_threshold(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    """Threshold from inner out-of-fold probabilities of the training rows only."""
    oof = np.zeros(len(y))
    for fit, score in StratifiedGroupKFold(INNER, shuffle=True, random_state=SEED).split(X[:, 0], y, g):
        oof[score] = predict_probe(fit_probe(X[fit], y[fit]), X[score])
    return ba_threshold(y, oof)


def group_bootstrap(frame: pd.DataFrame, statistic) -> tuple[float, float]:
    rng, ids = np.random.default_rng(SEED), frame["group"].unique()
    index = {g: np.flatnonzero(frame["group"].to_numpy() == g) for g in ids}
    values = []
    for _ in range(BOOT):
        pick = np.concatenate([index[g] for g in rng.choice(ids, len(ids))])
        try:
            values.append(statistic(frame.iloc[pick]))
        except ValueError:
            continue
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def evaluate(labels: pd.DataFrame, official_keys: set[str]) -> tuple[pd.DataFrame, dict]:
    """Held-out decisions for every labelled training pair, by grouped outer folds shared across models."""
    train = labels[~labels["problem_key"].isin(official_keys)].copy()
    problem_groups = train.drop_duplicates("problem_key")[["problem_key", "group"]]
    problem_y = train.groupby("problem_key")["is_robust"].mean().round().astype(int)
    folds = {}
    for k, (_, test) in enumerate(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED).split(
            problem_groups, problem_y.loc[problem_groups["problem_key"]], problem_groups["group"])):
        folds.update({key: k for key in problem_groups["problem_key"].iloc[test]})
    train["fold"] = train["problem_key"].map(folds)
    rows = []
    for model, part in train.groupby("model_id"):
        X_all, index = load_features(model)
        X = X_all[[index[k] for k in part["problem_key"]]]
        y, g, f = part["is_robust"].astype(int).to_numpy(), part["group"].to_numpy(), part["fold"].to_numpy()
        for k in range(FOLDS):
            fit, held = f != k, f == k
            if not held.any() or len(np.unique(y[fit])) < 2:
                continue
            t = nested_threshold(X[fit], y[fit], g[fit])
            p = predict_probe(fit_probe(X[fit], y[fit]), X[held])
            majority = bool(y[fit].mean() >= 0.5)
            rows += [{"model_id": model, "problem_key": key, "group": grp, "y": bool(yy), "probe": bool(pp >= t),
                      "majority": majority, "constants": CONSTANTS[model]}
                     for key, grp, yy, pp in zip(part["problem_key"].to_numpy()[held], g[held], y[held], p)]
    held = pd.DataFrame(rows)
    acc = lambda col: (lambda fr: accuracy_score(fr["y"], fr[col]))  # noqa: E731
    bal = lambda col: (lambda fr: balanced_accuracy_score(fr["y"], fr[col]))  # noqa: E731
    diff = lambda a, b: (lambda fr: accuracy_score(fr["y"], fr[a]) - accuracy_score(fr["y"], fr[b]))  # noqa: E731
    summary = {
        "rows": len(held), "problems": held["problem_key"].nunique(), "robust_share": float(held["y"].mean()),
        "probe_balanced": [float(bal("probe")(held)), *group_bootstrap(held, bal("probe"))],
        **{f"{c}_accuracy": [float(acc(c)(held)), *group_bootstrap(held, acc(c))] for c in ("probe", "majority", "constants")},
        **{f"probe_minus_{b}": [float(diff("probe", b)(held)), *group_bootstrap(held, diff("probe", b))]
           for b in ("majority", "constants")},
    }
    return held, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, help="self-labels: model_id, problem_key, is_robust (null = band)")
    parser.add_argument("--synthetic", action="store_true", help="random labels; reads no real label")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if bool(args.labels) == args.synthetic:
        parser.error("give exactly one of --labels or --synthetic")

    instances = pd.read_parquet(INSTANCES)
    official_keys = set(instances.loc[instances["dataset_id"] == "train-main-v2", "problem_key"])
    problem_group = groups(instances.drop_duplicates("problem_key"))
    if args.synthetic:
        rng = np.random.default_rng(SEED)
        labels = instances[["model_id", "problem_key"]].copy()
        labels["is_robust"] = rng.random(len(labels)) < 0.6
    else:
        labels = pd.read_parquet(args.labels) if args.labels.suffix == ".parquet" else pd.read_csv(args.labels)
        labels = labels.dropna(subset=["is_robust"])
        unknown = set(labels["problem_key"]) - set(instances["problem_key"])
        assert not unknown, f"{len(unknown)} labelled problems have no E55 features"
    labels["group"] = labels["problem_key"].map(problem_group)

    held, summary = evaluate(labels, official_keys)
    print(json.dumps(summary, indent=2))
    rules = {
        "1_balanced_ci_above_0_5": summary["probe_balanced"][1] > 0.5,
        "2_beats_both_baselines": summary["probe_minus_majority"][1] > 0 and summary["probe_minus_constants"][1] > 0,
    }
    print("ship rules 1-2:", rules, "| rule 3 (official read) runs only on real labels that pass 1-2")
    if args.synthetic or args.no_write:
        return
    # Rule 3 and the RESULTS section are written once real labels exist; see protocol section 5.3.
    raise SystemExit("real-label path: rule 3 (one official read) and the RESULTS section are implemented "
                     "with the first real labels, as a dated amendment, before they are read")


if __name__ == "__main__":
    main()
