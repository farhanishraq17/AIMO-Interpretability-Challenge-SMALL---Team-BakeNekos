"""Grouped-CV classifiers on Stage B generation-trace features.

The uncertainty-profiling baseline summarises one greedy generation with 14 token-level
uncertainty statistics. It is the only published method well above chance: 69.2% on the
proposal's split, 0.6262 grouped out-of-fold balanced accuracy in its own artifact.
`extract_generation_features.py` recomputes those features on our consolidated instances;
this script scores them on the same problem-grouped folds as every other method, so the
comparison with the 0.6732 text floor is like for like.

Beyond the 14 features it adds trace-shape features read from the generated text: length,
token-cap hits, reflection markers such as "wait", and whether an answer was boxed. It also
tries a leak-free late fusion with the char tf-idf text model, averaging each model's
out-of-fold probabilities with fixed equal weights, so nothing is fit on a validation fold.

Run (after `kaggle_launch.py output` for both e2 shards):

    uv run python ../train/train_stageb_features.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from results_io import METRICS_HEADER, metrics_row, score, upsert_section

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "getting-started" / "solutions" / "uncertainty-profiling"))
from uncertainty_profile.config import FEATURE_NAMES  # noqa: E402

# fp16 reruns only: bf16 and fp16 greedy traces diverge (e2-bench-dtype), so never glob both.
DEFAULT_GLOB = str(ROOT / "runs" / "e2f-stageb-fp16-s*" / "features" / "stageb-fp16-deepseek" / "shard_*" / "records.jsonl")
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
SPLITS = ROOT / "data" / "small_track" / "splits.json"
TEXT_FLOOR_BALANCED_ACCURACY = 0.6732

REFLECTION_MARKERS = ("wait", "hmm", "alternatively", "let me check", "double-check", "re-examine", "but actually")


def trace_features(text: str) -> dict[str, float]:
    """Shape of the reasoning trace, readable at inference from our own generation."""
    lower = (text or "").lower()
    n_words = max(len(lower.split()), 1)
    reflections = sum(lower.count(marker) for marker in REFLECTION_MARKERS)
    return {
        "trace_log_chars": float(np.log1p(len(lower))),
        "trace_has_boxed": float("\\boxed" in lower),
        "trace_closed_think": float("</think>" in lower),
        "trace_reflections": float(reflections),
        "trace_reflections_per_1k_words": 1000.0 * reflections / n_words,
    }


def oof_proba(build, X, y, folds) -> np.ndarray:
    proba = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        val = folds == fold
        train_X = X[~val] if not isinstance(X, list) else [x for x, v in zip(X, val) if not v]
        val_X = X[val] if not isinstance(X, list) else [x for x, v in zip(X, val) if v]
        proba[val] = build().fit(train_X, y[~val]).predict_proba(val_X)[:, 1]
    return proba


MODELS = {
    "logreg": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000),
    ),
    "hist-gradient-boosting": lambda: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=3, class_weight="balanced", random_state=0
    ),
    "random-forest": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample", random_state=0, n_jobs=-1
        ),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records-glob", default=DEFAULT_GLOB)
    parser.add_argument("--results", type=Path, default=ROOT / "RESULTS.md")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.records_glob))
    if not paths:
        raise SystemExit(f"no records match {args.records_glob}")
    rows = [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df = pd.DataFrame(rows).drop_duplicates("problem_key", keep="last").reset_index(drop=True)
    if (df["model_id"] != df["extracted_model_id"]).any():
        raise SystemExit("records carry another model's labels; Stage B scoring needs --filter-model runs")
    model_id = df["extracted_model_id"].iloc[0]

    instances = pd.read_parquet(INSTANCES)
    expected = instances[instances["model_id"] == model_id]
    missing = sorted(set(expected["problem_key"]) - set(df["problem_key"]))
    print(f"{len(paths)} shard files, {len(df)} instances for {model_id}; {len(missing)} of {len(expected)} not yet generated")

    df = df.merge(expected[["problem_key", "problem"]], on="problem_key", how="inner")
    fold_map = json.loads(SPLITS.read_text(encoding="utf-8"))["problem_fold"]
    df["fold"] = df["problem_key"].map(fold_map)
    df = df.dropna(subset=["fold"]).reset_index(drop=True)
    y = df["is_robust"].astype(bool).to_numpy()
    folds = df["fold"].astype(int).to_numpy()

    traces = pd.DataFrame([trace_features(t) for t in df["generated_text"]])
    df = pd.concat([df, traces], axis=1)
    df["log_generation_num_tokens"] = np.log1p(df["generation_num_tokens"].astype(float))
    df["hit_token_cap"] = df["hit_token_cap"].astype(float)

    feature_sets = {
        "14 uncertainty": list(FEATURE_NAMES),
        "14 uncertainty + length": [*FEATURE_NAMES, "log_generation_num_tokens", "hit_token_cap"],
        "14 uncertainty + length + trace shape": [
            *FEATURE_NAMES, "log_generation_num_tokens", "hit_token_cap", *traces.columns,
        ],
    }

    results: list[tuple[str, dict]] = []
    oof: dict[str, np.ndarray] = {}
    for set_name, columns in feature_sets.items():
        X = df[columns].to_numpy(dtype=np.float64)
        for model_name, build in MODELS.items():
            name = f"{model_name} on {set_name}"
            oof[name] = oof_proba(build, X, y, folds)
            results.append((name, score(y, oof[name])))
            print(f"{name:60s} bal_acc={results[-1][1]['balanced_accuracy']:.4f} auc={results[-1][1]['roc_auc']:.4f}")

    text_build = lambda: make_pipeline(  # noqa: E731 - same config as the text floor
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
    )
    text_oof = oof_proba(text_build, df["problem"].tolist(), y, folds)
    text_metrics = score(y, text_oof)

    best_name, best_metrics = max(results, key=lambda item: item[1]["roc_auc"])
    fused_metrics = score(y, (oof[best_name] + text_oof) / 2)
    print(f"\ntext (this subset) bal_acc={text_metrics['balanced_accuracy']:.4f}")
    print(f"fusion(text, {best_name}) bal_acc={fused_metrics['balanced_accuracy']:.4f} auc={fused_metrics['roc_auc']:.4f}")

    univariate = []
    for column in feature_sets["14 uncertainty + length + trace shape"]:
        values = df[column].astype(float)
        mask = values.notna().to_numpy()
        if mask.sum() > 10 and len(set(y[mask].tolist())) > 1:
            auc = roc_auc_score(y[mask], values[mask])
            univariate.append((column, auc))
    univariate.sort(key=lambda item: -abs(item[1] - 0.5))

    by_label = df.groupby("is_robust")["generation_num_tokens"].median()
    lines = [
        f"Model: `{model_id}`  ",
        f"Instances: {len(df)} ({int(y.sum())} robust / {int((~y).sum())} spurious)"
        + (f", **{len(missing)} still missing**" if missing else "") + "  ",
        f"Traces: greedy, 4096-token cap; {df['hit_token_cap'].mean():.1%} hit the cap; median tokens "
        f"robust={by_label.get(True, float('nan')):.0f} vs spurious={by_label.get(False, float('nan')):.0f}  ",
        "Evaluation: out-of-fold over the problem-grouped folds; fusion averages OOF probabilities with fixed weights",
        "",
        *METRICS_HEADER,
        metrics_row("always-spurious (majority)", score(y, np.zeros(len(y)))),
        metrics_row("logreg on tf-idf chars (text floor, this subset)", text_metrics),
        *[metrics_row(name, metrics) for name, metrics in results],
        metrics_row(f"**fusion: text + {best_name}**", fused_metrics),
        "",
        "The best trace model is picked by AUC across the rows above, so the fusion row carries "
        "a small selection effect; treat differences under ~0.03 as noise at this sample size.",
        "",
        "### Univariate AUC per feature (direction: >0.5 means higher values lean robust)",
        "",
        "| Feature | AUC |",
        "| --- | --- |",
        *[f"| `{name}` | {auc:.3f} |" for name, auc in univariate[:12]],
        "",
        f"Reference: text floor on the full 138 is {TEXT_FLOOR_BALANCED_ACCURACY:.4f} balanced accuracy. "
        "Regenerate with `train/train_stageb_features.py`.",
    ]
    upsert_section(args.results, "Stage B: generation-trace features", "\n".join(lines))


if __name__ == "__main__":
    main()
