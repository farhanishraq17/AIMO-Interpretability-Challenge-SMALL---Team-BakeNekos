"""E10 fusion: problem text + prompt hidden states + trace hidden states + generation traces.

The question behind gate G1 (PLAN.md §5): do model-derived signals add anything to what the
problem text already gives? Stage A showed that an all-layer hidden-state ensemble out-ranks
text by +0.065 AUC, but with a CI spanning zero, and with probabilities off-centre enough
that a fixed 0.5 threshold wastes the gain (PROGRESS.md Finding 11). This script combines
the blocks without leaking labels.

Everything is nested, so no label from a held-out fold touches anything scored on it:

* For each outer fold, every block is refit on the training folds (giving outer scores)
  and cross-validated within them (giving inner scores).
* Block probabilities become logits, standardised with inner-score statistics only.
* Two fusions are fixed in advance: the mean of standardised logits, and a class-balanced
  logistic stacker fit on the inner scores.
* Decision thresholds are learned on inner scores by maximising balanced accuracy, so
  off-centre blocks are not penalised by a fixed cut.

Blocks:

    text        char tf-idf logistic regression (the floor's configuration)
    stagea      mean probability of per-layer probes over every layer x pooling (E1)
    tracestate  the same layer ensemble over hidden states during the trace (E4)
    trace       logistic regression on 14 uncertainty + length + trace-shape features (E2)

The default block set, `text stagea trace`, is E10 v1 as declared in PLAN.md. Add
`tracestate` once E4 has run. Rows cover each block alone, every pair (as ablations), and
all selected blocks together (the headline, fixed in advance). Differences from text alone
come with 95% paired bootstrap CIs for both AUC and balanced accuracy.

Run (after the E1 and E2 outputs are downloaded):

    uv run python ../train/train_fusion.py
    uv run python ../train/train_fusion.py --blocks text stagea                    # before E2 lands
    uv run python ../train/train_fusion.py --blocks text stagea tracestate trace   # after E4
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from results_io import METRICS_HEADER, metrics_row, paired_difference, score, upsert_section
from train_stagea_probes import probe, text_model
from train_stageb_features import DEFAULT_GLOB, FEATURE_NAMES, MODELS, trace_features

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
SPLITS = ROOT / "data" / "small_track" / "splits.json"
# e1b, not e1: the original run's prompts went through a whitespace-stripping tokenizer (Finding 14).
DEFAULT_STAGEA = ROOT / "runs" / "e1b-stagea-deepseek" / "features" / "stagea-deepseek"
DEFAULT_TRACESTATE = ROOT / "runs" / "e4-tracestate-deepseek" / "features" / "tracestate-deepseek"
DEFAULT_OUT = ROOT / "runs" / "e10-fusion"
BLOCKS = ("text", "stagea", "tracestate", "trace")
LAYER_BLOCKS = ("stagea", "tracestate")
DEFAULT_BLOCKS = ["text", "stagea", "trace"]
EPS = 1e-6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_instances(model_id: str) -> pd.DataFrame:
    table = pd.read_parquet(INSTANCES)
    table = table[table["model_id"] == model_id][["problem_key", "problem", "is_robust"]].copy()
    fold_map = json.loads(SPLITS.read_text(encoding="utf-8"))["problem_fold"]
    table["fold"] = table["problem_key"].map(fold_map)
    return table.dropna(subset=["fold"]).astype({"fold": int})


def load_layer_features(features_dir: Path, model_id: str) -> tuple[list[str], dict[str, np.ndarray]]:
    """Merged hidden-state features (Stage A or E4 layout): keys and X_<pooling> arrays."""
    if not (features_dir / "features.npz").is_file():
        raise SystemExit(f"no features.npz in {features_dir}; download and merge that run first")
    payload = np.load(features_dir / "features.npz")
    records = pd.read_parquet(features_dir / "records.parquet")
    if (records["model_id"] != model_id).any() or (records["extracted_model_id"] != model_id).any():
        raise SystemExit(f"{features_dir} holds features or labels for another model")
    keys = [str(k) for k in payload["problem_keys"]]
    X = {
        name.removeprefix("X_"): np.nan_to_num(payload[name].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        for name in payload.files
        if name.startswith("X_")
    }
    return keys, X


def load_trace(records_glob: str, model_id: str) -> pd.DataFrame:
    paths = sorted(glob.glob(records_glob))
    if not paths:
        raise SystemExit(f"no Stage B records match {records_glob}; download E2 or drop the trace block")
    rows = [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df = pd.DataFrame(rows).drop_duplicates("problem_key", keep="last").reset_index(drop=True)
    if (df["model_id"] != model_id).any() or (df["extracted_model_id"] != model_id).any():
        raise SystemExit("Stage B records hold features or labels for another model")
    traces = pd.DataFrame([trace_features(t) for t in df["generated_text"]])
    df = pd.concat([df, traces], axis=1)
    df["log_generation_num_tokens"] = np.log1p(df["generation_num_tokens"].astype(float))
    df["hit_token_cap"] = df["hit_token_cap"].astype(float)
    columns = [*FEATURE_NAMES, "log_generation_num_tokens", "hit_token_cap", *traces.columns]
    return df.set_index("problem_key")[columns]


# ---------------------------------------------------------------------------
# Block predictions on every outer and inner split
# ---------------------------------------------------------------------------


def nested_splits(folds: np.ndarray) -> list[tuple[tuple[str, int, int], np.ndarray, np.ndarray]]:
    """(key, train, test) for each outer fold and each inner fold within its training set."""
    splits = []
    for outer in np.unique(folds):
        train = np.flatnonzero(folds != outer)
        splits.append((("outer", int(outer), -1), train, np.flatnonzero(folds == outer)))
        for inner in np.unique(folds[train]):
            splits.append(
                (("inner", int(outer), int(inner)), train[folds[train] != inner], train[folds[train] == inner])
            )
    return splits


def text_block(texts: list[str], y: np.ndarray, splits) -> dict:
    out = {}
    for key, train, test in splits:
        model = text_model().fit([texts[i] for i in train], y[train])
        out[key] = model.predict_proba([texts[i] for i in test])[:, 1]
    return out


def layer_split_probas(X_layer: np.ndarray, y: np.ndarray, splits, c: float) -> list[np.ndarray]:
    return [probe(c).fit(X_layer[train], y[train]).predict_proba(X_layer[test])[:, 1] for _, train, test in splits]


def layer_ensemble_block(X: dict[str, np.ndarray], y: np.ndarray, splits, c: float, n_jobs: int) -> dict:
    """Mean of per-layer probe probabilities over every layer and pooling, per split.

    One task per (pooling, layer) covering all splits keeps joblib from re-serialising a
    4096-dim slice for every split.
    """
    n_layers = next(iter(X.values())).shape[1]
    configs = list(product(sorted(X), range(n_layers)))
    per_config = Parallel(n_jobs=n_jobs)(
        delayed(layer_split_probas)(X[pooling][:, layer, :], y, splits, c) for pooling, layer in configs
    )
    return {key: np.mean([probas[i] for probas in per_config], axis=0) for i, (key, _, _) in enumerate(splits)}


def matrix_block(build, X: np.ndarray, y: np.ndarray, splits) -> dict:
    return {key: build().fit(X[train], y[train]).predict_proba(X[test])[:, 1] for key, train, test in splits}


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def learn_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Cut maximising balanced accuracy; ties resolve to the middle of the best range."""
    values = np.unique(scores)
    if len(values) == 1:
        return float(values[0])
    candidates = (values[:-1] + values[1:]) / 2
    accuracies = np.array([balanced_accuracy_score(y, scores >= t) for t in candidates])
    best = np.flatnonzero(np.isclose(accuracies, accuracies.max()))
    return float(candidates[best[len(best) // 2]])


def fuse(preds: dict, members: list[str], y: np.ndarray, folds: np.ndarray, method: str) -> np.ndarray:
    """Out-of-fold decision scores for one fusion; a score >= 0 means robust."""
    scores = np.full(len(y), np.nan)
    for outer in np.unique(folds):
        train = np.flatnonzero(folds != outer)
        test = np.flatnonzero(folds == outer)
        row_of = {index: row for row, index in enumerate(train)}

        inner = np.full((len(train), len(members)), np.nan)
        for inner_fold in np.unique(folds[train]):
            rows = [row_of[i] for i in train[folds[train] == inner_fold]]
            for col, member in enumerate(members):
                inner[rows, col] = logit(preds[member][("inner", int(outer), int(inner_fold))])
        held_out = np.column_stack([logit(preds[m][("outer", int(outer), -1)]) for m in members])

        mean, std = inner.mean(axis=0), inner.std(axis=0) + EPS
        inner_z, held_out_z = (inner - mean) / std, (held_out - mean) / std
        if method == "mean":
            threshold = learn_threshold(y[train], inner_z.mean(axis=1))
            scores[test] = held_out_z.mean(axis=1) - threshold
        else:
            stacker = LogisticRegression(C=1.0, class_weight="balanced").fit(inner_z, y[train])
            scores[test] = stacker.decision_function(held_out_z)
    return scores


def balanced_accuracy_at_zero(y: np.ndarray, scores: np.ndarray) -> float:
    return balanced_accuracy_score(y, scores >= 0)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--blocks", nargs="+", choices=BLOCKS, default=DEFAULT_BLOCKS)
    parser.add_argument("--stagea-dir", type=Path, default=DEFAULT_STAGEA)
    parser.add_argument("--tracestate-dir", type=Path, default=DEFAULT_TRACESTATE)
    parser.add_argument("--records-glob", default=DEFAULT_GLOB)
    parser.add_argument("--c", type=float, default=0.01, help="L2 strength of the per-layer probes")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--results", type=Path, default=ROOT / "RESULTS.md")
    parser.add_argument("--section", default="E10: fusion v1", help="RESULTS.md section to upsert")
    args = parser.parse_args()
    blocks = [b for b in BLOCKS if b in args.blocks]

    table = load_instances(args.model_id)
    available = set(table["problem_key"])
    layer_features: dict[str, tuple[list[str], dict[str, np.ndarray]]] = {}
    for block, directory in (("stagea", args.stagea_dir), ("tracestate", args.tracestate_dir)):
        if block in blocks:
            layer_features[block] = load_layer_features(directory, args.model_id)
            available &= set(layer_features[block][0])
    if "trace" in blocks:
        trace = load_trace(args.records_glob, args.model_id)
        available &= set(trace.index)
    n_expected = len(table)
    table = table[table["problem_key"].isin(available)].sort_values("problem_key").reset_index(drop=True)
    y = table["is_robust"].astype(bool).to_numpy()
    folds = table["fold"].to_numpy()
    print(f"{args.model_id}: {len(table)} of {n_expected} instances have every block ({', '.join(blocks)}); {int(y.sum())} robust")

    splits = nested_splits(folds)
    preds: dict[str, dict] = {}
    for block in blocks:
        started = time.time()
        if block == "text":
            preds[block] = text_block(table["problem"].tolist(), y, splits)
        elif block in LAYER_BLOCKS:
            keys, X = layer_features[block]
            order = {key: i for i, key in enumerate(keys)}
            index = [order[k] for k in table["problem_key"]]
            preds[block] = layer_ensemble_block({p: v[index] for p, v in X.items()}, y, splits, args.c, args.n_jobs)
        else:
            features = trace.loc[table["problem_key"]].to_numpy(dtype=np.float64)
            preds[block] = matrix_block(MODELS["logreg"], features, y, splits)
        print(f"  block {block}: {len(splits)} fits in {time.time() - started:.0f}s")

    rows: list[tuple[str, np.ndarray]] = [(f"{b} alone", fuse(preds, [b], y, folds, "mean")) for b in blocks]
    if len(blocks) > 2:
        for pair in combinations(blocks, 2):
            rows.append((f"mean fusion: {' + '.join(pair)}", fuse(preds, list(pair), y, folds, "mean")))
    if len(blocks) >= 2:
        rows.append(("**mean fusion: all blocks**", fuse(preds, blocks, y, folds, "mean")))
        rows.append(("**stacked fusion: all blocks**", fuse(preds, blocks, y, folds, "stack")))

    reference_name = "text alone" if "text" in blocks else rows[0][0]
    reference = dict(rows)[reference_name]
    print()
    comparisons = []
    for name, scores in rows:
        metrics = score(y, scores, threshold=0.0)
        print(f"{name:45s} bal_acc={metrics['balanced_accuracy']:.4f} auc={metrics['roc_auc']:.4f}")
        if name != reference_name:
            comparisons.append((
                name,
                paired_difference(y, scores, reference, roc_auc_score),
                paired_difference(y, scores, reference, balanced_accuracy_at_zero),
            ))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "fusion_oof.npz",
        problem_keys=table["problem_key"].to_numpy(dtype=str),
        y=y,
        folds=folds,
        **{name.strip("*").replace(": ", "__").replace(" + ", "_").replace(" ", "_"): s for name, s in rows},
    )

    def ci(diff: tuple[float, float, float]) -> str:
        return f"{diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]"

    headline = next((c for c in comparisons if c[0] == "**mean fusion: all blocks**"), None)
    lines = [
        f"Model: `{args.model_id}`  ",
        f"Instances: {len(table)} of {n_expected} with every block ({int(y.sum())} robust)  ",
        f"Blocks: {', '.join(blocks)}  ",
        "Protocol: nested grouped CV; block scores standardised on inner folds; thresholds learned "
        "on inner folds (a score >= 0 is robust); fusions fixed in advance",
        "",
        *METRICS_HEADER,
        metrics_row("always-spurious (majority)", score(y, np.zeros(len(y)))),
        *[metrics_row(name, score(y, scores, threshold=0.0)) for name, scores in rows],
        "",
        f"### Difference from {reference_name} (95% paired bootstrap CI)",
        "",
        "| Row | AUC difference | Balanced-accuracy difference |",
        "| --- | --- | --- |",
        *[f"| {name} | {ci(auc)} | {ci(bal)} |" for name, auc, bal in comparisons],
        "",
    ]
    if headline:
        _, auc, bal = headline

        def status(diff: tuple[float, float, float]) -> str:
            if diff[1] > 0:
                return "established gain"
            return "established loss" if diff[2] < 0 else "not established"

        # Report each metric on its own; one metric clearing zero says nothing about the other.
        lines.append(
            f"**Verdict:** relative to {reference_name}, fusing all blocks changes AUC by {ci(auc)} "
            f"({status(auc)}) and balanced accuracy by {ci(bal)} ({status(bal)}) at n={len(table)}."
        )
        lines.append("")
    lines.append(
        "Text alone here uses a learned threshold, so it can differ from the 0.5-threshold floor above. "
        "Pair rows are ablations; the all-block rows are the pre-declared result. "
        "Regenerate with `train/train_fusion.py`."
    )
    upsert_section(args.results, args.section, "\n".join(lines))


if __name__ == "__main__":
    main()
