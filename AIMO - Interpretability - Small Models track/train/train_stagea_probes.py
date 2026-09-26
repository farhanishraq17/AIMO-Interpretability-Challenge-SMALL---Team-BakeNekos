"""Grouped-CV linear probes on hidden-state features (every layer x pooling).

Written for Stage A (prompt states, E1), and also used for E4 trace states, which share the
layout; pass `--section` so each run keeps its own RESULTS.md section.

Answers the question Finding 2 left open. The shipped probe reads the last prompt token at
one layer and averages 0.5005 balanced accuracy per layer; its 58.4% headline comes from
picking the best fold after the fact. These features keep every layer and several
poolings, so here they are used honestly and compared against the char tf-idf text model,
recomputed on the same instances and folds.

Every row except the last is leak-free:

1. **Layer ensembles** average every layer's out-of-fold probabilities, so no label-driven
   selection happens anywhere.
2. **Nested estimate:** inside each outer fold, (pooling, layer, C) is selected by grouped
   CV on the four training folds alone, then scored once on the held-out fold.
3. **Fusion with text**, whose partner (the all-pooling ensemble) is fixed in advance.
4. **Next-token uncertainty scalars** from the same forward pass, when the records carry them.
5. **Per-layer curve**, for inspection only. Its maximum is optimistic by construction; it is
   shown to make the selection effect visible.

Balanced accuracy uses a fixed 0.5 threshold, which penalises models whose probabilities are
not centred there, so the verdict also compares ROC-AUC against text with a paired bootstrap
CI. Out-of-fold probabilities are saved to `stagea_oof.npz` for later fusion.

Run (after `datagen/kaggle_launch.py output e1-stagea-deepseek`):

    uv run python ../train/train_stagea_probes.py \
        --features-dir ../runs/e1-stagea-deepseek/features/stagea-deepseek

E4:

    uv run python ../train/train_stagea_probes.py \
        --features-dir ../runs/e4-tracestate-deepseek/features/tracestate-deepseek \
        --section "E4: trace-state probes"
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from results_io import METRICS_HEADER, metrics_row, paired_auc_difference, score, upsert_section

ROOT = Path(__file__).resolve().parents[1]
# e1b, not e1: the original run's prompts went through a whitespace-stripping tokenizer (Finding 14).
DEFAULT_FEATURES = ROOT / "runs" / "e1b-stagea-deepseek" / "features" / "stagea-deepseek"
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"

# Measured elsewhere: PLAN.md Finding 2.
SHIPPED_PROBE_MEAN_LAYER_BALANCED_ACCURACY = 0.5005

UNCERTAINTY_COLUMNS = [
    "next_token_entropy",
    "next_token_max_prob",
    "next_token_top5_prob_mass",
    "next_token_logit_margin",
    "next_token_perplexity",
]


def probe(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
    )


def text_model():
    """Same configuration as the char tf-idf floor in `train_text_baseline.py`."""
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
    )


def oof_proba(X: np.ndarray, y: np.ndarray, folds: np.ndarray, c: float) -> np.ndarray:
    """Out-of-fold P(robust), leaving each fold out in turn."""
    proba = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        val = folds == fold
        proba[val] = probe(c).fit(X[~val], y[~val]).predict_proba(X[val])[:, 1]
    return proba


def text_oof_proba(records: pd.DataFrame, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    problems = pd.read_parquet(INSTANCES).set_index(["model_id", "problem_key"])["problem"]
    texts = [problems[(m, k)] for m, k in zip(records["model_id"], records["problem_key"])]
    proba = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        val = folds == fold
        model = text_model().fit([t for t, v in zip(texts, val) if not v], y[~val])
        proba[val] = model.predict_proba([t for t, v in zip(texts, val) if v])[:, 1]
    return proba


def load(features_dir: Path, allow_label_mismatch: bool):
    payload = np.load(features_dir / "features.npz")
    records = pd.read_parquet(features_dir / "records.parquet")
    keys = [str(k) for k in payload["problem_keys"]]
    if keys != records["problem_key"].astype(str).tolist():
        raise SystemExit("features.npz and records.parquet disagree on instance order")

    mismatched = records["model_id"] != records["extracted_model_id"]
    if mismatched.any() and not allow_label_mismatch:
        raise SystemExit(
            f"{int(mismatched.sum())} rows carry another model's label (--no-filter-model run?); "
            "pass --allow-label-mismatch only for plumbing checks"
        )

    keep = (records["fold"] >= 0).to_numpy()
    if not keep.all():
        print(f"warning: dropping {int((~keep).sum())} instances with no fold assignment")

    poolings = sorted(name.removeprefix("X_") for name in payload.files if name.startswith("X_"))
    X = {}
    for pooling in poolings:
        array = payload[f"X_{pooling}"][keep].astype(np.float32)
        bad = ~np.isfinite(array)
        if bad.any():
            print(f"warning: X_{pooling} has {int(bad.sum())} non-finite values (fp16 overflow?); zeroed")
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        X[pooling] = array
    records = records[keep].reset_index(drop=True)
    return X, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--section", default="Stage A: hidden-state probes", help="RESULTS.md section to upsert")
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01], help="inverse L2 strengths searched in the inner CV")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--results", type=Path, default=ROOT / "RESULTS.md")
    parser.add_argument("--allow-label-mismatch", action="store_true")
    args = parser.parse_args()

    X, records = load(args.features_dir, args.allow_label_mismatch)
    y = records["is_robust"].astype(bool).to_numpy()
    folds = records["fold"].astype(int).to_numpy()
    poolings = sorted(X)
    n, n_layers, hidden = X[poolings[0]].shape
    model_id = records["extracted_model_id"].iloc[0]
    print(f"{model_id}: {n} instances ({int(y.sum())} robust), {n_layers} layers x {hidden} dims, poolings={poolings}")
    print(f"non-finite activations recorded at extraction: {int(records.get('n_nonfinite_activations', pd.Series([0])).sum())}")

    configs = list(product(poolings, range(n_layers), args.c_grid))
    parallel = Parallel(n_jobs=args.n_jobs)

    # Nested selection.
    nested = np.full(n, np.nan)
    chosen = []
    for outer in np.unique(folds):
        train = folds != outer
        inner = parallel(
            delayed(oof_proba)(X[p][:, layer, :][train], y[train], folds[train], c)
            for p, layer, c in configs
        )
        inner_auc = [roc_auc_score(y[train], proba) for proba in inner]
        best = int(np.argmax(inner_auc))
        pooling, layer, c = configs[best]
        model = probe(c).fit(X[pooling][:, layer, :][train], y[train])
        nested[~train] = model.predict_proba(X[pooling][:, layer, :][~train])[:, 1]
        chosen.append({"fold": int(outer), "pooling": pooling, "layer": layer, "C": c, "inner_auc": inner_auc[best]})
        print(f"  outer fold {outer}: chose {pooling} layer {layer} C={c} (inner AUC {inner_auc[best]:.3f})")
    nested_metrics = score(y, nested)

    # Next-token scalars from the same forward pass, when this feature set records them.
    has_scalars = all(column in records.columns for column in UNCERTAINTY_COLUMNS)
    scalar_proba = np.full(n, np.nan)
    scalar_metrics = None
    if has_scalars:
        # copy=True: pandas 3 copy-on-write hands back a read-only view otherwise.
        scalars = records[UNCERTAINTY_COLUMNS].to_numpy(dtype=np.float64, copy=True)
        scalars[:, -1] = np.log1p(np.clip(scalars[:, -1], 0, None))  # perplexity is heavy-tailed
        scalar_proba = oof_proba(scalars, y, folds, 1.0)
        scalar_metrics = score(y, scalar_proba)

    # Per-layer curve over all folds. Also the raw material for the layer ensembles.
    curve_configs = list(product(poolings, range(n_layers)))
    c_curve = args.c_grid[0]
    curve = parallel(delayed(oof_proba)(X[p][:, layer, :], y, folds, c_curve) for p, layer in curve_configs)
    curve_df = pd.DataFrame(
        [{"pooling": p, "layer": layer, **score(y, proba)} for (p, layer), proba in zip(curve_configs, curve)]
    )
    curve_path = args.features_dir / "layer_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    hindsight = curve_df.loc[curve_df["balanced_accuracy"].idxmax()]

    # Layer ensembles: averaging fixed sets of OOF probabilities involves no selection.
    by_config = dict(zip(curve_configs, curve))
    ensembles = {
        pooling: np.mean([by_config[(pooling, layer)] for layer in range(n_layers)], axis=0)
        for pooling in poolings
    }
    ensembles["all"] = np.mean(curve, axis=0)
    ensemble_metrics = {name: score(y, proba) for name, proba in ensembles.items()}

    # Does anything in the hidden states add to the problem text?
    text = text_oof_proba(records, y, folds)
    text_metrics = score(y, text)
    fused = (text + ensembles["all"]) / 2
    fused_metrics = score(y, fused)
    ensemble_vs_text = paired_auc_difference(y, ensembles["all"], text)
    fusion_vs_text = paired_auc_difference(y, fused, text)

    np.savez_compressed(
        args.features_dir / "stagea_oof.npz",
        problem_keys=records["problem_key"].astype(str).to_numpy(),
        y=y,
        folds=folds,
        nested=nested,
        text=text,
        scalars=scalar_proba,
        **{f"ensemble_{name}": proba for name, proba in ensembles.items()},
    )

    majority = score(y, np.zeros(n))
    print()
    rows = [
        ("text (char tf-idf, recomputed)", text_metrics),
        *([("next-token scalars", scalar_metrics)] if scalar_metrics else []),
        ("nested-selected probe", nested_metrics),
        *[(f"layer ensemble, {name} pooling", m) for name, m in ensemble_metrics.items()],
        ("fusion: text + all-pooling ensemble", fused_metrics),
    ]
    for name, m in rows:
        print(f"{name:40s} bal_acc={m['balanced_accuracy']:.4f} auc={m['roc_auc']:.4f}")
    print(f"{'hindsight best (optimistic)':40s} {hindsight['pooling']} layer {int(hindsight['layer'])} bal_acc={hindsight['balanced_accuracy']:.4f}")
    print(f"AUC(all-pooling ensemble) - AUC(text) = {ensemble_vs_text[0]:+.4f} [{ensemble_vs_text[1]:+.4f}, {ensemble_vs_text[2]:+.4f}]")
    print(f"AUC(fusion) - AUC(text)                = {fusion_vs_text[0]:+.4f} [{fusion_vs_text[1]:+.4f}, {fusion_vs_text[2]:+.4f}]")

    hidden_rows = {"nested-selected probe": nested_metrics}
    hidden_rows.update({f"layer ensemble ({name} pooling)": m for name, m in ensemble_metrics.items()})
    floor = text_metrics["balanced_accuracy"]
    above_by_accuracy = [name for name, m in hidden_rows.items() if m["balanced_accuracy"] > floor]
    above_by_auc = [name for name, m in hidden_rows.items() if m["roc_auc"] > text_metrics["roc_auc"]]

    def ci(diff: tuple[float, float, float]) -> str:
        return f"{diff[0]:+.4f} (95% paired bootstrap CI {diff[1]:+.4f} to {diff[2]:+.4f})"

    lines = [
        f"Model: `{model_id}`  ",
        f"Instances: {n} ({int(y.sum())} robust / {int((~y).sum())} spurious), "
        f"{n_layers} hidden-state layers x {hidden} dims, poolings {', '.join(poolings)}  ",
        # Repo-relative, so RESULTS.md does not carry one machine's absolute path.
        f"Features: `{(args.features_dir.resolve().relative_to(ROOT) if args.features_dir.resolve().is_relative_to(ROOT) else args.features_dir).as_posix()}`  ",
        f"Probe: standardised L2 logistic regression, class-balanced, C searched over {args.c_grid}",
        "",
        *METRICS_HEADER,
        metrics_row("always-spurious (majority)", majority),
        metrics_row("logreg on tf-idf chars (text floor, same instances and folds)", text_metrics),
        *([metrics_row("logreg on 5 next-token uncertainty scalars", scalar_metrics)] if scalar_metrics else []),
        metrics_row("**nested-selected hidden-state probe**", nested_metrics),
        *[metrics_row(f"layer ensemble, {name} pooling (no selection)", m) for name, m in ensemble_metrics.items()],
        metrics_row("fusion: text + all-pooling layer ensemble (partner fixed in advance)", fused_metrics),
        "",
        "**Verdict.** By balanced accuracy at the 0.5 threshold, "
        + (
            f"rows above the text floor: {', '.join(above_by_accuracy)}."
            if above_by_accuracy
            else f"no hidden-state row beats the text floor ({floor:.4f})."
        )
        + " By ROC-AUC, "
        + (
            f"rows ranking better than text: {', '.join(above_by_auc)}."
            if above_by_auc
            else "no hidden-state row ranks instances better than text."
        ),
        "",
        f"- All-pooling layer ensemble minus text, AUC: {ci(ensemble_vs_text)}.",
        f"- Fusion (text + that ensemble) minus text, AUC: {ci(fusion_vs_text)}.",
        "- An interval spanning zero means the difference is not established at this sample size.",
        f"- Hidden-state probabilities sit off-centre (all-pooling ensemble median "
        f"{np.median(ensembles['all']):.2f}; {ensemble_metrics['all']['n_predicted_robust']} of {n} "
        "predicted robust at 0.5), so any shipped version must learn its threshold in-fold.",
        "",
        "### Configurations chosen by the inner CV",
        "",
        "| Outer fold | Pooling | Layer | C | Inner AUC |",
        "| --- | --- | --- | --- | --- |",
        *[f"| {c['fold']} | {c['pooling']} | {c['layer']} | {c['C']} | {c['inner_auc']:.4f} |" for c in chosen],
        "",
        (
            "The inner CV picked the same configuration in every fold."
            if len({(c["pooling"], c["layer"], c["C"]) for c in chosen}) == 1
            else "Choices vary across folds, and inner AUCs well above the nested score mean the "
            "selection over layers and poolings is fitting noise."
        ),
        "",
        "### Per-layer curve (inspection only, optimistic)",
        "",
        "| Pooling | Mean bal. acc. over layers | Best layer | Best bal. acc. | Best AUC |",
        "| --- | --- | --- | --- | --- |",
    ]
    for pooling, group in curve_df.groupby("pooling"):
        top = group.loc[group["balanced_accuracy"].idxmax()]
        lines.append(
            f"| {pooling} | {group['balanced_accuracy'].mean():.4f} | {int(top['layer'])} "
            f"| {top['balanced_accuracy']:.4f} | {top['roc_auc']:.4f} |"
        )
    lines += [
        "",
        f"For scale, the shipped last-token probe averages {SHIPPED_PROBE_MEAN_LAYER_BALANCED_ACCURACY:.4f} "
        "per layer. The best-in-hindsight row is what picking a layer after seeing every fold "
        "gives, the same selection effect behind the baseline's 58.4%.",
        f"Full curve: `{curve_path.name}`; out-of-fold probabilities: `stagea_oof.npz`. "
        "Regenerate with `train/train_stagea_probes.py`.",
    ]
    upsert_section(args.results, args.section, "\n".join(lines))
    (args.features_dir / "stagea_probe_summary.json").write_text(
        json.dumps(
            {
                "nested": nested_metrics,
                "scalars": scalar_metrics,
                "text": text_metrics,
                "ensembles": ensemble_metrics,
                "fusion_text_all_layers": fused_metrics,
                "auc_difference_ensemble_all_minus_text": ensemble_vs_text,
                "auc_difference_fusion_minus_text": fusion_vs_text,
                "chosen": chosen,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
