"""Stress-test the pre-registered E28 familiarity statistics across scoring models.

E28d was selected from problem-level labels after averaging ranks from all four small-track
models.  That average is useful only if it is not carried by one scoring model.  This script
therefore evaluates the three candidates that were actually pre-registered for Codabench:

* mean NLL (the E27e baseline);
* suffix-half NLL (E28a); and
* the Min-K 10%-20% tail gap (E28d).

For every statistic it reports the problem-level AUC after leaving each scoring model out,
the worst leave-one-model-out AUC, pairwise rank agreement, and the evaluator-shaped accuracy
inside each model's labelled batch.  No new candidate is selected here.  The stability gate
for a successor is a worst leave-one-model-out AUC of at least 0.65.

Run with a Parquet reader compatible with the checked-in PyArrow-25 files:
    python train/eval_e28_stability.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from results_io import upsert_section  # noqa: E402

RECORDS = ROOT / "runs" / "e23-familiarity-*" / "features" / "familiarity-*" / "shard_*" / "records.jsonl"
LABELS = ROOT / "data" / "official_check" / "official_labels.parquet"
RESULTS = ROOT / "RESULTS.md"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen/Qwen3.5-4B"
DISPLAY = {
    QWEN: "Qwen",
    "Skywork/Skywork-OR1-Math-7B": "Skywork",
    "allenai/Olmo-3-7B-Think": "Olmo",
    DEEPSEEK: "DeepSeek",
}
CANDIDATES = ("mean_nll", "suffix_nll", "min_k_gap")
GATE = 0.65


def load_scores() -> pd.DataFrame:
    paths = sorted(glob.glob(str(RECORDS)))
    if not paths:
        raise SystemExit(f"no familiarity records match {RECORDS}")
    frame = pd.concat(
        [pd.read_json(path, lines=True, dtype={"problem_key": str}) for path in paths],
        ignore_index=True,
    )
    frame = frame.drop_duplicates(["problem_key", "scored_model_id"])
    frame["min_k_gap"] = frame["min_k_10"] - frame["min_k_20"]
    return frame


def familiar_half(values: pd.Series) -> np.ndarray:
    order = np.argsort(values.to_numpy(), kind="stable")
    prediction = np.zeros(len(values), dtype=bool)
    prediction[order[: (len(values) + 1) // 2]] = True
    return prediction


def pooled_labels(labels: pd.DataFrame) -> pd.DataFrame:
    pooled = labels.groupby("problem_key", as_index=False)["is_robust"].mean()
    pooled = pooled[pooled["is_robust"] != 0.5].copy()
    pooled["label"] = pooled["is_robust"] > 0.5
    return pooled[["problem_key", "label"]]


def main() -> None:
    scores = load_scores()
    labels = pd.read_parquet(LABELS).drop_duplicates(["model_id", "problem_key"])
    problems = pooled_labels(labels)
    official_keys = set(problems["problem_key"])
    ranked = scores[scores["problem_key"].isin(official_keys)].copy()
    for candidate in CANDIDATES:
        ranked[f"rank_{candidate}"] = ranked.groupby("scored_model_id", sort=False)[candidate].rank(
            method="average", pct=True
        )

    models = sorted(ranked["scored_model_id"].unique())
    problem_rows = []
    details: dict[str, dict[str, object]] = {}
    for candidate in CANDIDATES:
        pivot = ranked.pivot(index="problem_key", columns="scored_model_id", values=f"rank_{candidate}")
        joined = problems.set_index("problem_key").join(pivot, how="inner").dropna()
        y = joined["label"].astype(bool).to_numpy()
        all_auc = roc_auc_score(y, -joined[models].mean(axis=1))
        omitted = {
            DISPLAY[model]: roc_auc_score(y, -joined[[other for other in models if other != model]].mean(axis=1))
            for model in models
        }
        correlation = joined[models].corr(method="spearman").to_numpy()
        lower_triangle = correlation[np.tril_indices_from(correlation, k=-1)]
        worst_auc = min(omitted.values())
        min_correlation = float(np.nanmin(lower_triangle))
        details[candidate] = {
            "all_auc": all_auc,
            "omitted": omitted,
            "worst_auc": worst_auc,
            "min_correlation": min_correlation,
        }
        problem_rows.append((candidate, all_auc, omitted, worst_auc, min_correlation))

    labelled_rows = labels.merge(
        scores,
        left_on=["problem_key", "model_id"],
        right_on=["problem_key", "scored_model_id"],
        how="inner",
    )
    batch_rows: dict[str, dict[str, tuple[float, float, int]]] = {}
    for candidate in CANDIDATES:
        per_model = {}
        for model, batch in labelled_rows.groupby("model_id", sort=True):
            y = batch["is_robust"].astype(bool).to_numpy()
            values = batch[candidate]
            auc = roc_auc_score(y, -values) if len(np.unique(y)) == 2 else float("nan")
            accuracy = accuracy_score(y, familiar_half(values))
            per_model[DISPLAY[model]] = (auc, accuracy, len(batch))
        batch_rows[candidate] = per_model

    print("Problem-level leave-one-scoring-model-out stability")
    for candidate, all_auc, omitted, worst_auc, min_correlation in problem_rows:
        print(
            f"{candidate:12s} all={all_auc:.3f} "
            + " ".join(f"without_{model}={auc:.3f}" for model, auc in omitted.items())
            + f" worst={worst_auc:.3f} min_rho={min_correlation:.3f} gate={'pass' if worst_auc >= GATE else 'fail'}"
        )
    print("\nEvaluator-shaped labelled batches (AUC / familiar-half accuracy / n)")
    for candidate, per_model in batch_rows.items():
        print(candidate, {model: f"{auc:.3f}/{accuracy:.3f}/{n}" for model, (auc, accuracy, n) in per_model.items()})

    lines = [
        "The three rows are the pre-registered E27/E28 statistics, not a new candidate search. "
        "Problem labels are pooled across configurations, ranks are recomputed inside the official-problem "
        "set for each scoring model, and `without X` averages the other three ranks. The stability gate is "
        f"worst leave-one-model-out AUC >= {GATE:.2f}.",
        "",
        "| Statistic | All four AUC | Without Qwen | Without Skywork | Without Olmo | Without DeepSeek | Worst | Minimum pairwise rho | Gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate, all_auc, omitted, worst_auc, min_correlation in problem_rows:
        lines.append(
            f"| `{candidate}` | {all_auc:.3f} | {omitted['Qwen']:.3f} | {omitted['Skywork']:.3f} | "
            f"{omitted['Olmo']:.3f} | {omitted['DeepSeek']:.3f} | **{worst_auc:.3f}** | "
            f"{min_correlation:.3f} | {'**pass**' if worst_auc >= GATE else 'fail'} |"
        )
    lines += [
        "",
        "Evaluator-shaped batches (AUC / familiar-half accuracy / n):",
        "",
        "| Statistic | Qwen | Skywork | Olmo | DeepSeek |",
        "| --- | --- | --- | --- | --- |",
    ]
    for candidate, per_model in batch_rows.items():
        cells = []
        for model in ("Qwen", "Skywork", "Olmo", "DeepSeek"):
            auc, accuracy, n = per_model[model]
            cells.append(f"{auc:.3f} / {accuracy:.3f} / {n}")
        lines.append(f"| `{candidate}` | " + " | ".join(cells) + " |")
    upsert_section(RESULTS, "E28: cross-model stability", "\n".join(lines))


if __name__ == "__main__":
    main()
