"""Evaluate rank-based E27 successors before spending leaderboard submissions.

E27e establishes one useful fact on the hidden validation set: within each model batch,
lower mean problem NLL should be called robust.  This script asks whether a different
one-forward-pass familiarity statistic, or a small rank ensemble, is more repeatable.

No estimator is fitted here.  Every candidate predicts exactly the familiar half of a
batch as robust, matching the hidden-set class-prior assumption made by E27d/e.  Signals
are evaluated on:

* DeepSeek's 142 labelled Hendrycks MATH problems;
* the official olympiad rows, with a separate batch for each scored model; and
* the E22 problem-level table, averaging ranks across all four scoring models.

The first two views measure the policy Codabench actually calls.  The third prevents a
candidate from looking good only because one model has a different score scale.

Run:  python train/eval_e27_candidates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from eval_familiarity import DEEPSEEK, INSTANCES, SIGNALS, load_scores  # noqa: E402
from eval_problem_level import load_problems, load_rows, score  # noqa: E402
from results_io import upsert_section  # noqa: E402


BASE_SIGNALS = tuple(signal for signal in SIGNALS if signal != "relative_nll")
RESULTS = ROOT / "RESULTS.md"


def percentile(values: pd.Series) -> pd.Series:
    """Stable [0, 1] percentile rank; lower means more familiar."""
    return values.rank(method="average", pct=True)


def add_candidates(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Add scale-free signal ranks and conservative pre-specified ensembles."""
    out = frame.copy()
    for signal in BASE_SIGNALS:
        out[f"rank_{signal}"] = out.groupby(group, sort=False)[signal].transform(percentile)
    token_count = out["n_tokens"] - 1
    prefix_count = token_count // 2
    suffix_count = token_count - prefix_count
    out["prefix_nll"] = np.where(
        prefix_count > 0,
        (out["mean_nll"] * token_count - out["suffix_nll"] * suffix_count) / prefix_count,
        out["mean_nll"],
    )
    out["suffix_advantage"] = out["suffix_nll"] - out["prefix_nll"]
    out["tail_excess"] = out["min_k_20"] - out["mean_nll"]
    out["min_k_gap"] = out["min_k_10"] - out["min_k_20"]
    for signal in ("suffix_advantage", "tail_excess", "min_k_gap"):
        out[f"rank_{signal}"] = out.groupby(group, sort=False)[signal].transform(percentile)
    # Mean NLL won the first independent leaderboard test.  Ensembles retain it and add
    # only the two complementary scores with the strongest MATH within-length evidence.
    out["rank_mean_suffix"] = (out["rank_mean_nll"] + out["rank_suffix_nll"]) / 2
    out["rank_mean_mink10"] = (out["rank_mean_nll"] + out["rank_min_k_10"]) / 2
    out["rank_mean_suffix_mink10"] = (
        out["rank_mean_nll"] + out["rank_suffix_nll"] + out["rank_min_k_10"]
    ) / 3
    return out


def familiar_half(values: pd.Series) -> np.ndarray:
    """Call the lower-scoring ceil(n/2) rows robust with stable tie handling."""
    order = np.argsort(values.to_numpy(), kind="stable")
    prediction = np.zeros(len(values), dtype=bool)
    prediction[order[: (len(values) + 1) // 2]] = True
    return prediction


def batch_predictions(frame: pd.DataFrame, signal: str, group: str) -> np.ndarray:
    prediction = pd.Series(False, index=frame.index)
    for _, batch in frame.groupby(group, sort=False):
        prediction.loc[batch.index] = familiar_half(batch[signal])
    return prediction.to_numpy(dtype=bool)


def metrics(y: np.ndarray, score: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    return (
        roc_auc_score(y, -score),
        balanced_accuracy_score(y, prediction),
        accuracy_score(y, prediction),
    )


def main() -> None:
    scores = load_scores()
    candidates = [
        "rank_mean_nll",
        "rank_suffix_nll",
        "rank_min_k_10",
        "rank_min_k_20",
        "rank_min_k_pp_20",
        "rank_zlib_ratio",
        "rank_suffix_advantage",
        "rank_tail_excess",
        "rank_min_k_gap",
        "rank_mean_suffix",
        "rank_mean_mink10",
        "rank_mean_suffix_mink10",
    ]

    instances = pd.read_parquet(INSTANCES)
    math = instances[
        (instances["model_id"] == DEEPSEEK) & instances["dataset_id"].str.startswith("hendrycks")
    ].merge(scores[scores["scored_model_id"] == DEEPSEEK], on="problem_key", how="inner")
    math["batch"] = DEEPSEEK
    math = add_candidates(math, "batch")

    official = load_rows()
    # Multiple reasoning-effort rows for one (model, problem) carry the same label in the
    # released table.  Codabench batches only by model_id, so retain one row per pair.
    official = official.drop_duplicates(["model_id", "problem_key"])
    official = official.merge(
        scores,
        left_on=["problem_key", "model_id"],
        right_on=["problem_key", "scored_model_id"],
        how="inner",
    )
    official = add_candidates(official, "model_id")

    problems = load_problems()
    ranked = add_candidates(scores, "scored_model_id")
    problem_scores = ranked.groupby("problem_key")[candidates].mean()
    problems = problems.join(problem_scores, on="problem_key", how="inner")

    print("Candidate                         MATH AUC  MATH BA  Official AUC  Official BA  Official acc  Problem AUC")
    print("--------------------------------  --------  -------  ------------  -----------  ------------  -----------")
    y_math = math["is_robust"].astype(bool).to_numpy()
    y_official = official["is_robust"].astype(bool).to_numpy()
    y_problem = problems["label"].astype(bool).to_numpy()
    summaries = []
    for candidate in candidates:
        math_prediction = batch_predictions(math, candidate, "batch")
        official_prediction = batch_predictions(official, candidate, "model_id")
        m_auc, m_ba, _ = metrics(y_math, math[candidate].to_numpy(), math_prediction)
        o_auc, o_ba, o_acc = metrics(y_official, official[candidate].to_numpy(), official_prediction)
        p_auc = roc_auc_score(y_problem, -problems[candidate].to_numpy())
        summaries.append((candidate, m_auc, m_ba, o_auc, o_ba, o_acc, p_auc))
        print(
            f"{candidate:32}  {m_auc:8.3f}  {m_ba:7.3f}  {o_auc:12.3f}  {o_ba:11.3f}  "
            f"{o_acc:12.3f}  {p_auc:11.3f}"
        )

    print("\nOfficial accuracy by model (familiar half robust):")
    header = "Candidate" + "".join(f"  {model.split('/')[-1][:12]:>12}" for model in sorted(official["model_id"].unique()))
    print(header)
    for candidate in candidates:
        values = []
        prediction = batch_predictions(official, candidate, "model_id")
        official[f"prediction_{candidate}"] = prediction
        for model_id in sorted(official["model_id"].unique()):
            batch = official[official["model_id"] == model_id]
            values.append(accuracy_score(batch["is_robust"].astype(bool), batch[f"prediction_{candidate}"]))
        print(f"{candidate:32}" + "".join(f"  {value:12.3f}" for value in values))

    # The winning problem-level candidate is selected on these labels, so its ordinary
    # permutation p-value is optimistic.  Compare it with the maximum AUC reached by the
    # entire candidate family under each label permutation (Westfall-Young maxT).
    matrix = np.column_stack([-problems[candidate].to_numpy() for candidate in candidates])
    observed = np.array([roc_auc_score(y_problem, matrix[:, i]) for i in range(matrix.shape[1])])
    winner_index = int(np.argmax(observed))
    winner = candidates[winner_index]
    rng = np.random.default_rng(0)
    selected = score(y_problem, matrix[:, winner_index], problems["template"].to_numpy(), rng)
    n_permutations = 5000
    permuted_y = np.vstack([rng.permutation(y_problem) for _ in range(n_permutations)]).astype(float)
    # Mann-Whitney form of AUC. Pandas supplies average ranks for the rare tied score.
    ranks = np.column_stack([pd.Series(matrix[:, i]).rank(method="average").to_numpy() for i in range(matrix.shape[1])])
    n_positive = int(y_problem.sum())
    n_negative = len(y_problem) - n_positive
    null_auc = (permuted_y @ ranks - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    null_max = null_auc.max(axis=1)
    adjusted_p = (1 + np.sum(null_max >= observed[winner_index])) / (1 + n_permutations)
    print(
        f"\nSelected problem-level candidate: {winner}, AUC={selected['auc']:.3f}, "
        f"template bootstrap CI=[{selected['low']:.3f}, {selected['high']:.3f}], "
        f"ordinary p={selected['p']:.4f}, maxT-adjusted p={adjusted_p:.4f}."
    )
    lines = [
        "All candidates call the familiar half of each complete per-model batch robust. No estimator is fitted. "
        "MATH uses DeepSeek's 142 labelled Hendrycks problems; official-row metrics preserve the evaluator's "
        "per-model batches; problem AUC averages each rank over all four scoring models. Regenerated by "
        "`train/eval_e27_candidates.py`.",
        "",
        "| Candidate | MATH AUC | MATH balanced acc. | Official-row AUC | Official-row balanced acc. | Official-row accuracy | Problem AUC |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate, m_auc, m_ba, o_auc, o_ba, o_acc, p_auc in summaries:
        lines.append(
            f"| `{candidate.removeprefix('rank_')}` | {m_auc:.3f} | {m_ba:.3f} | {o_auc:.3f} | "
            f"{o_ba:.3f} | {o_acc:.3f} | {p_auc:.3f} |"
        )
    lines += [
        "",
        f"The selected problem-level candidate is **`{winner.removeprefix('rank_')}`**: AUC "
        f"**{selected['auc']:.3f}**, template-bootstrap 95% CI [{selected['low']:.3f}, "
        f"{selected['high']:.3f}], ordinary permutation p = {selected['p']:.4f}. Because it was selected "
        f"from {len(candidates)} candidates, the maxT family-wise adjusted p-value is **{adjusted_p:.4f}**.",
    ]
    upsert_section(RESULTS, "E28: familiarity rank candidates", "\n".join(lines))


if __name__ == "__main__":
    main()
