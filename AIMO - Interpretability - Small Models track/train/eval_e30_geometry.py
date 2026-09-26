"""E30 stage 1: screen architecture-invariant prompt-activation geometry.

The four small-track checkpoints have different hidden widths and layer counts, so their raw
states cannot share one linear probe.  Geometry can: norm growth, late-layer residual change,
and disagreement between last/mean/max token poolings are scalar and become scale-free after
ranking inside each model.

This stage uses only the already-extracted, officially labelled ``train-main-v2`` rows.  The
candidate family is fixed below before reading its labels: three statistics for each of the
three token poolings plus three final-layer cross-pooling cosine distances (12 tests total).
Direction is learned out of fold, leaving out an entire near-duplicate problem template.  A
maxT permutation test corrects selection over all 12 candidates.

Gate E30-G1: launch the all-model/all-problem extraction only if one candidate has LOTO AUC
>= 0.70, a template-bootstrap lower bound above 0.50, and maxT-adjusted p <= 0.05.  Otherwise
the geometry lane closes without using Kaggle or Codabench.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "train"))

from inspect_train_main_v2 import cluster_templates  # noqa: E402
from results_io import upsert_section  # noqa: E402

INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
FEATURE_GLOB = "e21-stagea-*-official/features/*/features.npz"
RESULTS = ROOT / "RESULTS.md"
POOLINGS = ("last", "mean", "max")
CANDIDATES = (
    "last_growth",
    "last_relchange_tail",
    "last_coschange_tail",
    "mean_growth",
    "mean_relchange_tail",
    "mean_coschange_tail",
    "max_growth",
    "max_relchange_tail",
    "max_coschange_tail",
    "final_last_mean_cosdist",
    "final_last_max_cosdist",
    "final_mean_max_cosdist",
)
GATE_AUC = 0.70
GATE_LOW = 0.50
GATE_P = 0.05


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return 1.0 - numerator / np.maximum(denominator, 1e-12)


def pooling_geometry(states: np.ndarray) -> dict[str, np.ndarray]:
    """Return the three pre-registered scalars for one token pooling."""
    states = states.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(states * states, axis=2))
    before, after = states[:, :-1], states[:, 1:]
    delta = after - before
    relative_change = np.linalg.norm(delta, axis=2) / np.maximum(np.linalg.norm(before, axis=2), 1e-12)
    cosine_change = 1.0 - np.einsum("nld,nld->nl", before, after) / np.maximum(
        np.linalg.norm(before, axis=2) * np.linalg.norm(after, axis=2), 1e-12
    )
    tail = max(0, relative_change.shape[1] * 3 // 4)
    return {
        "growth": np.log(np.maximum(rms[:, -1], 1e-12) / np.maximum(rms[:, 0], 1e-12)),
        "relchange_tail": relative_change[:, tail:].mean(axis=1),
        "coschange_tail": cosine_change[:, tail:].mean(axis=1),
    }


def load_labelled_rows() -> pd.DataFrame:
    rows = pd.read_parquet(INSTANCES)
    rows = rows[rows["dataset_id"].eq("train-main-v2")].copy()
    texts = sorted(rows["problem"].map(lambda value: " ".join(value.split())).unique())
    templates = cluster_templates(texts)
    rows["template"] = rows["problem"].map(lambda value: templates[" ".join(value.split())])
    return rows


def load_geometry() -> tuple[pd.DataFrame, pd.DataFrame]:
    labelled = load_labelled_rows()
    parts = []
    math_parts = []
    for path in sorted((ROOT / "runs").glob(FEATURE_GLOB)):
        print(f"loading {path.relative_to(ROOT)}", flush=True)
        records = pd.read_parquet(path.with_name("records.parquet"))
        keep = records["dataset_id"].eq("train-main-v2").to_numpy()
        if not keep.any():
            continue
        part = records.loc[keep, ["problem_key", "model_id", "is_robust", "n_prompt_tokens"]].reset_index(drop=True)
        finals: dict[str, np.ndarray] = {}
        with np.load(path) as archive:
            for pooling in POOLINGS:
                all_states = archive[f"X_{pooling}"]
                states = all_states[keep]
                geometry = pooling_geometry(states)
                for name, values in geometry.items():
                    part[f"{pooling}_{name}"] = values
                finals[pooling] = states[:, -1].astype(np.float32, copy=True)
                if pooling == "mean":
                    math_keep = records["dataset_id"].str.startswith("hendrycks").to_numpy()
                    if math_keep.any():
                        math_states = all_states[math_keep].astype(np.float32, copy=False)
                        rms = np.sqrt(np.mean(math_states * math_states, axis=2))
                        math_parts.append(
                            pd.DataFrame(
                                {
                                    "is_robust": records.loc[math_keep, "is_robust"].to_numpy(bool),
                                    "n_prompt_tokens": records.loc[math_keep, "n_prompt_tokens"].to_numpy(int),
                                    "mean_growth": np.log(
                                        np.maximum(rms[:, -1], 1e-12) / np.maximum(rms[:, 0], 1e-12)
                                    ),
                                }
                            )
                        )
                        del math_states, rms
                del states, all_states
        for left, right in (("last", "mean"), ("last", "max"), ("mean", "max")):
            part[f"final_{left}_{right}_cosdist"] = cosine_distance(finals[left], finals[right])
        parts.append(part)
        print(f"  retained {len(part)} official rows", flush=True)
    if not parts:
        raise SystemExit(f"no E21 feature archives match runs/{FEATURE_GLOB}")
    frame = pd.concat(parts, ignore_index=True)
    expected = labelled[["problem_key", "model_id", "template"]]
    frame = frame.merge(expected, on=["problem_key", "model_id"], how="inner", validate="one_to_one")
    if len(frame) != len(labelled):
        raise SystemExit(f"expected {len(labelled)} labelled feature rows, found {len(frame)}")
    for candidate in CANDIDATES:
        frame[candidate] = frame.groupby("model_id", sort=False)[candidate].rank(method="average", pct=True)
    math = pd.concat(math_parts, ignore_index=True) if math_parts else pd.DataFrame()
    return frame, math


def problem_table(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("problem_key", sort=False)
    table = pd.DataFrame(
        {
            "label": grouped["is_robust"].mean(),
            "template": grouped["template"].first(),
            "n_models": grouped.size(),
            "n_prompt_tokens": grouped["n_prompt_tokens"].mean(),
            **{candidate: grouped[candidate].mean() for candidate in CANDIDATES},
        }
    ).reset_index()
    table = table[table["label"].ne(0.5)].copy()
    table["label"] = table["label"].gt(0.5)
    return table


def binary_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Small-array Mann-Whitney AUC without sklearn's per-call validation overhead."""
    positive, negative = scores[y], scores[~y]
    if not len(positive) or not len(negative):
        return 0.5
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def loto_orient(y: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int]:
    """Learn only a signal's direction, leaving the held template out."""
    prediction = np.full(len(y), np.nan)
    positive = 0
    for group in np.unique(groups):
        held = groups == group
        train_y, train_score = y[~held], scores[~held]
        use_positive = binary_auc(train_y, train_score) >= 0.5
        prediction[held] = scores[held] if use_positive else 1.0 - scores[held]
        positive += int(use_positive)
    return prediction, positive


def cluster_interval(
    y: np.ndarray, scores: np.ndarray, groups: np.ndarray, rng: np.random.Generator, n_boot: int,
) -> tuple[float, float]:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    values = []
    for _ in range(n_boot):
        index = np.concatenate([members[group] for group in rng.choice(unique, size=len(unique))])
        if y[index].any() and not y[index].all():
            values.append(roc_auc_score(y[index], scores[index]))
    return tuple(np.percentile(values, [2.5, 97.5]))


def evaluate(rows: pd.DataFrame, seed: int = 0, n_boot: int = 2000, n_perm: int = 2000) -> pd.DataFrame:
    table = problem_table(rows)
    y = table["label"].to_numpy(bool)
    groups = table["template"].to_numpy()
    rng = np.random.default_rng(seed)
    result_rows = []
    observed = []
    for candidate in CANDIDATES:
        prediction, positive = loto_orient(y, table[candidate].to_numpy(float), groups)
        auc = binary_auc(y, prediction)
        low, high = cluster_interval(y, prediction, groups, rng, n_boot)
        result_rows.append(
            {
                "candidate": candidate,
                "auc": auc,
                "low": low,
                "high": high,
                "balanced_accuracy": balanced_accuracy_score(y, prediction >= 0.5),
                "positive_folds": positive,
                "n_folds": len(np.unique(groups)),
            }
        )
        observed.append(auc)

    null_max = np.empty(n_perm)
    signal_matrix = table[list(CANDIDATES)].to_numpy(float)
    for index in range(n_perm):
        permuted = rng.permutation(y)
        null_max[index] = max(
            binary_auc(permuted, loto_orient(permuted, signal_matrix[:, column], groups)[0])
            for column in range(len(CANDIDATES))
        )
        if (index + 1) % 250 == 0:
            print(f"  maxT permutations {index + 1}/{n_perm}", flush=True)
    for row, auc in zip(result_rows, observed, strict=True):
        row["maxT_p"] = (1 + int(np.sum(null_max >= auc))) / (1 + n_perm)
        row["gate"] = bool(auc >= GATE_AUC and row["low"] > GATE_LOW and row["maxT_p"] <= GATE_P)
    return pd.DataFrame(result_rows).sort_values(["auc", "low"], ascending=False).reset_index(drop=True)


def within_length_auc(y: np.ndarray, scores: np.ndarray, length: np.ndarray, n_bins: int = 5) -> float:
    bins = np.minimum((rankdata(length, method="ordinal") - 1) * n_bins // len(length), n_bins - 1)
    wins = pairs = 0.0
    for bin_index in range(n_bins):
        positive = scores[(bins == bin_index) & y]
        negative = scores[(bins == bin_index) & ~y]
        wins += (positive[:, None] > negative[None, :]).sum()
        wins += 0.5 * (positive[:, None] == negative[None, :]).sum()
        pairs += len(positive) * len(negative)
    return float(wins / pairs)


def confirm_on_math(math: pd.DataFrame, seed: int = 0, n_boot: int = 5000, n_perm: int = 5000) -> dict:
    """Confirm the selected official-data signal on untouched Hendrycks MATH rows."""
    y = math["is_robust"].to_numpy(bool)
    scores = math["mean_growth"].to_numpy(float)
    length = math["n_prompt_tokens"].to_numpy(int)
    rng = np.random.default_rng(seed)
    auc = binary_auc(y, scores)
    boot = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        if y[index].any() and not y[index].all():
            boot.append(binary_auc(y[index], scores[index]))
    null = np.array([binary_auc(rng.permutation(y), scores) for _ in range(n_perm)])
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(y),
        "robust": int(y.sum()),
        "auc": auc,
        "low": float(low),
        "high": float(high),
        "within_length": within_length_auc(y, scores, length),
        "p": float((1 + np.sum(null >= auc)) / (1 + n_perm)),
        "rho_length": float(pd.Series(scores).corr(pd.Series(length), method="spearman")),
    }


def render(rows: pd.DataFrame, results: pd.DataFrame, math: dict) -> str:
    problems = problem_table(rows)
    lines = [
        f"{len(rows)} official small-track rows over {len(problems)} problems and "
        f"{problems['template'].nunique()} near-duplicate templates. Signals are ranked inside each model; "
        "direction is learned leave-one-template-out. The maxT p-value corrects selection over all 12 "
        "pre-registered geometry signals.",
        "",
        "| Signal | LOTO AUC | Template 95% CI | Balanced acc. | Positive-direction folds | maxT p | Gate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| `{row.candidate}` | {row.auc:.3f} | [{row.low:.3f}, {row.high:.3f}] | "
            f"{row.balanced_accuracy:.3f} | {row.positive_folds}/{row.n_folds} | {row.maxT_p:.4f} | "
            f"{'**pass**' if row.gate else 'fail'} |"
        )
    winner = results.iloc[0]
    lines += [
        "",
        "### Independent confirmation of the selected signal",
        "",
        "`mean_growth` was selected only after the 12-way official-data screen, so it is checked once on "
        "the untouched DeepSeek Hendrycks MATH rows in the same archive. The official-data direction is "
        "higher growth -> robust.",
        "",
        "| Set | n | Robust | AUC | 95% CI | Within-length AUC | p | Spearman with length |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| Hendrycks MATH | {math['n']} | {math['robust']} | {math['auc']:.3f} | "
        f"[{math['low']:.3f}, {math['high']:.3f}] | {math['within_length']:.3f} | "
        f"{math['p']:.4f} | {math['rho_length']:+.3f} |",
        "",
    ]
    discovery_passes = bool(winner["gate"])
    confirmation_passes = bool(math["auc"] >= 0.65 and math["low"] > 0.5 and math["within_length"] >= 0.6)
    decision = (
        "E30 passes both gates; extract every official problem under every scoring model before considering a package."
        if discovery_passes and confirmation_passes
        else "E30 closes: the 12-way discovery is not family-wise significant and its selected signal reverses on the independent MATH set. Do not spend GPU or leaderboard submissions on activation geometry."
    )
    lines += [f"**Decision:** {decision}"]
    return "\n".join(lines)


def main() -> None:
    rows, math_rows = load_geometry()
    results = evaluate(rows)
    math = confirm_on_math(math_rows)
    body = render(rows, results, math)
    print(body)
    upsert_section(RESULTS, "E30: architecture-invariant activation geometry", body)


if __name__ == "__main__":
    main()
