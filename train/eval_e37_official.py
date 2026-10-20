"""Score E37's frozen budget-path model on the official AIMO training labels once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "train"))

from eval_e36_budget_path import build_examples, feature_columns, make_model, read_shards  # noqa: E402
from inspect_train_main_v2 import cluster_templates  # noqa: E402


SMALL_MODELS = {
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}
CONSTANT_TRUE = {"Qwen/Qwen3.5-4B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
SEED = 37


def read_feature_records(paths: list[Path]) -> pd.DataFrame:
    records = []
    for root in paths:
        for path in sorted(root.rglob("features.jsonl")):
            records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not records:
        raise SystemExit("no features.jsonl records found")
    frame = pd.DataFrame(records)
    if frame.duplicated(["model_id", "problem_key"]).any():
        raise ValueError("duplicate official feature record")
    return frame


def half_rank(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    prediction = np.zeros(len(frame), dtype=bool)
    for _, part in frame.groupby(["model_id", "reasoning_effort"], sort=True):
        index = part.index.to_numpy()
        order = np.lexsort((part["problem_key"].to_numpy(str), scores[index]))
        prediction[index[order[len(order) // 2 :]]] = True
    return prediction


def clustered_auc(y: np.ndarray, scores: np.ndarray, templates: np.ndarray, replicates: int) -> tuple[float, float]:
    unique = np.unique(templates)
    members = {template: np.flatnonzero(templates == template) for template in unique}
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(replicates):
        index = np.concatenate([members[value] for value in rng.choice(unique, len(unique), replace=True)])
        if np.unique(y[index]).size == 2:
            values.append(roc_auc_score(y[index], scores[index]))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def runtime_seconds(paths: list[Path]) -> float:
    total = 0.0
    found = 0
    for root in paths:
        for path in sorted(root.rglob("metadata.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("stage") == "E37 official budget-path extraction" and "seconds" in data:
                total += float(data["seconds"])
                found += 1
    if found != 4:
        raise ValueError(f"expected four completed model metadata files, found {found}")
    return total


def evaluate(
    math_frame: pd.DataFrame,
    official_rows: pd.DataFrame,
    feature_frame: pd.DataFrame,
    runtime: float,
    bootstrap: int = 5_000,
) -> dict:
    train = build_examples(math_frame, "math_perturb")
    small = official_rows.loc[
        official_rows["model_id"].isin(SMALL_MODELS) & official_rows["is_labelled"].astype(bool)
    ].copy()
    expected = set(zip(small["model_id"], small["problem_key"]))
    observed = set(zip(feature_frame["model_id"], feature_frame["problem_key"]))
    if observed != expected:
        raise ValueError(f"official feature coverage mismatch: missing={len(expected-observed)} extra={len(observed-expected)}")
    labels = small[["model_id", "reasoning_effort", "problem_key", "problem", "is_robust"]]
    frame = feature_frame.drop(columns=["is_robust", "budget_summaries"], errors="ignore").merge(
        labels, on=["model_id", "reasoning_effort", "problem_key"], validate="one_to_one"
    )
    columns = feature_columns(train)
    model = make_model().fit(train[columns], train["is_robust"].to_numpy(bool))
    scores = model.predict_proba(frame[columns])[:, 1]
    y = frame["is_robust"].astype(bool).to_numpy()
    decisions = half_rank(frame, scores)
    constants = frame["model_id"].isin(CONSTANT_TRUE).to_numpy(bool)

    score_table = pd.DataFrame({"problem_key": frame["problem_key"], "score": scores})
    problem_scores = score_table.groupby("problem_key", as_index=False)["score"].mean()
    labelled = official_rows.loc[official_rows["is_labelled"].astype(bool)].copy()
    grouped = labelled.groupby("problem_key")
    problems = pd.DataFrame({
        "problem": grouped["problem"].first(),
        "n": grouped.size(),
        "robust": grouped["is_robust"].sum(),
    }).reset_index()
    share = problems["robust"] / problems["n"]
    problems["label"] = np.where(share > 0.5, 1.0, np.where(share < 0.5, 0.0, np.nan))
    problems = problems.dropna(subset=["label"]).merge(problem_scores, on="problem_key", validate="one_to_one")
    normalized = sorted(problems["problem"].map(lambda text: " ".join(str(text).split())).unique())
    mapping = cluster_templates(normalized)
    problems["template"] = problems["problem"].map(lambda text: mapping[" ".join(str(text).split())])
    problem_y = problems["label"].astype(bool).to_numpy()
    problem_score = problems["score"].to_numpy(float)
    problem_auc = float(roc_auc_score(problem_y, problem_score))
    ci = clustered_auc(problem_y, problem_score, problems["template"].to_numpy(), bootstrap)
    row_ba = float(balanced_accuracy_score(y, decisions))
    constant_ba = float(balanced_accuracy_score(y, constants))

    by_model = {}
    for model_name, part in frame.assign(score=scores).groupby("model_id", sort=True):
        local_y = part["is_robust"].astype(bool).to_numpy()
        by_model[str(model_name)] = None if np.unique(local_y).size < 2 else float(
            roc_auc_score(local_y, part["score"])
        )
    gates = {
        "problem_auc_at_least_0_70": problem_auc >= 0.70,
        "template_ci_lower_above_0_50": ci[0] > 0.50,
        "half_rank_ba_above_e27_constants": row_ba > constant_ba,
        "runtime_below_3000_seconds": runtime < 3_000,
    }
    return {
        "rows": int(len(frame)),
        "problems": int(len(problems)),
        "templates": int(problems["template"].nunique()),
        "problem_auc": problem_auc,
        "problem_auc_ci95_template_bootstrap": list(ci),
        "row_auc": float(roc_auc_score(y, scores)),
        "row_half_rank_balanced_accuracy": row_ba,
        "e27_constant_balanced_accuracy": constant_ba,
        "model_aucs": by_model,
        "summed_t4_runtime_seconds": runtime,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-dir", type=Path, required=True)
    parser.add_argument("--official-rows", type=Path, default=ROOT / "data" / "train_main" / "rows.parquet")
    parser.add_argument("--feature-run", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(
        read_shards(args.math_dir),
        pd.read_parquet(args.official_rows),
        read_feature_records(args.feature_run),
        runtime_seconds(args.feature_run),
        args.bootstrap,
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
