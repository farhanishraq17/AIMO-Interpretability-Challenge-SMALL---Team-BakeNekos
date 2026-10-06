"""E49: score E37's frozen budget-path predictor once on all 34 official small-track rows.

`protocols/E49_PROTOCOL.md` fixes everything here before any E49 output exists. The predictor,
its 18 features, the four budgets and the within-model half-rank decision are E37's, unchanged.
What E49 changes is how the features are generated (`datagen/extract_budget_path_hf.py`: batched
`transformers`, the path a submission would take) and the gates. E37's were written for a run
that also had to prove its runtime on a T4 pair. E49's submission is anytime, so runtime is
recorded here rather than gated (protocol §4).

Gates, all on the same 34 rows:

1. coverage: every labelled small-track row, all four models, from clean E49 runs (no smoke
   divisor, no foreign labels);
2. problem-level AUC, as E37 defines it (20 problems carry small-track rows), of at least 0.65;
3. half-rank balanced accuracy above 0.55.

`--simulate-gates` prints how often gates 2 and 3 together pass a ranking of a given true row AUC,
on these exact rows, batches and label pooling: about 11% for a useless ranking and 69% for a true
AUC of 0.70. The thresholds were chosen from that table; at n = 34 no gate does much better.

Reported, not gated: the template-bootstrap CI (20 problems cannot put it above 0.5), and the
shipped familiarity ranking's accuracy on the same rows and batches (0.353). Any half split beats
0.353 here, so as a gate it would be vacuous.

E37's own metrics and gates are reported beside them, unchanged, for the record.

Run:  python train/eval_e49_official.py --simulate-gates       # the false-pass / power table, CPU
      python train/eval_e49_official.py --math-dir runs/e35-external/math_perturb \
          --feature-run runs/e49-qwen-<ALIAS> --feature-run runs/e49-skywork-<ALIAS> \
          --feature-run runs/e49-olmo-<ALIAS> --feature-run runs/e49-deepseek-<ALIAS> \
          --out kernels/runs/e49-official/result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "datagen"))

from eval_e36_budget_path import build_examples, feature_columns, make_model, read_shards  # noqa: E402
from eval_e37_official import SMALL_MODELS, evaluate, half_rank, read_feature_records  # noqa: E402
from eval_familiarity import load_scores  # noqa: E402

STAGE = "E49 official budget-path extraction (transformers, batched)"
MIN_PROBLEM_AUC = 0.65
MIN_BALANCED_ACCURACY = 0.55
CONFIRMATION = ROOT / "kernels" / "runs" / "e37-confirmation-Arifin" / "result.json"


def verify_archive(math_dir: Path) -> list[str]:
    """The training shards must be byte-identical to the four E37 was confirmed with.

    The pinned archive answered 401 to anonymous downloads on 7 Oct, so `prepare_e37_model.py`
    cannot fetch it. E35 kept the same four files; checking their hashes keeps the predictor frozen
    wherever the bytes come from.
    """
    provenance = json.loads(CONFIRMATION.read_text(encoding="utf-8"))["provenance"]["files"]
    expected = {e["sha256"] for e in provenance if e["path"].startswith("data/math_perturb/")}
    found = {hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(math_dir.glob("*.parquet"))}
    if found != expected or len(expected) != 4:
        raise ValueError(f"{math_dir} is not E37's MATH-Perturb archive: {len(found & expected)} of 4 shards match")
    return sorted(found)


def problem_labels(official_rows: pd.DataFrame) -> pd.Series:
    """E37's problem label: majority over every labelled configuration, ties dropped."""
    labelled = official_rows.loc[official_rows["is_labelled"].astype(bool)]
    grouped = labelled.groupby("problem_key")["is_robust"]
    share = grouped.sum() / grouped.size()
    return share[share != 0.5] > 0.5


def simulate_gates(official_rows: pd.DataFrame, draws: int = 4_000, seed: int = 49) -> list[str]:
    """How often gates 2 and 3 pass a binormal row ranking of each true AUC, on the real rows."""
    small = official_rows.loc[
        official_rows["model_id"].isin(SMALL_MODELS) & official_rows["is_labelled"].astype(bool)
    ].reset_index(drop=True)
    y = small["is_robust"].astype(bool).to_numpy()
    labels = problem_labels(official_rows)
    rng = np.random.default_rng(seed)
    lines = [
        f"{len(small)} rows ({y.mean():.3f} robust), {small['problem_key'].isin(labels.index).sum()} rows on "
        f"{small.loc[small['problem_key'].isin(labels.index), 'problem_key'].nunique()} labelled problems",
        "",
        "| true row AUC | P(problem AUC >= 0.65 and balanced acc. > 0.55) |",
        "| --- | --- |",
    ]
    for auc in (0.50, 0.60, 0.65, 0.70, 0.75):
        passes = 0
        for _ in range(draws):
            scores = rng.normal(size=len(y)) + np.sqrt(2) * norm.ppf(auc) * y
            by_problem = pd.Series(scores).groupby(small["problem_key"]).mean()
            by_problem = by_problem[by_problem.index.isin(labels.index)]
            problem_auc = roc_auc_score(labels[by_problem.index], by_problem)
            balanced = balanced_accuracy_score(y, half_rank(small, scores))
            passes += problem_auc >= MIN_PROBLEM_AUC and balanced > MIN_BALANCED_ACCURACY
        lines.append(f"| {auc:.2f} | {passes / draws:.3f} |")
    return lines


def read_runs(paths: list[Path]) -> dict[str, dict]:
    """Per-model runtime record from every E49 metadata.json; refuses smoke or foreign-label runs."""
    runs: dict[str, dict] = {}
    for root in paths:
        for path in sorted(root.rglob("metadata.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("stage") != STAGE:
                continue
            if data["smoke_divisor"] != 1 or not data["rows_filtered_to_model"]:
                raise ValueError(f"{path}: smoke or foreign-label run, not scoreable")
            entry = runs.setdefault(data["model_id"], {
                "rows": 0, "generated_tokens": 0, "generation_seconds": 0.0, "session_seconds": 0.0,
                "sessions": 0, "dtype": set(), "gpus": set(), "preflight_agreement_tokens": [],
            })
            entry["rows"] += data["rows_written"]
            for session in data["sessions"]:
                entry["sessions"] += 1
                entry["generated_tokens"] += session.get("generated_tokens", 0)
                entry["generation_seconds"] += session.get("generation_seconds", 0.0)
                entry["session_seconds"] += session.get("seconds", 0.0)
                entry["preflight_agreement_tokens"].append(session["preflight"]["padding_agreement_tokens"])
            entry["dtype"].add(data["dtype"])
            entry["gpus"].add(", ".join(data["gpus"]))
    if set(runs) != SMALL_MODELS:
        raise ValueError(f"E49 runs cover {sorted(runs)}; all four small-track models are required")
    for entry in runs.values():
        entry["dtype"], entry["gpus"] = sorted(entry["dtype"]), sorted(entry["gpus"])
        entry["tokens_per_row"] = entry["generated_tokens"] / max(entry["rows"], 1)
        entry["tokens_per_second"] = entry["generated_tokens"] / max(entry["generation_seconds"], 1e-9)
    return runs


def score(math_frame: pd.DataFrame, official_rows: pd.DataFrame, features: pd.DataFrame, runs: dict, bootstrap: int) -> dict:
    if "label_model_id" in features and (features["label_model_id"] != features["model_id"]).any():
        raise ValueError("feature records carry another model's labels")
    runtime = sum(entry["session_seconds"] for entry in runs.values())
    # E37's own evaluation, unchanged: it refuses incomplete or extra coverage, which is gate 1.
    e37 = evaluate(math_frame, official_rows, features, runtime, bootstrap)

    train = build_examples(math_frame, "math_perturb")
    columns = feature_columns(train)
    model = make_model().fit(train[columns], train["is_robust"].to_numpy(bool))
    labels = official_rows.loc[
        official_rows["model_id"].isin(SMALL_MODELS) & official_rows["is_labelled"].astype(bool),
        ["model_id", "reasoning_effort", "problem_key", "is_robust"],
    ]
    frame = features.drop(columns=["is_robust", "budget_summaries"], errors="ignore").merge(
        labels, on=["model_id", "reasoning_effort", "problem_key"], validate="one_to_one"
    ).reset_index(drop=True)
    y = frame["is_robust"].astype(bool).to_numpy()
    decisions = half_rank(frame, model.predict_proba(frame[columns])[:, 1])

    # The incumbent's ranking on the same rows: lower mean NLL is more familiar, called robust.
    nll = frame.merge(
        load_scores()[["problem_key", "scored_model_id", "mean_nll"]],
        left_on=["problem_key", "model_id"], right_on=["problem_key", "scored_model_id"],
        how="left", validate="one_to_one",
    )["mean_nll"].to_numpy(float)
    if np.isnan(nll).any():
        raise ValueError("E23 familiarity scores missing for some official rows")
    familiar = half_rank(frame, -nll)

    metrics = {
        "rows": int(len(frame)),
        "problem_auc": e37["problem_auc"],
        "problem_auc_ci95_template_bootstrap": e37["problem_auc_ci95_template_bootstrap"],
        "row_auc": e37["row_auc"],
        "model_aucs": e37["model_aucs"],
        "half_rank_balanced_accuracy": float(balanced_accuracy_score(y, decisions)),
        "half_rank_accuracy": float(accuracy_score(y, decisions)),
        "familiarity_half_rank_accuracy": float(accuracy_score(y, familiar)),
        "familiarity_half_rank_balanced_accuracy": float(balanced_accuracy_score(y, familiar)),
        "familiarity_row_auc": float(roc_auc_score(y, -nll)),
        "always_robust_accuracy": float(y.mean()),
    }
    gates = {
        "coverage_complete_and_clean": True,  # evaluate() and read_runs() raise otherwise
        "problem_auc_at_least_0_65": metrics["problem_auc"] >= MIN_PROBLEM_AUC,
        "half_rank_balanced_accuracy_above_0_55": metrics["half_rank_balanced_accuracy"] > MIN_BALANCED_ACCURACY,
    }
    return {
        "stage": "E49 official transfer",
        "metrics": metrics,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "runtime_record": runs,
        "e37_gates_for_the_record": e37["gates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--math-dir", type=Path)
    parser.add_argument("--official-rows", type=Path, default=ROOT / "data" / "train_main" / "rows.parquet")
    parser.add_argument("--feature-run", type=Path, action="append")
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--simulate-gates", action="store_true", help="print the gates' false-pass and power table, then exit")
    args = parser.parse_args()
    if args.simulate_gates:
        print("\n".join(simulate_gates(pd.read_parquet(args.official_rows))))
        return
    if not args.math_dir or not args.feature_run:
        parser.error("--math-dir and at least one --feature-run are required")
    archive = verify_archive(args.math_dir)
    result = score(
        read_shards(args.math_dir),
        pd.read_parquet(args.official_rows),
        read_feature_records(args.feature_run),
        read_runs(args.feature_run),
        args.bootstrap,
    )
    result["training_archive_sha256"] = archive
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
