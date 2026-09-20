"""E21: does anything we have survive the jump to olympiad problems, or across models?

`aimo-interp/train-main-v2` (Finding 30) finally makes two questions answerable, and both
of them decide whether more modelling is worth any GPU time.

1. **Out-of-sample olympiad.** The shipped E11 artifact was fitted on 138 Hendrycks MATH
   instances. The official split adds 10 DeepSeek olympiad instances that no fit has seen,
   and the private test set is entirely olympiad. Score the frozen artifact on them, then
   refit the leak-free layer ensemble on MATH alone and score it the same way. The char
   tf-idf text model already fails this transfer (0.30 balanced accuracy, Finding 30); the
   question is whether hidden states do any better.

2. **Across models.** Hidden states have a different width per model, but the five
   next-token uncertainty scalars from the same forward pass do not. Pool the four models'
   official rows and hold one model out at a time. Per-model standardisation is legitimate
   here: at evaluation the harness hands over a whole batch for one model.

Every number is tiny-sample (34 official rows, 3 of them Skywork) and is reported with its
n. Treat them as go/no-go evidence, not measurements.

Run (after `datagen/kaggle_launch.py output e21-stagea-*`):

    uv run python ../train/eval_official_holdout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission" / "e11-text-stagea"))

import e11_inference as e11  # noqa: E402
from results_io import upsert_section  # noqa: E402
from train_stagea_probes import UNCERTAINTY_COLUMNS, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
RESULTS = ROOT / "RESULTS.md"
OFFICIAL_DATASET = "train-main-v2"  # the organizers' split; everything else here is MATH

DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
FEATURE_DIRS = {
    DEEPSEEK: "e21-stagea-deepseek-official/features/stagea-deepseek-official",
    "Qwen/Qwen3.5-4B": "e21-stagea-qwen35-official/features/stagea-qwen35-official",
    "Skywork/Skywork-OR1-Math-7B": "e21-stagea-skywork-official/features/stagea-skywork-official",
    "allenai/Olmo-3-7B-Think": "e21-stagea-olmo-think-official/features/stagea-olmo-think-official",
}


def problem_text() -> pd.Series:
    return pd.read_parquet(INSTANCES).set_index(["model_id", "problem_key"])["problem"]


def summarise(name: str, y: np.ndarray, score: np.ndarray, threshold: float = 0.0) -> str:
    pred = score >= threshold
    auc = roc_auc_score(y, score) if len(set(y.tolist())) > 1 else float("nan")
    return (
        f"| {name} | {len(y)} | {(pred == y).mean():.3f} "
        f"| {balanced_accuracy_score(y, pred):.3f} | {auc:.3f} | {int(pred.sum())} |"
    )


def frozen_e11(X: dict[str, np.ndarray], records: pd.DataFrame, texts: pd.Series) -> list[str]:
    """The shipped artifact, unchanged, on the rows it was never fitted on."""
    artifact = e11.load_artifact()
    olympiad = (records["dataset_id"] == OFFICIAL_DATASET).to_numpy()
    y = records["is_robust"].to_numpy(bool)[olympiad]
    problems = [texts[(DEEPSEEK, k)] for k in records["problem_key"][olympiad]]

    text_prob = artifact["text_model"].predict_proba(problems)[:, 1]
    stagea_prob = np.array([
        e11.stagea_probability({p: X[p][i] for p in artifact["poolings"]}, artifact)
        for i in np.flatnonzero(olympiad)
    ])
    return [
        summarise("E11 fused (shipped artifact)", y, e11.fused_decisions(text_prob, stagea_prob, artifact)),
        summarise("E11 text block only", y, e11.text_only_decisions(text_prob, artifact)),
        summarise("all-spurious", y, np.full(len(y), -1.0)),
        summarise("all-robust", y, np.full(len(y), 1.0)),
    ]


def refit_transfer(X: dict[str, np.ndarray], records: pd.DataFrame) -> list[str]:
    """Train the layer ensemble on MATH only, test on the official olympiad rows.

    No layer or pooling is selected, so nothing leaks from the 10 held-out labels.
    """
    olympiad = (records["dataset_id"] == OFFICIAL_DATASET).to_numpy()
    # Train on MATH alone: the two aimo-2025-reference rows are olympiad too, and belong on
    # neither side of a transfer test the shipped artifact has already seen them in.
    math = records["dataset_id"].str.startswith("hendrycks").to_numpy()
    y_train = records["is_robust"].to_numpy(bool)[math]
    y_test = records["is_robust"].to_numpy(bool)[olympiad]

    probabilities = []
    for array in X.values():
        for layer in range(array.shape[1]):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000),
            )
            model.fit(array[math, layer], y_train)
            probabilities.append(model.predict_proba(array[olympiad, layer])[:, 1])
    ensemble = np.mean(probabilities, axis=0)
    return [summarise(f"layer ensemble refit on {int(math.sum())} MATH rows", y_test, ensemble, 0.5)]


def cross_model(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Leave-one-model-out on the width-independent uncertainty scalars, official rows only."""
    parts = []
    for model_id, records in frames.items():
        official = records[records["dataset_id"] == OFFICIAL_DATASET].copy()
        if official.empty:
            continue
        official["model_id"] = model_id
        parts.append(official)
    pooled = pd.concat(parts, ignore_index=True)

    # Standardise within a model: at evaluation the harness supplies one model's whole batch.
    features = pooled.groupby("model_id")[UNCERTAINTY_COLUMNS].transform(
        # A one-row model has an undefined std, and a constant column a zero one; both -> 1.
        lambda column: (column - column.mean()) / (column.std() if column.std() > 0 else 1.0)
    ).to_numpy(float)
    y = pooled["is_robust"].to_numpy(bool)

    rows = []
    for model_id in sorted(set(pooled["model_id"])):
        held = (pooled["model_id"] == model_id).to_numpy()
        if len(set(y[~held].tolist())) < 2 or len(set(y[held].tolist())) < 1:
            continue
        model = LogisticRegression(class_weight="balanced", max_iter=5000).fit(features[~held], y[~held])
        proba = model.predict_proba(features[held])[:, 1]
        prior = np.full(int(held.sum()), float(y[~held].mean()))
        rows.append(summarise(f"held out `{model_id}`", y[held], proba, 0.5))
        rows.append(summarise("&nbsp;&nbsp;other models' base rate", y[held], prior, 0.5))
    return rows


def main() -> None:
    texts = problem_text()
    frames: dict[str, pd.DataFrame] = {}
    deepseek_blocks: list[str] = []

    for model_id, relative in FEATURE_DIRS.items():
        directory = ROOT / "runs" / relative
        if not (directory / "features.npz").is_file():
            print(f"missing {directory} - run `kaggle_launch.py output` for it first")
            continue
        X, records = load(directory, allow_label_mismatch=False)
        frames[model_id] = records
        official = int((records["dataset_id"] == OFFICIAL_DATASET).sum())
        print(f"{model_id}: {len(records)} instances, {official} official")
        if model_id == DEEPSEEK and official:
            deepseek_blocks = frozen_e11(X, records, texts) + refit_transfer(X, records)

    header = [
        "| Method | n | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    body = []
    if deepseek_blocks:
        body += ["### DeepSeek: MATH-trained, olympiad-tested", "", *header, *deepseek_blocks, ""]
    if len(frames) > 1:
        body += [
            "### Cross-model: next-token uncertainty, one model held out",
            "",
            "Five scalars from the same forward pass, standardised within each model.",
            "",
            *header,
            *cross_model(frames),
            "",
        ]
    if not body:
        print("nothing to report yet")
        return
    body += ["Regenerate with `train/eval_official_holdout.py`."]
    upsert_section(RESULTS, "E21: official-split holdouts", "\n".join(body))
    print("\n".join(body))


if __name__ == "__main__":
    main()
