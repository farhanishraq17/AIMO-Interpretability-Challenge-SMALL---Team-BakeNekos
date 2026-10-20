"""Rebuild the frozen E37 predictor locally from its pinned MATH archive.

This prepares official-transfer scoring and a reproducible local sklearn artifact;
it does not fit to GSM-Plus or official AIMO outcomes.
Run: python train/prepare_e37_model.py
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib

from eval_e36_budget_path import build_examples, feature_columns, make_model
from run_e37_confirmation import DATASET, REVISION, MODELS, load_partition

ROOT = Path(__file__).resolve().parents[1]


def main():
    confirmation = json.loads((ROOT / "kernels/runs/e37-confirmation-Arifin/result.json").read_text())
    assert confirmation["passed"] and all(confirmation["gates"].values())
    files = [f"data/math_perturb/{model}.parquet" for model in sorted(MODELS)]
    frame, manifests = load_partition("math_perturb", files, ROOT / "data/e37-hf-cache")
    expected = {entry["path"]: entry["sha256"] for entry in confirmation["provenance"]["files"]}
    assert all(entry["sha256"] == expected[entry["path"]] for entry in manifests)
    data = ROOT / "data/e37/math_perturb"
    data.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(data / "selected.parquet", index=False)
    train = build_examples(frame, "math_perturb")
    columns = feature_columns(train)
    model = make_model().fit(train[columns], train.is_robust.to_numpy(bool))
    artifact = ROOT / "data/e37/frozen_predictor.joblib"
    joblib.dump({"model": model, "feature_columns": columns}, artifact)
    result = {"dataset": DATASET, "revision": REVISION, "training_rows": len(train),
              "training_problems": int(train.problem_id.nunique()), "feature_columns": columns,
              "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "files": manifests,
              "packages": {p: importlib.metadata.version(p) for p in
                           ("numpy", "pandas", "scipy", "scikit-learn", "joblib")}}
    dest = ROOT / "kernels/runs/e37-local-predictor"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
