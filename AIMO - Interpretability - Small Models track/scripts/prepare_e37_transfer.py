"""Materialize the pinned official input only after E37 confirmation passes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from inspect_train_main_v2 import DATASET, REVISION, PARQUET, SMALL_TRACK, expected_label

ROOT = Path(__file__).resolve().parents[1]


def main():
    confirmation = ROOT / "runs/e37-confirmation-Arifin/e37/result.json"
    result = json.loads(confirmation.read_text())
    assert result["passed"] and all(result["gates"].values()), "E37 gate closed"
    path = Path(hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=REVISION,
                              cache_dir=ROOT / "data/hf-cache"))
    frame = pd.read_parquet(path)
    assert all(expected_label(r.max_drop) == r.is_robust for r in frame.itertuples())
    frame["problem_key"] = frame.problem.map(
        lambda text: hashlib.sha1(" ".join(text.split()).encode()).hexdigest()[:16])
    frame["is_small_track"] = frame.model_id.isin(SMALL_TRACK)
    frame["is_labelled"] = frame.is_robust.notna()
    frame["reasoning_effort"] = frame.reasoning_effort.fillna("default")
    small = frame.loc[frame.is_small_track & frame.is_labelled]
    counts = small.groupby("model_id").size().to_dict()
    assert counts == {"Qwen/Qwen3.5-4B": 12, "Skywork/Skywork-OR1-Math-7B": 3,
                      "allenai/Olmo-3-7B-Think": 9, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 10}
    assert not small.duplicated(["model_id", "problem_key"]).any()
    dest = ROOT / "data/train_main"
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest / "rows.parquet", index=False)
    record = {"dataset": DATASET, "revision": REVISION, "file": PARQUET,
              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "rows_sha256": hashlib.sha256((dest / "rows.parquet").read_bytes()).hexdigest(),
              "small_labelled_counts": counts,
              "confirmation_sha256": hashlib.sha256(confirmation.read_bytes()).hexdigest()}
    (dest / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
