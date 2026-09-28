"""Preserve the E37 transfer runtime evidence and regenerate its result table.

Run with --collect after downloading the three accessible retry outputs; then run
without arguments to rebuild RESULTS.md using only the committed evidence.
No incomplete-data accuracy scores are computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from results_io import upsert_section

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ("qwen", "MUBAMAN", "Qwen/Qwen3.5-4B"),
    ("skywork", "Arifin", "Skywork/Skywork-OR1-Math-7B"),
    ("deepseek", "Nipun", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"),
)
RESULT = ROOT / "kernels/runs/e37-runtime-result.json"


def collect():
    import pandas as pd

    official_path = ROOT / "data/train_main/rows.parquet"
    official_hash = hashlib.sha256(official_path.read_bytes()).hexdigest()
    official = pd.read_parquet(official_path)
    official = official.loc[official.is_small_track.astype(bool) & official.is_labelled.astype(bool)]
    records = []
    for name, account, model in RUNS:
        job = f"e37-retry-{name}-{account}"
        source = ROOT / "runs" / job / f"e37-{name}"
        destination = ROOT / "kernels/runs" / job
        launch = json.loads((destination / "launch.json").read_text())
        assert official_hash == launch["source_sha256"]["data/train_main/rows.parquet"]
        metadata = json.loads((source / "metadata.json").read_text())
        execution = json.loads((source / "execution.json").read_text())
        features = [json.loads(line) for line in (source / "features.jsonl").read_text().splitlines() if line.strip()]
        keys = [row["problem_key"] for row in features]
        expected = set(official.loc[official.model_id == model, "problem_key"])
        assert len(keys) == len(set(keys)) and set(keys) <= expected
        assert metadata["model_id"] == model
        assert metadata["budgets"] == [2048, 4096, 8192, 16384]
        assert all(row["model_id"] == model for row in features)
        assert len(features) == metadata["rows_done_this_session"]
        complete = execution["exit_code"] == 0 and set(keys) == expected
        files = {}
        for filename in ("metadata.json", "execution.json", "cuda-link-check.json", "environment.txt", "features.jsonl", "runner.log"):
            path = source / filename
            files[filename] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            # Traces and row-level features remain in ignored runs/, retrievable from Kaggle.
            if filename not in ("features.jsonl", "runner.log"):
                shutil.copyfile(path, destination / filename)
        records.append({
            "model": name, "account": account, "ref": launch["ref"],
            "rows": len(features), "expected_rows": len(expected), "complete": complete,
            "extraction_seconds": metadata["seconds"],
            "process_seconds": execution["process_seconds"],
            "end_to_end_seconds": execution["end_to_end_seconds"],
            "termination_reason": execution["termination_reason"], "files": files,
        })
    completed_seconds = round(sum(row["extraction_seconds"] for row in records if row["complete"]), 1)
    # Even this conservative lower bound already exceeds the frozen whole-run limit.
    assert completed_seconds >= 3000
    result = {
        "stage": "E37 official transfer runtime rejection",
        "official_rows_sha256": official_hash, "runs": records,
        "recovered_rows": sum(row["rows"] for row in records), "expected_rows": len(official),
        "unavailable_run": {"model": "olmo", "account": "Tamim3",
                            "ref": "abrarrafi1552/aimo-small-e37-retry-olmo",
                            "status": "access_denied", "rows": None},
        "completed_models_runtime_lower_bound_seconds": completed_seconds,
        "runtime_limit_seconds": 3000, "runtime_gate_passed": False,
        "accuracy_evaluated": False, "submission_eligible": False,
        "decision": "Close E37 under its frozen runtime gate; no further GPU retry or Codabench submission.",
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def report():
    result = json.loads(RESULT.read_text())
    lines = [
        "Source: `kernels/runs/e37-runtime-result.json`; exact execution metadata and output hashes are preserved beside each launch.", "",
        "| Model | Account | Recovered rows | Extraction seconds | Outcome |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in result["runs"]:
        outcome = "complete" if row["complete"] else row["termination_reason"]
        lines.append(f"| {row['model']} | {row['account']} | {row['rows']}/{row['expected_rows']} | {row['extraction_seconds']:,.1f} | {outcome} |")
    lines += [
        "| olmo | Tamim3 | unknown | unknown | API access denied; token identity verified |", "",
        f"Recovered coverage: **{result['recovered_rows']}/{result['expected_rows']}**. No incomplete-data accuracy or AUC is reported.", "",
        f"Completed models alone took **{result['completed_models_runtime_lower_bound_seconds']:,.1f} seconds**, "
        f"already above the frozen **{result['runtime_limit_seconds']:,}-second** whole-run limit. "
        "These extraction times include model loading and generation; environment installation is additional. "
        "DeepSeek's guard stopped the process at two hours, with partial rows preserved.", "",
        "**Decision: E37 fails the runtime gate and is closed without a Codabench submission.** "
        "This does not establish an accuracy outcome on the official set. E27k remains the incumbent.", "",
        "Rebuild this table with `python train/report_e37_runtime.py`. "
        "After downloading the listed outputs, add `--collect` to validate coverage and rebuild the evidence JSON.",
    ]
    upsert_section(ROOT / "RESULTS.md", "E37: official transfer runtime gate", "\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    if parser.parse_args().collect:
        collect()
    report()
