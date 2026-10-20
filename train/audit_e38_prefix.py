"""Audit exact cross-budget prefix identity without reading labels (E38a).

Run: python train/audit_e38_prefix.py --out-dir runs/e38-prefix-audit
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET = "ssubhnil/Qwen3-Math-Eval"
REVISION = "f64c82f5ce9249155ad7c70991a9711d92963cce"
MODELS = ("qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b")
BUDGETS = (2048, 4096, 8192, 16384)
COLUMNS = ["model", "problem_id", "variant", "token_budget", "trace", "temperature", "seed", "answer_forced"]


def audit_cell(frame: pd.DataFrame, anchor: str) -> dict:
    rows = frame.loc[frame.variant.eq(anchor) & frame.token_budget.isin(BUDGETS)].copy()
    if rows.empty or rows.duplicated(["problem_id", "token_budget"]).any():
        raise ValueError("empty or duplicate anchor observations")
    if not rows.temperature.eq(0).all() or not rows.seed.eq(42).all():
        raise ValueError("non-frozen sampling settings")
    if rows.trace.isna().any() or rows.trace.str.len().eq(0).any():
        raise ValueError("empty trace")
    counts = {str(b): {"comparisons": 0, "mismatches": 0, "forced_comparisons": 0,
                      "forced_mismatches": 0, "natural_comparisons": 0, "natural_mismatches": 0}
              for b in BUDGETS[:-1]}
    problems = 0
    for _, part in rows.groupby("problem_id", sort=True):
        if set(part.token_budget) != set(BUDGETS) or len(part) != 4:
            raise ValueError("incomplete budget matrix")
        part = part.set_index("token_budget")
        longest = part.loc[16384, "trace"].encode("utf-8")
        for budget in BUDGETS[:-1]:
            shorter = part.loc[budget, "trace"].encode("utf-8")
            mismatch = not longest.startswith(shorter)
            kind = "forced" if bool(part.loc[budget, "answer_forced"]) else "natural"
            c = counts[str(budget)]
            c["comparisons"] += 1
            c["mismatches"] += int(mismatch)
            c[f"{kind}_comparisons"] += 1
            c[f"{kind}_mismatches"] += int(mismatch)
        problems += 1
    return {"problems": problems, "budgets": counts,
            "comparisons": sum(c["comparisons"] for c in counts.values()),
            "mismatches": sum(c["mismatches"] for c in counts.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/e38-hf-cache"))
    args = parser.parse_args()
    cells, files = {}, []
    for source, anchor in (("math_perturb", "base"), ("gsm_plus", "problem understanding")):
        for model in MODELS:
            name = f"data/{source}/{model}.parquet"
            path = Path(hf_hub_download(DATASET, name, repo_type="dataset", revision=REVISION,
                                       cache_dir=args.cache_dir, token=False))
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            frame = pd.read_parquet(path, columns=COLUMNS)
            if set(frame.model) != {model}:
                raise ValueError(f"unexpected model in {name}")
            result = audit_cell(frame, anchor)
            cells[f"{source}/{model}"] = result
            files.append({"path": name, "sha256": digest, "bytes": path.stat().st_size})
            print(f"{source}/{model}: {result['mismatches']}/{result['comparisons']} mismatches", flush=True)
    gates = {"at_least_50_problems_per_cell": all(c["problems"] >= 50 for c in cells.values()),
             "exact_prefixes_everywhere": all(c["mismatches"] == 0 for c in cells.values())}
    result = {"stage": "E38a exact-prefix audit", "dataset": DATASET, "revision": REVISION,
              "columns_read": COLUMNS, "cells": cells, "files": files,
              "comparisons": sum(c["comparisons"] for c in cells.values()),
              "mismatches": sum(c["mismatches"] for c in cells.values()),
              "gates": gates, "passed": all(gates.values()),
              "packages": {p: importlib.metadata.version(p) for p in ("pandas", "pyarrow", "huggingface-hub")}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"gates": gates, "passed": result["passed"]}), flush=True)


if __name__ == "__main__":
    main()
