"""Run the frozen E37 confirmation from its pinned public archive, recording provenance.

Download only MATH-Perturb and GSM-Plus. Keep original traces at the four frozen budgets
and variant correctness at the label budget; variant traces never enter the predictor.
The evaluator and gates remain exactly those in E37_PROTOCOL.md.

Run: python train/run_e37_confirmation.py --out-dir runs/e37-confirmation
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from eval_e36_budget_path import BUDGETS, LABEL_BUDGET
from eval_e37_gsm_confirmation import ANCHOR, evaluate

DATASET = "ssubhnil/Qwen3-Math-Eval"
REVISION = "f64c82f5ce9249155ad7c70991a9711d92963cce"
COLUMNS = ["model", "token_budget", "problem_id", "variant", "predicted_answer",
           "correct_strict", "answer_forced", "trace"]
MODELS = {"qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"}


def load_partition(config: str, files: list[str], cache: Path) -> tuple[pd.DataFrame, list[dict]]:
    selected = sorted(f for f in files if f.startswith(f"data/{config}/") and f.endswith(".parquet"))
    if len(selected) != 4:
        raise ValueError(f"expected four model shards for {config}, found {len(selected)}")
    anchor = "base" if config == "math_perturb" else ANCHOR
    frames, manifest = [], []
    for filename in selected:
        path = Path(hf_hub_download(DATASET, filename, repo_type="dataset", revision=REVISION,
                                    cache_dir=cache, token=False))
        frame = pd.read_parquet(path, columns=COLUMNS)
        keep = frame["token_budget"].isin(BUDGETS) & (
            frame["variant"].eq(anchor) | frame["token_budget"].eq(LABEL_BUDGET)
        )
        retained = frame.loc[keep].copy()
        retained.loc[~retained["variant"].eq(anchor), ["trace", "predicted_answer"]] = ""
        frames.append(retained)
        manifest.append({"path": filename, "bytes": path.stat().st_size,
                         "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                         "source_rows": len(frame), "retained_rows": len(retained)})
        print(f"Loaded {filename}: {len(retained)} retained rows", flush=True)
    result = pd.concat(frames, ignore_index=True)
    if set(result["model"]) != MODELS:
        raise ValueError(f"unexpected models in {config}")
    if result.duplicated(["model", "token_budget", "problem_id", "variant"]).any():
        raise ValueError(f"duplicate observations in {config}")
    return result, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/e37_hf_cache"))
    args = parser.parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = HfApi(token=False).list_repo_files(DATASET, repo_type="dataset", revision=REVISION)
    math, math_manifest = load_partition("math_perturb", files, args.cache_dir)
    gsm, gsm_manifest = load_partition("gsm_plus", files, args.cache_dir)
    result = evaluate(math, gsm, bootstrap=5_000)
    result["provenance"] = {
        "dataset": DATASET, "revision": REVISION, "files": math_manifest + gsm_manifest,
        "python": platform.python_version(), "bootstrap_replicates": 5_000,
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "huggingface-hub")},
        "seconds": time.time() - started,
    }
    payload = json.dumps(result, indent=2)
    (args.out_dir / "result.json").write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
