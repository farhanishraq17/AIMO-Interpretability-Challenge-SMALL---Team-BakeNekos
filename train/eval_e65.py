"""E65: E64's difficulty score on DeepSeek's MATH rows, a population it was never read on.

E64 dropped its MATH check (D4) because E55's states do not cover those problems; `e1b-stagea-deepseek` holds them in
the same format (extractor, system prompt, chat template, 2048-token limit, mean pooling, bf16). Under
`protocols/E65_PROTOCOL.md` this applies E64's shipped DeepSeek arithmetic, unrefitted, to those states:

* gate: AUC of -(predicted difficulty) on the 137 rows, row-bootstrap interval, lower bound above 0.5;
* reported: balanced accuracy at E64's DeepSeek cutoff, mean NLL's AUC on the same rows, and their Spearman.

Writes runs/e65/result.json and the `E65: E64's difficulty on DeepSeek MATH` section of RESULTS.md.

Run:  python train/eval_e65.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for path in ("train", "scripts", "datagen", "submission/e64-difficulty"):
    sys.path.insert(0, str(ROOT / path))

import eval_e54_per_problem as e54  # noqa: E402
import eval_e64 as e64  # noqa: E402
import solution  # noqa: E402  (E64's shipped arithmetic)
from eval_familiarity import DEEPSEEK, INSTANCES, load_scores  # noqa: E402
from results_io import upsert_section  # noqa: E402

STATES = ROOT / "runs" / "e1b-stagea-deepseek" / "features" / "stagea-deepseek" / "features.npz"
OUT = ROOT / "runs" / "e65"
HEADING = "E65: E64's difficulty on DeepSeek MATH"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()
    e64.SEED = e54.SEED = 65

    arrays = dict(np.load(ROOT / "runs" / "e64" / "artifact.npz"))
    bundle = np.load(STATES)
    index = {str(k): i for i, k in enumerate(bundle["problem_keys"])}
    X = bundle["X_mean"].astype(np.float32)

    instances = pd.read_parquet(INSTANCES)
    math = instances[(instances["model_id"] == DEEPSEEK) & instances["dataset_id"].str.startswith("hendrycks")]
    math = math[math["problem_key"].isin(index)].copy()
    assert len(math) == 137 and math["is_robust"].sum() == 36, (len(math), math["is_robust"].sum())
    math["difficulty"] = [solution.difficulty(arrays, "deepseek", X[index[k]]) for k in math["problem_key"]]
    nll = load_scores()
    math = math.merge(nll.loc[nll["scored_model_id"] == DEEPSEEK, ["problem_key", "mean_nll"]], on="problem_key",
                      validate="one_to_one")
    math["template"] = math["group"] = math["problem_key"]
    math["easier"], math["familiar"] = -math["difficulty"], -math["mean_nll"]
    cutoff = float(arrays["deepseek__cutoff"])
    math["E64"] = math["difficulty"] <= cutoff

    gate = e64.template_auc(math, "easier")
    result = {"rows": len(math), "robust": int(math["is_robust"].sum()), "auc_difficulty": gate,
              "auc_nll": e64.template_auc(math, "familiar"), "decisions": e54.balanced(math, "E64"),
              "called_robust": float(math["E64"].mean()), "cutoff": cutoff,
              "spearman_nll": float(spearmanr(math["difficulty"], math["mean_nll"]).statistic),
              "pass": gate["ci"][0] > 0.5}

    def cell(m: dict, key: str) -> str:
        return f"{m[key]:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"

    d = result["decisions"]
    body = "\n".join([
        "Fixed in `protocols/E65_PROTOCOL.md` before any score was computed on these rows. E64's shipped DeepSeek "
        "arithmetic (nothing refitted) on `e1b-stagea-deepseek`'s prompt states for DeepSeek's Hendrycks MATH "
        f"instances: {result['rows']} rows, {result['robust']} robust, older label rule. 95% row-bootstrap intervals "
        "(2,000 draws, seed 65).", "",
        "| score | value |", "| --- | --- |",
        f"| **gate: AUC of -(predicted difficulty)** | **{cell(result['auc_difficulty'], 'auc')}** |",
        f"| AUC of -(mean NLL), E61's statistic, same rows | {cell(result['auc_nll'], 'auc')} |",
        f"| E64's decisions at DeepSeek's cutoff ({cutoff:.4f}): balanced accuracy | {cell(d, 'balanced_accuracy')} "
        f"(accuracy {d['accuracy']:.3f}; {result['called_robust']:.2f} called robust) |",
        f"| Spearman, predicted difficulty vs mean NLL | {result['spearman_nll']:.3f} |", "",
        f"**Gate (lower bound above 0.5): {'pass' if result['pass'] else 'fail'}.**"])
    print(body)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", HEADING, body)
        print(f"\nwrote the {HEADING} section of RESULTS.md")


if __name__ == "__main__":
    main()
