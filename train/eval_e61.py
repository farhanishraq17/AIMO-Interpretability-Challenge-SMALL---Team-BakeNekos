"""E61: the familiarity ranking as an absolute per-problem threshold, thresholds and offline checks.

`e27k`'s batch ranking cannot act on one-problem calls (Finding 79). E61 keeps its statistic (mean
token NLL of the bare problem) and replaces the batch rank with a per-model threshold fixed on
unlabelled problems, under `protocols/E61_PROTOCOL.md`, which fixed every rule below before this
script existed:

* E61b, the candidate: robust iff NLL <= the p_m quantile of the model's NLLs on E55's 548 unlabelled
  problems, with p_m = (n_m + d_m) / 2n_m from the probe balances (Qwen 0.6, DeepSeek 2/3, Skywork
  and Olmo 1/3).
* E61a, the control: E54's frozen S1, below the 75-problem AMC reference median.

Checks computed here: C2 (E61b on DeepSeek's 142 MATH instances, interval lower bound above 0.5),
C3 (both rules on the 34 official rows, reported only) and C4 (threshold stability, reported only).
C1, C5 and C6 need the ZIP and a GPU (`submission/make_e61.py`, `scripts/e61_gpu_bundle.py`).

Writes runs/e61/thresholds.json (read by make_e61.py), runs/e61/result.json and the
`E61: familiarity as a per-problem threshold` section of RESULTS.md.

Run:  python train/eval_e61.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in ("train", "scripts", "datagen"):
    sys.path.insert(0, str(ROOT / path))

import eval_e54_per_problem as e54  # noqa: E402
from eval_familiarity import DEEPSEEK, INSTANCES, load_scores  # noqa: E402
from eval_problem_level import load_rows  # noqa: E402
from results_io import upsert_section  # noqa: E402

E55_RECORDS = "runs/e55-features-*/e55-features-{short}/familiarity-{short}/shard_*/records.jsonl"
SHORT = e54.SHORT
# (n_m + d_m) / 2n_m: E50's case counts and the probes' balances (Findings 45, 64, 75).
P_ROBUST = {"Qwen/Qwen3.5-4B": (5 + 1) / 10, DEEPSEEK: (3 + 1) / 6,
            "Skywork/Skywork-OR1-Math-7B": (3 - 1) / 6, "allenai/Olmo-3-7B-Think": (3 - 1) / 6}
MARGIN = 0.01  # nats; C4's numeric-margin band
OUT = ROOT / "runs" / "e61"


def reference() -> dict[str, np.ndarray]:
    """Each model's mean NLL on E55's 548 problems."""
    out = {}
    for model, short in SHORT.items():
        rows = [json.loads(line) for path in ROOT.glob(E55_RECORDS.format(short=short))
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert {r["scored_model_id"] for r in rows} == {model}, f"{short}: records from another model"
        keys = {r["problem_key"] for r in rows}
        assert len(rows) == len(keys) == 548, f"{short}: {len(rows)} records, {len(keys)} problems"
        out[model] = np.array([r["mean_nll"] for r in rows])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()
    e54.SEED = 61  # the protocol's bootstrap seed; everything else in `balanced` is E54's

    ref = reference()
    rule_b = {m: float(np.quantile(v, P_ROBUST[m])) for m, v in ref.items()}
    rule_a = {m: json.loads((ROOT / "runs" / "e54" / "result.json").read_text(encoding="utf-8"))
              ["thresholds"]["nll_reference_median"][m] for m in SHORT}

    nll = load_scores()[["problem_key", "scored_model_id", "mean_nll"]]
    official = load_rows()
    official = official[official["model_id"].isin(SHORT)].merge(
        nll, left_on=["problem_key", "model_id"], right_on=["problem_key", "scored_model_id"], validate="one_to_one")
    official["group"] = official["template"]
    official["E61a"] = official["mean_nll"] < official["model_id"].map(rule_a)
    official["E61b"] = official["mean_nll"] <= official["model_id"].map(rule_b)

    instances = pd.read_parquet(INSTANCES)
    math = instances[(instances["model_id"] == DEEPSEEK) & instances["dataset_id"].str.startswith("hendrycks")]
    math = math.merge(nll[nll["scored_model_id"] == DEEPSEEK], on="problem_key", validate="one_to_one")
    math["group"] = math["problem_key"]
    math["E61a"] = math["mean_nll"] < rule_a[DEEPSEEK]
    math["E61b"] = math["mean_nll"] <= rule_b[DEEPSEEK]

    cells = {pop: {rule: e54.balanced(frame, rule) for rule in ("E61a", "E61b")}
             for pop, frame in (("P1 official rows", official), ("P2 DeepSeek MATH", math))}
    c2 = cells["P2 DeepSeek MATH"]["E61b"]
    c2_pass = c2["ci"][0] > 0.5
    stability = {m: {"p": P_ROBUST[m], "e61b": rule_b[m], "e61a": rule_a[m],
                     "e61a_quantile_in_R": float((ref[m] < rule_a[m]).mean()),
                     "share_within_margin": float((np.abs(ref[m] - rule_b[m]) <= MARGIN).mean())}
                 for m in SHORT}

    lines = ["Rules fixed in `protocols/E61_PROTOCOL.md` before this table was computed. Balanced accuracy of "
             "the per-problem decisions, 95% problem-bootstrap interval (2,000 draws, seed 61); any constant "
             "scores 0.5. E61a: below E54's 75-problem AMC median. E61b: at or below the p_m quantile of E55's "
             "548 unlabelled problems.", "",
             "| population | rows | robust | E61a | E61b |", "| --- | --- | --- | --- | --- |"]
    for pop, rules in cells.items():
        size = rules["E61b"]
        lines.append(f"| {pop} | {size['n']} | {size['robust_share']:.2f} | "
                     + " | ".join(f"{m['balanced_accuracy']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}] "
                                  f"(acc {m['accuracy']:.3f})" for m in rules.values()) + " |")
    lines += ["", "| model | p_m | E61b threshold | E61a threshold | E61a's quantile in R | R within 0.01 nats of E61b |",
              "| --- | --- | --- | --- | --- | --- |"]
    for m, s in stability.items():
        lines.append(f"| `{m}` | {s['p']:.3f} | {s['e61b']:.4f} | {s['e61a']:.4f} | {s['e61a_quantile_in_R']:.3f} | "
                     f"{s['share_within_margin']:.3f} |")
    lines += ["", f"**C2 (E61b on DeepSeek MATH, interval lower bound above 0.5): {'pass' if c2_pass else 'fail'}** "
              f"({c2['balanced_accuracy']:.3f} [{c2['ci'][0]:.3f}, {c2['ci'][1]:.3f}]). C3 and C4 are reported only. "
              "C1, C5 and C6 are decided on the ZIP."]

    body = "\n".join(lines)
    print(body)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thresholds.json").write_text(json.dumps({"E61a": rule_a, "E61b": rule_b}, indent=2), encoding="utf-8")
    (OUT / "result.json").write_text(json.dumps({"cells": cells, "stability": stability, "c2_pass": c2_pass},
                                                indent=2), encoding="utf-8")
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E61: familiarity as a per-problem threshold", body)
        print("\nwrote the E61 section of RESULTS.md")


if __name__ == "__main__":
    main()
