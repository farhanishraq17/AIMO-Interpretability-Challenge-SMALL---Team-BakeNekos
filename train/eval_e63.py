"""E63: why E61's absolute familiarity cutoffs failed, and the same cutoffs recalibrated on the organizers' problems.

E61 (Finding 86) scored 8/14 and 9/14 although the batch ranking it replaced scored 12/14. This script, under
`protocols/E63_PROTOCOL.md`, does three things:

1. Diagnosis: where E61's cutoffs sit inside the NLL distribution of the 36 `train-main-v2` problems, the organizers'
   own problems; and the board score each rule is expected to get if the hidden problems are drawn from that
   distribution with labels following familiarity inside each batch, as E50 recovered (Qwen R S R S R or S R R S R,
   DeepSeek R R S, Skywork and Olmo R S S, most familiar first). Also the ceiling: the best expected score any
   per-model cutoff can reach under that model.
2. E63's cutoffs: E61's `p_m` quantiles, taken over the 36 official problems instead of E55's 548; Qwen at its
   constant `True`.
3. Checks: C2 (DeepSeek MATH, interval lower bound above 0.5) and C3 (the official rows, reported only).

Writes runs/e63/thresholds.json, runs/e63/result.json and the `E63: familiarity cutoffs on the organizers' problems`
section of RESULTS.md.

Run:  python train/eval_e63.py
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
from eval_e61 import E55_RECORDS, P_ROBUST, SHORT  # noqa: E402
from eval_familiarity import DEEPSEEK, INSTANCES, load_scores  # noqa: E402
from eval_problem_level import load_rows  # noqa: E402
from results_io import upsert_section  # noqa: E402

QWEN = "Qwen/Qwen3.5-4B"
# E50's solved hidden set: labels by familiarity rank inside each model's batch, most familiar first.
LABELS = {QWEN: ["RSRSR", "SRRSR"], DEEPSEEK: ["RRS"], "Skywork/Skywork-OR1-Math-7B": ["RSS"],
          "allenai/Olmo-3-7B-Think": ["RSS"]}
DRAWS, SEED = 6000, 63
OUT = ROOT / "runs" / "e63"
HEADING = "E63: familiarity cutoffs on the organizers' problems"


def official_nll() -> dict[str, dict[str, float]]:
    """Each model's E55 mean NLL on the 36 `train-main-v2` problems, by problem key."""
    official = set(pd.read_parquet(ROOT / "data" / "e55" / "instances.parquet").query("dataset_id == 'train-main-v2'")["problem_key"])
    out = {}
    for model, short in SHORT.items():
        rows = [json.loads(line) for path in ROOT.glob(E55_RECORDS.format(short=short))
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[model] = {r["problem_key"]: r["mean_nll"] for r in rows if r["problem_key"] in official}
        assert len(out[model]) == 36, f"{short}: {len(out[model])} official problems"
    return out


def expected(rule: dict, nll: dict, draws: dict) -> dict[str, float]:
    """Expected correct cases per model when each batch is drawn from the official problems (see the docstring).
    `rule[model]` is (threshold, strict) or None for the model's constant."""
    out = {}
    for model, orders in LABELS.items():
        n, counts = len(orders[0]), []
        for keys in draws[n]:
            x = np.array([nll[model][k] for k in keys])
            ranks = np.argsort(x)
            for labels in orders:
                if rule[model] is None:
                    pred = np.full(n, model in (QWEN, DEEPSEEK))
                else:
                    t, strict = rule[model]
                    pred = x < t if strict else x <= t
                counts.append(sum((labels[r] == "R") == pred[i] for r, i in enumerate(ranks)))
        out[model] = float(np.mean(counts))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()
    e54.SEED = SEED

    nll = official_nll()
    keys = sorted(nll[QWEN])
    rng = np.random.default_rng(SEED)
    draws = {n: [rng.choice(keys, n, replace=False) for _ in range(DRAWS)] for n in (3, 5)}
    e61 = json.loads((ROOT / "runs" / "e61" / "thresholds.json").read_text(encoding="utf-8"))

    placement = {m: {"intended": P_ROBUST[m], "e61b_share_called_robust": float(np.mean(np.array(list(v.values())) <= e61["E61b"][m])),
                     "e61a_share_called_robust": float(np.mean(np.array(list(v.values())) < e61["E61a"][m]))}
                 for m, v in nll.items()}
    e63 = {m: float(np.quantile(list(nll[m].values()), P_ROBUST[m])) for m in SHORT if m != QWEN}
    rules = {"E61a (board 8)": {m: (e61["E61a"][m], True) for m in SHORT},
             "E61b (board 9)": {m: (e61["E61b"][m], False) for m in SHORT},
             "E63": {**{m: (t, False) for m, t in e63.items()}, QWEN: None},
             "constants": {m: None for m in SHORT}}
    sims = {name: expected(rule, nll, draws) for name, rule in rules.items()}
    ceiling = {}
    for m in SHORT:
        vals = list(nll[m].values())
        grid = [expected({**{x: None for x in SHORT}, m: (float(np.quantile(vals, q)), False)}, nll, draws)[m]
                for q in np.linspace(0, 1, 21)]
        ceiling[m] = max(max(grid), expected({x: None for x in SHORT}, nll, draws)[m])

    pops = {}
    official = load_rows()
    official = official[official["model_id"].isin(SHORT)].merge(
        load_scores()[["problem_key", "scored_model_id", "mean_nll"]], left_on=["problem_key", "model_id"],
        right_on=["problem_key", "scored_model_id"], validate="one_to_one")
    official["group"] = official["template"]
    official["E63"] = [bool(m in (QWEN, DEEPSEEK)) if m not in e63 else bool(x <= e63[m])
                       for m, x in zip(official["model_id"], official["mean_nll"])]
    pops["P1 official rows (reported only)"] = e54.balanced(official, "E63")
    instances = pd.read_parquet(INSTANCES)
    math = instances[(instances["model_id"] == DEEPSEEK) & instances["dataset_id"].str.startswith("hendrycks")]
    math = math.merge(load_scores().query("scored_model_id == @DEEPSEEK"), on="problem_key", validate="one_to_one")
    math["group"], math["E63"] = math["problem_key"], math["mean_nll"] <= e63[DEEPSEEK]
    pops["P2 DeepSeek MATH (C2)"] = e54.balanced(math, "E63")
    c2 = pops["P2 DeepSeek MATH (C2)"]
    c2_pass = c2["ci"][0] > 0.5

    lines = ["Fixed in `protocols/E63_PROTOCOL.md` before this table was computed. Expected board scores assume the hidden "
             "problems are drawn from the 36 `train-main-v2` problems with labels following familiarity inside each batch "
             f"(E50); {DRAWS} draws, seed {SEED}.", "",
             "| model | intended share robust | E61b's share on the 36 | E61a's share on the 36 | E63 cutoff |", "| --- | --- | --- | --- | --- |"]
    for m in SHORT:
        p = placement[m]
        lines.append(f"| `{m}` | {p['intended']:.2f} | {p['e61b_share_called_robust']:.2f} | {p['e61a_share_called_robust']:.2f} | "
                     + (f"{e63[m]:.4f} |" if m in e63 else "constant `True` |"))
    lines += ["", "| rule | expected board score | " + " | ".join(SHORT[m] for m in SHORT) + " |", "| --- | --- |" + " --- |" * len(SHORT)]
    for name, sim in sims.items():
        lines.append(f"| {name} | **{sum(sim.values()):.2f}** | " + " | ".join(f"{sim[m]:.2f}" for m in SHORT) + " |")
    lines.append(f"| ceiling: best cutoff or constant per model | **{sum(ceiling.values()):.2f}** | "
                 + " | ".join(f"{ceiling[m]:.2f}" for m in SHORT) + " |")
    lines += ["", "| population | rows | robust | E63 balanced accuracy |", "| --- | --- | --- | --- |"]
    lines += [f"| {pop} | {r['n']} | {r['robust_share']:.2f} | {r['balanced_accuracy']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] (acc {r['accuracy']:.3f}) |"
              for pop, r in pops.items()]
    lines += ["", f"**C2: {'pass' if c2_pass else 'fail'}.** C3 reported only; C1 and C5 carry from E61 (identical solution code)."]

    body = "\n".join(lines)
    print(body)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thresholds.json").write_text(json.dumps({"E63": e63}, indent=2), encoding="utf-8")
    (OUT / "result.json").write_text(json.dumps({"placement": placement, "expected": sims, "ceiling": ceiling,
                                                 "populations": pops, "c2_pass": c2_pass}, indent=2), encoding="utf-8")
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", HEADING, body)
        print(f"\nwrote the {HEADING} section of RESULTS.md")


if __name__ == "__main__":
    main()
