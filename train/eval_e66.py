"""E66: E64 refitted on a difficulty target that is not cut off at 32k tokens.

E64's target is Qwen's mean log trace length at a 32,768-token cap, flat at log(32768) on the 194 problems where Qwen
hit the cap, and Qwen's and DeepSeek's cutoffs sit in that flat region. Under `protocols/E66_PROTOCOL.md` those 194 were
re-sampled at a 65,536-token cap (seed 66, k = 2). This rebuilds the target with them, refits with E64's code
unchanged (`eval_e64.fit_models`), and scores the label-free and official-row gates:

* G1: share of the 497 with both samples at the target's cap, <= 0.142;
* G2: E64's D1 on the new target (5-fold CV Spearman, seed 64), >= 0.30 for every model;
* G3, G4: AUC of -(predicted difficulty) and the shipped decisions' accuracy on the 34 official rows, at least E64's
  exact values (amendment 1);
* reported only: D3, E65's MATH check with the new DeepSeek arrays, how many of the 36 official problems change
  decision, and the new samples' cap-hit rate.

G5 is `submission/make_e64.py --experiment e66`, `scripts/e64_gpu_bundle.py --verify --experiment e66` and
`run_local.py --small`; G6 is one board upload.

Writes runs/e66/artifact.npz, predicted_difficulty.csv, result.json and the `E66: E64 refitted on a 64k-cap target`
section of RESULTS.md.

Run:  python train/eval_e66.py
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

import eval_e64 as e64  # noqa: E402
import eval_e65 as e65  # noqa: E402
from eval_e61 import SHORT  # noqa: E402
from eval_problem_level import load_rows  # noqa: E402
from results_io import upsert_section  # noqa: E402

CAP = 65536
NEW = "runs/e66-qwen-?-*/e66-qwen-?/base/shard_*/records.jsonl"
OUT = ROOT / "runs" / "e66"
HEADING = "E66: E64 refitted on a 64k-cap target"


def tokens(pattern: str) -> dict[str, list[int]]:
    return {r["pool_id"].removeprefix("e62:"): r["completion_tokens"] for path in ROOT.glob(pattern)
            for r in map(json.loads, filter(str.strip, path.read_text(encoding="utf-8").splitlines()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()

    old, new = tokens(e64.BASE), tokens(NEW)
    capped = set(pd.read_parquet(ROOT / "data" / "e66" / "pool.parquet")["pool_id"].str.removeprefix("e62:"))
    assert len(old) == 497 and len(capped) == 194 and set(new) <= capped, (len(old), len(capped), len(set(new) - capped))
    assert all(max(old[k]) < e64.CAP for k in set(old) - capped), "a problem outside the 194 reached the 32k cap"
    missing = sorted(k for k in capped if len(new.get(k, [])) < 2)
    if len(missing) > 0.10 * len(capped):
        raise SystemExit(f"{len(missing)} of {len(capped)} re-sampled problems lack two samples: E66 stops, E64 stays (§3)")
    target = {k: (new[k] if k in capped else old[k]) for k in old if k not in missing}
    y = pd.Series({k: float(np.mean(np.log(np.minimum(t, CAP)))) for k, t in target.items()})
    both = sum(min(t) >= CAP for t in target.values()) / len(old)
    samples = np.concatenate([new[k] for k in capped if k in new])
    hit_rate = float(np.mean(samples >= CAP))
    print(f"target on {len(y)} problems ({len(missing)} dropped); G1 both at cap {both:.3f}; new samples at cap {hit_rate:.3f}")

    arrays, g2, d3, cutoffs, predicted = e64.fit_models(y)
    official = e64.official_keys()
    rows = load_rows()
    rows = rows[rows["model_id"].isin(SHORT)].copy()
    rows["neg_difficulty"] = [-predicted[m].loc[k] for m, k in zip(rows["model_id"], rows["problem_key"])]
    g3 = e64.template_auc(rows, "neg_difficulty")
    truth = rows["is_robust"].astype(bool)
    shipped = pd.Series([bool(predicted[m].loc[k] <= cutoffs[m]) for m, k in zip(rows["model_id"], rows["problem_key"])],
                        index=rows.index)
    g4 = float((shipped == truth).mean())
    ref = json.loads((ROOT / "runs" / "e64" / "result.json").read_text(encoding="utf-8"))
    ref_auc, ref_acc = ref["d2"]["auc"], ref["official_decisions"]["e64"]["accuracy"]
    gates = {"G1": both <= 0.142, "G2": all(v >= 0.30 for v in g2.values()),
             "G3": g3["auc"] >= ref_auc - 1e-9, "G4": g4 >= ref_acc - 1e-9}

    e64_pred = pd.read_csv(ROOT / "runs" / "e64" / "predicted_difficulty.csv", index_col=0)
    changed = {m: int(((predicted[m].loc[official] <= cutoffs[m]) != (e64_pred.loc[official, m] <= ref["cutoffs"][m])).sum())
               for m in SHORT}
    calls = {m: float(np.mean(predicted[m].loc[official] <= cutoffs[m])) for m in SHORT}
    math = e65.math_frame(arrays)
    math["template"], math["easier"] = math["problem_key"], -math["difficulty"]
    math_auc = e64.template_auc(math, "easier")

    verdict = "pass" if all(gates.values()) else "fail"
    mark = {k: "pass" if v else "fail" for k, v in gates.items()}
    lines = ["Fixed in `protocols/E66_PROTOCOL.md` before any sample was drawn. E64's fit, unchanged, on a new target: the "
             f"194 problems E62 sampled at the 32k cap re-sampled by Qwen at a 65,536-token cap (seed 66, k = 2), the other "
             f"303 keep E62's target. {len(y)} problems in the fit, {len(missing)} dropped for lacking two samples.", "",
             f"**G1 (label-free, gate <= 0.142):** {both:.3f} of the 497 have both samples at the cap (E64: 141/497 = 0.284). "
             f"{mark['G1']}. Reported: {hit_rate:.3f} of the {len(samples)} new samples reach 65,536 tokens.", "",
             "| model | G2 CV Spearman (gate >= 0.30) | D3 Spearman with mean NLL | cutoff (E64's share, official 36) | "
             "share of the 36 called robust | of the 36, decisions that differ from E64 |",
             "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| `{m}` | {g2[m]:.3f} | {d3[m]:.3f} | {cutoffs[m]:.4f} | {calls[m]:.2f} | {changed[m]} |" for m in SHORT]
    lines += ["", f"**G3 (official AUC, gate >= E64's {ref_auc:.4f}):** {g3['auc']:.4f} [{g3['ci'][0]:.3f}, {g3['ci'][1]:.3f}] "
              f"on {g3['n']} rows. {mark['G3']}.",
              f"**G4 (official decisions, gate >= E64's {ref_acc:.4f}):** accuracy {g4:.4f} ({int((shipped == truth).sum())}/"
              f"{len(truth)}). {mark['G4']}.", "",
              f"Reported only: E65's check with the new DeepSeek arrays, AUC {math_auc['auc']:.3f} "
              f"[{math_auc['ci'][0]:.3f}, {math_auc['ci'][1]:.3f}] on DeepSeek's {math_auc['n']} MATH rows (E64: 0.346).", "",
              f"**G1-G4: {verdict}.** G5 (contract) and G6 (board) are run after this; the ship rule is the protocol's §6."]
    body = "\n".join(lines)
    print(body)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "artifact.npz", **arrays)
    pd.DataFrame(predicted).to_csv(OUT / "predicted_difficulty.csv")
    (OUT / "result.json").write_text(json.dumps({
        "n_fit": len(y), "dropped": missing, "g1_both_at_cap": both, "new_sample_cap_rate": hit_rate, "g2": g2, "d3": d3,
        "g3": g3, "g4": g4, "e64_reference": {"auc": ref_auc, "accuracy": ref_acc}, "cutoffs": cutoffs,
        "official_called_robust": calls, "official_changed_vs_e64": changed, "math_auc": math_auc, "gates": gates}, indent=2),
        encoding="utf-8")
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", HEADING, body)
        print(f"\nwrote the {HEADING} section of RESULTS.md")


if __name__ == "__main__":
    main()
