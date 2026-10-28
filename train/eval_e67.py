"""E67: does E64's difficulty direction *cause* longer reasoning in Qwen3.5-4B?

Finding 93 found what E64 reads: a difficulty subspace in the prompt states. A readout is not a mechanism, so
`protocols/E67_PROTOCOL.md` pushed E64's Qwen ridge direction into the residual stream at prefill
(`datagen/perturb/steer.py`, kernels `e67-a` to `e67-d`) and sampled. This scores what was fixed there:

* G1: +2 sigma along u minus no push, in each problem's mean log(min(tokens, 16384)): paired problem-bootstrap 95%
  interval above 0;
* G2: -2 sigma along u minus no push: interval below 0;
* G3 (control): +2 sigma along a random direction minus no push: interval contains 0;
* S (sanity): E64's own Qwen arithmetic on the steered E55 prompt states the kernels saved, averaged over the layers
  past the steered one (the steered layer's own read moves by construction): +2 above no push and -2 below,
  intervals excluding 0.

Reported only: E64's read averaged over every layer, +2 against -2, accuracy, cap hits and declined answers by
condition, and the fp16 parity of the unsteered read against E64's offline predictions. Exploratory, added after the
gates were read: the +2 against -2 change in length that E64's own read of the steered states would predict, were
length to follow it one-for-one (E64's in-sample slope of observed on predicted length, also printed, says how
conservative that is).

Writes runs/e67/result.json and the `E67: steering E64's difficulty direction` section of RESULTS.md.

Run:  python train/eval_e67.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in ("train", "datagen/perturb"):
    sys.path.insert(0, str(ROOT / path))

from eval_e52 import grade  # noqa: E402
from results_io import upsert_section  # noqa: E402

CAP, SEED, BOOT, MIN_PROBLEMS = 16384, 67, 2_000, 45
CONDITIONS = ("minus", "zero", "plus", "random")
RUNS = "runs/e67-[a-d]-*/e67-[a-d]/shard_*"
OUT = ROOT / "runs" / "e67"
HEADING = "E67: steering E64's difficulty direction"


def load() -> dict[str, tuple[dict, Path]]:
    """problem_key -> (record, its shard directory); a resumed kernel's repeat keeps the last copy."""
    found = {}
    for shard in sorted(ROOT.glob(RUNS)):
        for line in (shard / "records.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                found[record["problem_key"]] = (record, shard)
    return found


def interval(diff: np.ndarray) -> dict:
    draws = np.random.default_rng(SEED).integers(0, len(diff), (BOOT, len(diff)))
    lo, hi = np.percentile(diff[draws].mean(axis=1), [2.5, 97.5])
    return {"mean": float(diff.mean()), "ci": [float(lo), float(hi)], "n": int(len(diff))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()

    found = load()
    pool = pd.read_parquet(ROOT / "data" / "e67" / "pool.parquet")
    assert set(found) <= set(pool["problem_key"]), "records outside the E67 pool"
    keys = sorted(found)
    layer = int(np.load(ROOT / "data" / "e67" / "directions.npz")["layer"])

    # Outcome: per problem and condition, the mean over its k samples of log(min(tokens, cap)).
    y = pd.DataFrame({c: [np.mean(np.log(np.minimum(found[k][0]["completion_tokens"][c], CAP))) for k in keys]
                      for c in CONDITIONS}, index=keys)
    contrasts = {"plus": interval((y["plus"] - y["zero"]).to_numpy()),
                 "minus": interval((y["minus"] - y["zero"]).to_numpy()),
                 "random": interval((y["random"] - y["zero"]).to_numpy()),
                 "plus_vs_minus": interval((y["plus"] - y["minus"]).to_numpy())}

    # Sanity: E64's Qwen arithmetic on the steered E55 prompt states, all layers and past the steered one.
    arrays = np.load(ROOT / "runs" / "e64" / "artifact.npz")
    reads, deeper = [], []
    for k in keys:
        states = np.load(found[k][1] / "states" / f"{k}.npz")["states"].astype(np.float32)
        per_layer = (((states - arrays["qwen__mean"]) / arrays["qwen__scale"]) * arrays["qwen__coef"]).sum(-1) \
            + arrays["qwen__intercept"]
        reads.append(per_layer.mean(axis=1))
        deeper.append(per_layer[:, layer + 1:].mean(axis=1))
    reads, deeper = np.array(reads), np.array(deeper)  # (problems, conditions)
    col = {c: i for i, c in enumerate(CONDITIONS)}
    sanity = {name: {c: interval(block[:, col[c]] - block[:, col["zero"]]) for c in ("plus", "minus", "random")}
              for name, block in (("all_layers", reads), ("past_layer", deeper))}
    offline = pd.read_csv(ROOT / "runs" / "e64" / "predicted_difficulty.csv", index_col=0)["Qwen/Qwen3.5-4B"]
    parity = float(np.max(np.abs(reads[:, col["zero"]] - offline.loc[keys].to_numpy())))

    by_condition = {}
    for c in CONDITIONS:
        tokens = np.array([found[k][0]["completion_tokens"][c] for k in keys])
        tails = [t for k in keys for t in found[k][0]["sample_tails"][c]]
        grades = [grade(tail, n, CAP) for tail, n in zip(tails, tokens.ravel())]
        by_condition[c] = {"accuracy": float(np.mean([found[k][0]["n_correct"][c] / tokens.shape[1] for k in keys])),
                           "cap_hit": float((tokens >= CAP).mean()), "declined": grades.count("detected") / len(grades),
                           "geometric_mean_tokens": float(np.exp(np.log(np.minimum(tokens, CAP)).mean()))}
    seconds = [found[k][0]["seconds"] for k in keys]
    sessions = sum(json.loads(path.read_text())["end_to_end_seconds"] for path in ROOT.glob("runs/e67-[a-d]-*/e67-[a-d]/execution.json"))

    # Exploratory (after the gates): what E64's read of the steered states predicts for plus - minus, one-for-one.
    from eval_e64 import difficulty
    target = difficulty()
    slope = float(np.polyfit(offline.loc[target.index].to_numpy(), target.to_numpy(), 1)[0])
    expected = {name: float((block[:, col["plus"]] - block[:, col["minus"]]).mean())
                for name, block in (("all_layers", reads), ("past_layer", deeper))}

    enough = len(keys) >= MIN_PROBLEMS
    gates = {"G1": contrasts["plus"]["ci"][0] > 0, "G2": contrasts["minus"]["ci"][1] < 0,
             "G3": contrasts["random"]["ci"][0] <= 0 <= contrasts["random"]["ci"][1],
             "S": sanity["past_layer"]["plus"]["ci"][0] > 0 and sanity["past_layer"]["minus"]["ci"][1] < 0}

    def fmt(r: dict) -> str:
        return f"{r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"

    lines = [f"Fixed in `protocols/E67_PROTOCOL.md` before any sample. Qwen3.5-4B, E62's base recipe at a {CAP:,}-token cap, "
             f"k = {len(found[keys[0]][0]['completion_tokens']['zero'])} per condition; the push is added at hidden-state "
             f"index {layer} to every prompt token during prefill. {len(keys)} of {len(pool)} problems complete"
             f"{'' if enough else f' (fewer than {MIN_PROBLEMS}: underpowered, no reading)'}. Outcome: each problem's mean "
             "log(min(tokens, cap)); 95% paired problem-bootstrap intervals.", "",
             "| contrast | change in mean log trace length | gate | result |", "| --- | --- | --- | --- |",
             f"| +2 sigma along u, minus no push | {fmt(contrasts['plus'])} | G1: above 0 | {'pass' if gates['G1'] else 'fail'} |",
             f"| -2 sigma along u, minus no push | {fmt(contrasts['minus'])} | G2: below 0 | {'pass' if gates['G2'] else 'fail'} |",
             f"| +2 sigma along random r, minus no push | {fmt(contrasts['random'])} | G3: contains 0 | {'pass' if gates['G3'] else 'fail'} |",
             f"| +2 minus -2 along u | {fmt(contrasts['plus_vs_minus'])} | reported | |", "",
             "Sanity (S): E64's Qwen read of the steered E55 prompt states, minus the unsteered read (log-token units).", "",
             "| push | layers past the steered one (S) | all layers averaged |", "| --- | --- | --- |"]
    lines += [f"| {c} | {fmt(sanity['past_layer'][c])} | {fmt(sanity['all_layers'][c])} |" for c in ("plus", "minus", "random")]
    lines += ["", f"**S: {'pass' if gates['S'] else 'fail'}.** Unsteered read against E64's offline predictions (fp16 here, "
              f"bf16 in E55): max abs diff {parity:.3f}.", "",
              "| condition | accuracy | cap hit | declined | geometric mean tokens |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {c} | {v['accuracy']:.3f} | {v['cap_hit']:.3f} | {v['declined']:.3f} | {v['geometric_mean_tokens']:,.0f} |"
              for c, v in by_condition.items()]
    lines += ["", "Exploratory, added after the gates were read: were trace length to follow E64's read of the steered states "
              f"one-for-one, +2 minus -2 along u would be {expected['all_layers']:+.3f} (every layer averaged) to "
              f"{expected['past_layer']:+.3f} (past the steered layer); observed {fmt(contrasts['plus_vs_minus'])}. E64's "
              f"in-sample slope of observed on predicted length over E62's 497 problems is {slope:.2f}, so one-for-one is "
              "the smaller expectation.",
              "", f"Run time: {np.sum(seconds) / 3600:.1f} GPU-process hours, median {np.median(seconds) / 60:.1f} min a problem; "
              f"{sessions / 3600:.1f} Kaggle session-hours for the four kernels.",
              "", f"**Gates: G1 {'pass' if gates['G1'] else 'fail'}, G2 {'pass' if gates['G2'] else 'fail'}, "
              f"G3 {'pass' if gates['G3'] else 'fail'}, S {'pass' if gates['S'] else 'fail'}.**"]
    body = "\n".join(lines)
    print(body)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps({"n_problems": len(keys), "enough": enough, "contrasts": contrasts,
                                                 "sanity": sanity, "parity_max_abs": parity, "by_condition": by_condition,
                                                 "gates": gates, "exploratory": {"expected_plus_vs_minus": expected,
                                                                                 "calibration_slope": slope}},
                                                indent=2), encoding="utf-8")
    y.to_csv(OUT / "log_length.csv")
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", HEADING, body)
        print(f"\nwrote the {HEADING} section of RESULTS.md")


if __name__ == "__main__":
    main()
