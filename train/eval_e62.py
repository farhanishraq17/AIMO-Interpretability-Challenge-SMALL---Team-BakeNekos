"""E62: does Qwen flag added contradictions less often on the problems it finds familiar?

`protocols/E62_PROTOCOL.md` fixed everything here before any sample existed. Qwen3.5-4B sampled E62's 497
non-official olympiad problems (k = 2, 32k cap, the prompt that can decline); an eligible problem (both samples
correct) then got k = 4 on each of its two kept premise contradictions. A sample is flagged when E52's grader
scores it `detected`. For every eligible problem with all 8 contradiction samples, the detection rate is
flagged / 8, and its familiarity is Qwen's E55 mean NLL (lower = more familiar).

* Primary: Spearman(mean NLL, detection rate), 95% problem-bootstrap interval (2,000 draws, seed 62). The
  memorisation mechanism predicts a positive value.
* Secondary: the same within character-length terciles, averaged with equal weight; detection by NLL quintile.
* Fewer than 40 qualifying problems: reported as underpowered.

Writes runs/e62/result.json and the `E62: familiarity and contradiction detection` section of RESULTS.md.

Run:  python train/eval_e62.py
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
sys.path.insert(0, str(ROOT / "train"))

from eval_e52 import grade  # noqa: E402
from eval_e58 import RECORDS, _records, correct  # noqa: E402
from results_io import upsert_section  # noqa: E402

CAP, N_SAMPLES, MIN_PROBLEMS, BOOT, SEED = 32768, 8, 40, 2_000, 62
E55_QWEN = "runs/e55-features-qwen-*/e55-features-qwen/familiarity-qwen/shard_*/records.jsonl"
HEADING = "E62: familiarity and contradiction detection"


def problems() -> pd.DataFrame:
    """One row per sampled problem: eligibility, contradiction outcomes, Qwen's mean NLL and length."""
    for phase in ("base", "variants"):
        for meta in ROOT.glob(RECORDS.format(experiment="e62", short="qwen", phase=phase).replace("records.jsonl", "metadata.json")):
            assert json.loads(meta.read_text(encoding="utf-8"))["args"]["max_new_tokens"] == CAP, meta
    base = {r["pool_id"]: r for r in _records("e62", "qwen", "base")}
    if not base:
        raise SystemExit("no e62 base records; download the e62-qwen-* kernels first")
    variants: dict[str, list[dict]] = {}
    for record in _records("e62", "qwen", "variants"):
        variants.setdefault(record["pool_id"], []).append(record)
    nll = {r["problem_key"]: r["mean_nll"] for p in ROOT.glob(E55_QWEN)
           for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}
    pool = pd.read_parquet(ROOT / "data" / "e62" / "pool.parquet").set_index("pool_id")
    rows = []
    for pool_id, record in base.items():
        outcomes = [grade(tail, tokens, CAP) for v in variants.get(pool_id, [])
                    for tail, tokens in zip(v["sample_tails"], v["completion_tokens"])]
        rows.append({"pool_id": pool_id, "problem_key": pool.at[pool_id, "problem_key"],
                     "length": len(pool.at[pool_id, "problem"]), "mean_nll": nll[pool.at[pool_id, "problem_key"]],
                     "base_samples": record["n_samples"], "base_truncated": sum(t >= CAP for t in record["completion_tokens"]),
                     "eligible": record["n_samples"] == 2 and all(correct(record, CAP)),
                     "n_variants": len(variants.get(pool_id, [])), "n_contradiction": len(outcomes),
                     **{o: outcomes.count(o) for o in ("detected", "answered", "truncated", "other")}})
    frame = pd.DataFrame(rows)
    frame["qualifies"] = frame["eligible"] & (frame["n_variants"] == 2) & (frame["n_contradiction"] == N_SAMPLES)
    frame["detection"] = frame["detected"] / N_SAMPLES
    return frame


def tercile_spearman(x: np.ndarray, y: np.ndarray, length: np.ndarray) -> float:
    """Spearman within each character-length tercile, averaged with equal weight (nan terciles skipped)."""
    cuts = np.quantile(length, [1 / 3, 2 / 3])
    bins = np.digitize(length, cuts)
    values = [spearmanr(x[bins == b], y[bins == b]).statistic for b in range(3) if (bins == b).sum() > 2]
    return float(np.nanmean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()

    frame = problems()
    q = frame[frame["qualifies"]].reset_index(drop=True)
    x, y, length = q["mean_nll"].to_numpy(), q["detection"].to_numpy(), q["length"].to_numpy()
    result = {"sampled": int(len(frame)), "eligible": int(frame["eligible"].sum()), "qualifying": int(len(q)),
              "base_cap_hit": float(frame["base_truncated"].sum() / max(1, frame["base_samples"].sum())),
              "contradiction_cap_hit": float(frame["truncated"].sum() / max(1, frame["n_contradiction"].sum())),
              "mean_detection": float(y.mean()) if len(q) else None}
    lines = ["Fixed in `protocols/E62_PROTOCOL.md` (with amendments) before any sample existed. Qwen3.5-4B, 32k cap, "
             "the prompt that can decline; detection rate = flagged / 8 over two premise contradictions, k = 4 each.", "",
             f"Sampled {result['sampled']} problems; eligible {result['eligible']}; qualifying (eligible, two kept variants, "
             f"8 samples) **{result['qualifying']}**. Cap hit: base {result['base_cap_hit']:.3f}, contradictions "
             f"{result['contradiction_cap_hit']:.3f}."]
    if len(q) < MIN_PROBLEMS:
        result["reading"] = "underpowered"
        lines += ["", f"**Underpowered:** fewer than {MIN_PROBLEMS} qualifying problems (protocol §4); no reading."]
    else:
        rng = np.random.default_rng(SEED)
        draws = [rng.integers(0, len(q), len(q)) for _ in range(BOOT)]
        primary = float(spearmanr(x, y).statistic)
        within = tercile_spearman(x, y, length)
        boot_p = [spearmanr(x[i], y[i]).statistic for i in draws]
        boot_w = [tercile_spearman(x[i], y[i], length[i]) for i in draws]
        ci_p, ci_w = (np.nanpercentile(b, [2.5, 97.5]).tolist() for b in (boot_p, boot_w))
        q["quintile"] = pd.qcut(q["mean_nll"], 5, labels=False, duplicates="drop") + 1
        table = q.groupby("quintile").agg(problems=("detection", "size"), nll_max=("mean_nll", "max"),
                                          detection=("detection", "mean"))
        result.update(spearman=primary, spearman_ci=ci_p, within_length=within, within_length_ci=ci_w,
                      quintiles=table.reset_index().to_dict("records"),
                      reading="positive" if ci_p[0] > 0 else "negative" if ci_p[1] < 0 else "spans zero")
        lines += ["", "| measure | value | 95% interval |", "| --- | --- | --- |",
                  f"| Spearman(mean NLL, detection rate) | **{primary:.3f}** | [{ci_p[0]:.3f}, {ci_p[1]:.3f}] |",
                  f"| the same within length terciles, averaged | {within:.3f} | [{ci_w[0]:.3f}, {ci_w[1]:.3f}] |",
                  "", "| NLL quintile (1 = most familiar) | problems | max NLL | mean detection |", "| --- | --- | --- | --- |"]
        lines += [f"| {int(k)} | {int(r.problems)} | {r.nll_max:.3f} | {r.detection:.3f} |" for k, r in table.iterrows()]
        lines += ["", f"**Reading (§7):** the primary interval is {result['reading']}; the memorisation mechanism "
                  "predicts positive (less familiar, more detection)."]

    body = "\n".join(lines)
    print(body)
    out = ROOT / "runs" / "e62"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    frame.to_csv(out / "problems.csv", index=False)
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", HEADING, body)
        print(f"\nwrote the {HEADING} section of RESULTS.md")


if __name__ == "__main__":
    main()
