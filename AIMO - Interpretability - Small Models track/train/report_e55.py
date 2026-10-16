"""E55: coverage, provenance and one-pass timing of the feature runs, against the protocol's criteria.

`protocols/E55_PROTOCOL.md` asks for features on at least 95% of (model, problem) pairs with zero
non-finite activations and full provenance, plus a timing record. Each kernel loaded its model once
and scored problems one at a time, the evaluator's pattern since 15 Oct (Finding 79), so its records
are the one-pass timing. The projection uses the protocol's fixed rule: 60 one-problem calls across
the four models, on the evaluation card taken as one third of the measured T4 time, against 50 minutes.

Writes the `E55: one-pass features and per-problem timing` section of RESULTS.md.

Run:  python train/report_e55.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from results_io import upsert_section  # noqa: E402

SHORT = {"qwen": "Qwen/Qwen3.5-4B", "skywork": "Skywork/Skywork-OR1-Math-7B",
         "olmo": "allenai/Olmo-3-7B-Think", "deepseek": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
INSTANCES = ROOT / "data" / "e55" / "instances.parquet"
RUN = "runs/e55-features-{short}-*/e55-features-{short}"
CALLS, SPEEDUP, LIMIT_MIN = 60, 3, 50  # protocol section 5


def one(short: str) -> dict:
    roots = sorted(ROOT.glob(RUN.format(short=short)))
    assert len(roots) == 1, f"{short}: expected one downloaded run, found {roots}"
    root = roots[0]
    wanted = set(pd.read_parquet(INSTANCES).query("model_id == @SHORT[@short]")["problem_key"])
    fam_rows = [json.loads(line) for path in root.glob(f"familiarity-{short}/shard_*/records.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fam_meta = json.loads(next(root.glob(f"familiarity-{short}/shard_*/metadata.json")).read_text(encoding="utf-8"))
    stage = pd.read_parquet(root / f"stagea-{short}" / "records.parquet")
    stage_meta = json.loads(next(root.glob(f"stagea-{short}/shard_*/metadata.json")).read_text(encoding="utf-8"))
    stage_rows = [json.loads(line) for path in root.glob(f"stagea-{short}/shard_*/records.jsonl")
                  for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    execution = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    return {
        "model": SHORT[short],
        "familiarity": len({r["problem_key"] for r in fam_rows} & wanted) / len(wanted),
        "stagea": len(set(stage["problem_key"]) & wanted) / len(wanted),
        "nonfinite": sum(r.get("n_nonfinite_activations", 0) for r in stage_rows),
        "dtype": stage_meta["dtype"], "prompt_sha1": stage_meta["system_prompt_sha1"][:8],
        "seconds_per_problem": fam_meta["seconds_per_problem"],
        "steps": execution["steps"], "exit_codes": {k: v["exit_code"] for k, v in execution["steps"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    runs = {short: one(short) for short in SHORT}

    lines = ["| model | familiarity coverage | Stage A coverage | non-finite | dtype | prompt SHA-1 | "
             "seconds per problem (one pass, T4) | familiarity step, load included (s) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs.values():
        lines.append(f"| `{r['model']}` | {r['familiarity']:.3f} | {r['stagea']:.3f} | {r['nonfinite']} | {r['dtype']} | "
                     f"{r['prompt_sha1']} | {r['seconds_per_problem']:.2f} | {r['steps']['familiarity']['seconds']:.0f} |")
    ok = all(min(r["familiarity"], r["stagea"]) >= 0.95 and r["nonfinite"] == 0 and not any(r["exit_codes"].values())
             for r in runs.values())
    per_call = sum(r["seconds_per_problem"] for r in runs.values()) / len(runs)
    projected = CALLS * per_call / SPEEDUP / 60
    lines += ["", f"**Success criteria (protocol §4): {'met' if ok else 'NOT met'}.** One-pass cost: {CALLS} one-problem "
              f"calls at the mean T4 rate ({per_call:.2f} s) and one third of it on the evaluation card come to "
              f"**{projected:.1f} minutes** of compute against {LIMIT_MIN}, before model loading; the budget-path path "
              "was decided by E49's record (amendment 1: 496 minutes, E56 not run)."]
    body = "\n".join(lines)
    print(body)
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E55: one-pass features and per-problem timing", body)
        print("\nwrote the E55 section of RESULTS.md")


if __name__ == "__main__":
    main()
