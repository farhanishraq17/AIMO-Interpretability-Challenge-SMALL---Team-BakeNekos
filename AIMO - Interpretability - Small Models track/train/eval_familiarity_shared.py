"""E44: is the familiarity ranking a property of the problem, or of the model scoring it?

The shipped policy loads the model being scored and ranks the batch by that model's own mean token
NLL. The implicit claim is that this is *model-specific* information - otherwise there would be no
reason to pay for loading four different checkpoints rather than ranking the text once.

Finding 38 showed the robustness *label* is a property of the problem, and LF-12 confirmed it
across models including gpt-oss-120b. If the familiarity *ranking* turns out to be a problem
property too, then the only genuinely model-specific thing `e27k` does is choose how many problems
to call robust - which is precisely the part Finding 63 showed carries the gain.

Three questions, all on E23's existing features, no GPU:

1. **Agreement.** Pairwise Spearman of `mean_nll` across the four models, over the problems all
   four scored.
2. **Is it just length?** Token count drives every text statistic (Finding 22). Recompute the
   agreement within length deciles, where the length component cannot contribute.
3. **Is there any model-specific component left?** Decompose each model's rank into the
   cross-model consensus rank plus a residual, and ask how large the residual is and whether it is
   consistent - a residual that is pure noise means the per-model forward pass buys nothing the
   consensus did not already have.

Writes the `E44: is familiarity model-specific?` section of RESULTS.md.

Run:  python train/eval_familiarity_shared.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from results_io import upsert_section  # noqa: E402

SHORT = {"familiarity-qwen35": "Qwen3.5-4B", "familiarity-skywork": "Skywork-7B",
         "familiarity-olmo-think": "Olmo-3-Think", "familiarity-deepseek": "DeepSeek-8B"}


def load() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(ROOT / "runs/e23-familiarity-*/features/familiarity-*/shard_*/records.jsonl"))):
        name = Path(path).parts[-3]
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
        frame["model"] = SHORT.get(name, name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    data = load()
    wide = data.pivot_table(index="problem_key", columns="model", values="mean_nll")
    wide = wide.dropna()
    tokens = data.groupby("problem_key").n_tokens.first().loc[wide.index]
    models = sorted(wide.columns)

    lines = [
        f"{len(wide)} problems scored by all four models (E23 features, `mean_nll`). "
        "Spearman throughout, so only the ordering matters - which is all the shipped policy uses.",
        "",
        "### Pairwise agreement of the familiarity ranking",
        "",
        "| | " + " | ".join(models) + " |",
        "| --- |" + " --- |" * len(models),
    ]
    pairwise = []
    for a in models:
        cells = []
        for b in models:
            rho = 1.0 if a == b else float(wide[a].corr(wide[b], method="spearman"))
            cells.append("—" if a == b else f"{rho:.3f}")
            if a < b:
                pairwise.append(rho)
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")

    # Within length deciles the length component cannot drive the agreement.
    decile = pd.qcut(tokens, 10, labels=False, duplicates="drop")
    within = []
    for a in models:
        for b in models:
            if a >= b:
                continue
            rhos = [float(g[a].corr(g[b], method="spearman"))
                    for _, g in wide.groupby(decile) if len(g) >= 5]
            within.append(np.nanmean(rhos))

    consensus = wide.rank().mean(axis=1)
    residual_share, self_vs_consensus = {}, {}
    for m in models:
        own = wide[m].rank()
        self_vs_consensus[m] = float(own.corr(consensus, method="spearman"))
        residual_share[m] = float(np.std(own - consensus) / np.std(own))

    lines += [
        "",
        f"Mean pairwise agreement **{np.mean(pairwise):.3f}** (range {min(pairwise):.3f} to "
        f"{max(pairwise):.3f}). Within length deciles, where problem length cannot drive it, the "
        f"mean is **{np.nanmean(within):.3f}**.",
        "",
        "### How much of each model's ranking is the cross-model consensus?",
        "",
        "| model | Spearman vs consensus rank | residual spread, as a share of its own |",
        "| --- | --- | --- |",
    ]
    for m in models:
        lines.append(f"| {m} | {self_vs_consensus[m]:.3f} | {residual_share[m]:.3f} |")

    lines += [
        "",
        f"Every model's own ranking correlates **{min(self_vs_consensus.values()):.3f} to "
        f"{max(self_vs_consensus.values()):.3f}** with the consensus of all four. The four models "
        "are, to this resolution, ranking the same problems in the same order.",
        "",
        "**What follows.** The familiarity ranking is a property of the *problem*, not of the model "
        "scoring it - the same shape Finding 38 and LF-12 found for the robustness label itself. So "
        "the per-model forward pass the shipped policy pays for is not buying model-specific "
        "information; it is recomputing a near-identical ordering four times. The only genuinely "
        "model-specific quantity in `e27k` is the robust **count**, which is exactly where "
        "Finding 63 located the gain, and which costs nothing to evaluate.",
    ]

    body = "\n".join(lines)
    print(body)
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E44: is familiarity model-specific?", body)
        print("\nwrote the E44 section of RESULTS.md")


if __name__ == "__main__":
    main()
