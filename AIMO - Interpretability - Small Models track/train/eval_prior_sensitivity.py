"""E43: how far can the private test set's class prior drift before `e27k` becomes the wrong ship?

`e27k` differs from `e27e` in exactly one thing: Skywork and Olmo call one fewer problem robust,
because the constant probes measured both as spurious-leaning on the validation set. That is worth
two cases there (12/14 against 10/14). The prizes, however, are scored on a private test set of 30
unseen problems whose class prior nobody has measured, and PAPER section 9 states the risk in
words without a number. This puts a number on it.

The arithmetic is short enough to state. With a batch of `n` cases of which `r` are robust, a
policy that calls `c` of them robust using a ranking that carries no information scores, in
expectation,

    E[correct] = c * (2r/n - 1) + n - r

which is linear in `c` with slope `2r/n - 1`. So raising the robust count helps exactly when the
batch is majority robust and hurts otherwise, and the crossover is at **r/n = 0.5** regardless of
`n`. `e27k` lowers `c` by one for the two tilted models, so:

* `r/n < 0.5` for Skywork and Olmo  ->  `e27k` wins, by `1 - 2r/n` cases per tilted model;
* `r/n > 0.5`                        ->  `e27e` wins, by the same amount;
* `r/n = 0.5`                        ->  they tie.

The tilts for Qwen and DeepSeek are zero, so those two models contribute nothing to the difference
whatever their prior does. The whole bet rides on two models.

This matters because the two sources disagree about those two models. The validation probes make
both spurious-leaning (balance -1, so r/n < 0.5). The organizers' own labelled split makes both
**majority robust**: Skywork 2 of 3, Olmo 7 of 9. If the private test set resembles the official
split rather than the validation set, the tilts are backwards.

Writes the `E43: prior sensitivity of the tilts` section of RESULTS.md.

Run:  python train/eval_prior_sensitivity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from results_io import upsert_section  # noqa: E402

OFFICIAL = ROOT / "data" / "official_check" / "official_labels.parquet"
TILTED = ("Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think")
# Measured by the constant probes on the hidden validation set, twice: 18 Sep and 30 Sep
# (Findings 45, 64). Balance d = robust - spurious, so d < 0 is spurious-leaning.
MEASURED_BALANCE = {"Qwen/Qwen3.5-4B": +1, "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": +1,
                    "Skywork/Skywork-OR1-Math-7B": -1, "allenai/Olmo-3-7B-Think": -1}


def advantage(robust_share: float) -> float:
    """Cases `e27k` gains over `e27e` per tilted model, under a ranking with no information.

    Positive favours `e27k`. This is `-(2r/n - 1)`: the slope of E[correct] in the robust count,
    negated because `e27k` lowers that count by one.
    """
    return 1.0 - 2.0 * robust_share


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    labels = pd.read_parquet(OFFICIAL)
    official = {m: (int(g.is_robust.sum()), len(g)) for m, g in labels.groupby("model_id")}

    lines = [
        "`e27k` is `e27e` with Skywork and Olmo calling one fewer problem robust. Under a ranking "
        "carrying no information, `E[correct] = c(2r/n - 1) + n - r` is linear in the robust count "
        "`c` with slope `2r/n - 1`, so lowering `c` by one gains `1 - 2r/n` cases. The crossover is "
        "at a robust share of **0.5**, independently of batch size, and only the two tilted models "
        "contribute: Qwen and DeepSeek have zero tilt whatever their prior does.",
        "",
        "| robust share of a tilted model | cases `e27k` gains per tilted model | ship |",
        "| --- | --- | --- |",
    ]
    for share in (0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.667, 0.778):
        gain = advantage(share)
        verdict = "`e27k`" if gain > 1e-9 else ("tie" if abs(gain) <= 1e-9 else "**`e27e`**")
        lines.append(f"| {share:.3f} | {gain:+.3f} | {verdict} |")

    lines += [
        "",
        "**The two sources disagree about exactly the two models the bet rides on.**",
        "",
        "| model | validation probes (Findings 45, 64) | official split `train-main-v2` | agree? |",
        "| --- | --- | --- | --- |",
    ]
    for model in TILTED:
        robust, n = official[model]
        share = robust / n
        probe = "spurious-leaning (balance -1, share < 0.5)"
        lines.append(f"| {model.split('/')[-1]} | {probe} | **{robust} of {n} robust, share "
                     f"{share:.3f}** | **no** |")
    for model, balance in MEASURED_BALANCE.items():
        if model in TILTED:
            continue
        robust, n = official[model]
        lines.append(f"| {model.split('/')[-1]} | robust-leaning (balance +1) | {robust} of {n} "
                     f"robust, share {robust / n:.3f} | yes |")

    worst = [advantage(official[m][0] / official[m][1]) for m in TILTED]
    lines += [
        "",
        f"If the private test set's per-model shares matched the official split, the tilts would "
        f"cost **{-sum(worst):.2f} cases** rather than gain any, under a null ranking. If they "
        "match the validation set, the tilts gain and `e27k` is right. Nobody has measured the "
        "private set, and the organizers state only that the validation set covers all model types "
        "in the test set.",
        "",
        "**What this does and does not change.** It does not touch the measurement: the balances "
        "were reproduced exactly twelve days apart (Finding 64), so `e27k` is the correct policy "
        "*for the validation distribution*. What it shows is that the two-case gain is a bet on "
        "two models' priors holding, that the only other labelled source we have points the other "
        "way for both, and that the bet is symmetric - the same magnitude is lost if the prior "
        "flips. `e27e` is the prior-free alternative and gives up two known cases to buy that.",
    ]

    body = "\n".join(lines)
    print(body)
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E43: prior sensitivity of the tilts", body)
        print("\nwrote the E43 section of RESULTS.md")


if __name__ == "__main__":
    main()
