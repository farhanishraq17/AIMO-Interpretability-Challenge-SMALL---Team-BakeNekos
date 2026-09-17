"""E12: how well does each label-free signal separate robust from spurious on DeepSeek?

No model is fit, so there is no CV: each signal's ROC-AUC is read directly against the
138 DeepSeek labels, with a paired bootstrap CI against chance (a constant score). A signal
with AUC below 0.5 is still useful with its sign flipped; the interval says whether either
direction is real. Only signals that clear chance are candidates for the models without
labels.

Two checks follow, because a signal can predict DeepSeek's labels without reading anything
model-specific:

* **Length control.** Each signal's AUC, counting only pairs within the same problem-length
  quintile. Length is the strongest single surface feature. On synthetic data this cuts a
  pure length proxy from about 0.8 to 0.56 while leaving a real signal intact. Regressing
  ranks on length instead fails for heavily tied signals such as agreement fractions, whose
  residual becomes length itself.
* **Cross-model transfer.** The same signals computed by the unlabelled models
  (`e13-*-plumbing`) and scored against DeepSeek's labels. A signal that works equally well
  from another model is a property of the problem, not of DeepSeek.

Run:  python train/eval_label_free.py [--records "runs/e12-labelfree-deepseek/**/records.jsonl"]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from results_io import paired_auc_difference, upsert_section

ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
OTHER_MODELS = ROOT / "runs" / "e13-*-plumbing" / "features" / "labelfree-*" / "*" / "records.jsonl"
TRANSFER_SIGNALS = ["problem_nll", "next_token_entropy", "next_token_logit_margin", "answer_entropy"]
SIGNALS = [
    "variant_agree", "variant_kl", "answer_entropy", "answer_max_prob", "problem_nll",
    "next_token_entropy", "next_token_max_prob", "next_token_logit_margin",
]


def within_length_auc(y: np.ndarray, s: np.ndarray, length: np.ndarray, n_bins: int = 5) -> float:
    """Mann-Whitney AUC over robust/spurious pairs that share a length quintile (ties count half)."""
    bins = np.minimum((rankdata(length, method="ordinal") - 1) * n_bins // len(length), n_bins - 1)
    wins = pairs = 0.0
    for b in range(n_bins):
        pos, neg = s[(bins == b) & y], s[(bins == b) & ~y]
        wins += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        pairs += len(pos) * len(neg)
    return wins / pairs if pairs else float("nan")


def load(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True, dtype={"immediate_answer": str})


def load_deepseek(pattern: str) -> pd.DataFrame:
    """DeepSeek-scored, DeepSeek-labelled records, one per problem, with the problem text and length."""
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        raise SystemExit(f"no records match {pattern}")
    df = pd.concat([load(p) for p in paths]).drop_duplicates("problem_key")
    df = df[(df.scored_model_id == DEEPSEEK) & (df.label_model_id == DEEPSEEK)]
    problems = pd.read_parquet(INSTANCES, columns=["model_id", "problem_key", "problem"])
    problems = problems[problems.model_id == DEEPSEEK].drop(columns="model_id")
    df = df.merge(problems, on="problem_key", how="left").reset_index(drop=True)
    df["problem_length"] = df.problem.str.len()
    return df


def signal_table(df: pd.DataFrame, signals: list[str]) -> list[str]:
    """Markdown rows: each signal's AUC against the labels, its CI against chance, and after length control."""
    y_all = df.is_robust.to_numpy(bool)
    lines = [
        f"{len(df)} DeepSeek training problems ({y_all.sum()} robust). AUC > 0.5 means higher values "
        "indicate robust. The CI is a paired bootstrap against a constant score (AUC 0.5); rows where "
        "a signal is undefined are dropped for that signal.",
        "",
        "| Signal | n | ROC-AUC | 95% CI vs chance | AUC within length quintiles |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in [*signals, "problem_length"]:
        keep = df[name].notna().to_numpy()
        y, s = y_all[keep], df.loc[keep, name].to_numpy(float)
        if len(set(y)) < 2:
            continue
        diff, low, high = paired_auc_difference(y, s, np.zeros_like(s))
        controlled = (
            "—" if name == "problem_length"
            else f"{within_length_auc(y, s, df.loc[keep, 'problem_length'].to_numpy(float)):.4f}"
        )
        lines.append(
            f"| {name} | {keep.sum()} | {0.5 + diff:.4f} | [{0.5 + low:.4f}, {0.5 + high:.4f}] | {controlled} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(ROOT / "runs" / "e12-labelfree-deepseek" / "**" / "records.jsonl"))
    parser.add_argument("--results", type=Path, default=ROOT / "RESULTS.md")
    args = parser.parse_args()

    df = load_deepseek(args.records)
    lines = signal_table(df, SIGNALS)
    agree = df.variant_agree.dropna()
    lines += [
        "",
        f"Immediate answers stable across all rewrites: {(agree == 1).mean():.1%} of {len(agree)} problems. "
        f"Spearman correlation with problem length: "
        + ", ".join(f"{n} {df[n].corr(df.problem_length, method='spearman'):+.2f}" for n in TRANSFER_SIGNALS)
        + ".",
    ]

    others = sorted(glob.glob(str(OTHER_MODELS)))
    if others:
        lines += [
            "",
            "### Cross-model transfer: other models' signals against DeepSeek's labels",
            "",
            "| Scored by | n | " + " | ".join(TRANSFER_SIGNALS) + " | Spearman with DeepSeek's next_token_entropy |",
            "| --- | --- |" + " --- |" * (len(TRANSFER_SIGNALS) + 1),
        ]
        for path in others:
            other = load(path)
            joined = df[["problem_key", "is_robust", "next_token_entropy"]].merge(
                other, on="problem_key", suffixes=("_deepseek", "")
            )
            y = joined.is_robust_deepseek.to_numpy(bool)
            aucs = " | ".join(f"{roc_auc_score(y, joined[n]):.4f}" for n in TRANSFER_SIGNALS)
            rho = joined.next_token_entropy.corr(joined.next_token_entropy_deepseek, method="spearman")
            lines.append(f"| `{other.scored_model_id.iloc[0]}` | {len(joined)} | {aucs} | {rho:+.2f} |")
        lines += ["", "Labels are DeepSeek's throughout; no other model has labels."]
    body = "\n".join(lines)
    print(body)
    upsert_section(args.results, "E12 label-free signals (DeepSeek)", body)


if __name__ == "__main__":
    main()
