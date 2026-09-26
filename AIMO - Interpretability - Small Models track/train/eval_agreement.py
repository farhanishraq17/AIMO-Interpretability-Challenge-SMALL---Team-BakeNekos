"""E14: does the model's answer survive rewrites, and does that predict DeepSeek's labels?

Reads the `datagen/extract_agreement.py` records and reports, per signal, the ROC-AUC against
the 138 DeepSeek labels, a paired bootstrap CI against chance, and the AUC after problem
length is controlled (the same table as E12). It also reports the cost of generating these
signals, which decides whether they could run inside the 3600 s evaluation budget.
Whether the signals add to text and hidden states is `train_fusion.py --blocks ... agreement`.

Run:  python train/eval_agreement.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from eval_label_free import load_deepseek, signal_table
from results_io import upsert_section

ROOT = Path(__file__).resolve().parents[1]
RUN_GLOB = str(ROOT / "runs" / "e14-agreement-deepseek-s*" / "features" / "agreement-deepseek" / "*")
KINDS = ["latex_to_plaintext", "premise_reorder", "typos", "distract"]
SIGNALS = [
    "rewrite_agree", "rewrite_min_agree", "orig_self_agree", "orig_answered",
    "orig_tokens", "rewrite_token_ratio", "hit_length_frac", *[f"agree_{k}" for k in KINDS],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-glob", default=RUN_GLOB, help="shard directories holding records.jsonl + metadata.json")
    parser.add_argument("--results", type=Path, default=ROOT / "RESULTS.md")
    args = parser.parse_args()

    df = load_deepseek(str(Path(args.run_glob) / "records.jsonl"))
    for kind in KINDS:  # a kind that never applied has no column
        df[f"agree_{kind}"] = df.get(f"agree_{kind}")
    lines = signal_table(df, SIGNALS)

    shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in glob.glob(str(Path(args.run_glob) / "metadata.json"))]
    tokens = sum(m["generated_tokens"] for m in shards)
    seconds = sum(m["seconds"] for m in shards)
    k = shards[0]["args"]["k"]
    n_prompts = len(df) + int(df.n_rewrites.sum())
    agree = df.rewrite_agree.dropna()
    lines += [
        "",
        f"Setup: k={k} samples per prompt, {shards[0]['args']['max_new_tokens']}-token cap, "
        f"{shards[0]['backend']} {shards[0]['backend_version']} in {shards[0]['args']['dtype']} on T4s. "
        f"{n_prompts} prompts ({n_prompts - len(df)} rewrites: "
        + ", ".join(f"{kind} {int(df[f'agree_{kind}'].notna().sum())}" for kind in KINDS)
        + ").",
        f"Anchor answer found for {int(df.rewrite_agree.notna().sum())} of {len(df)} problems; "
        f"every rewrite sample matched it for {(agree == 1).mean():.1%} of them.",
        f"Cost: {tokens / 1e6:.2f}M generated tokens, {tokens / len(df):,.0f} per problem, "
        f"{seconds / 3600:.1f} GPU-session hours ({tokens / seconds:.0f} tok/s on 2x T4).",
    ]
    body = "\n".join(lines)
    print(body)
    upsert_section(args.results, "E14: test-time answer agreement (DeepSeek)", body)


if __name__ == "__main__":
    main()
