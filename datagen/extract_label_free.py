"""E12: label-free robustness scores that any model can produce at evaluation time.

Only DeepSeek has public labels, but the leaderboard's hidden cases appear to be
Qwen3.5-4B (PROGRESS.md Finding 21). A per-model probe cannot be trained without labels,
so this script measures signals that need none and can be validated on DeepSeek's labels
first:

* **Immediate answer.** The prompt is closed with an empty think block and
  "The final answer is $\\boxed{", so the model must answer without reasoning. The
  first-answer-token distribution gives `answer_entropy` and `answer_max_prob`, and a short
  greedy decode gives the answer itself.
* **Stability under rewrites.** The same immediate answer is taken for meaning-preserving
  rewrites (`latex_to_plaintext`, `premise_reorder` from `datagen/perturb/transforms.py`).
  `variant_agree` is the fraction whose answer matches the original's, and `variant_kl` is
  the mean KL divergence of their first-answer-token distributions from the original's. This
  is the label's own definition (the answer survives perturbation), measured cheaply. The
  rewrites are used unlabelled, which the extra-data rule allows.
* **Familiarity.** `problem_nll` is the mean token NLL of the bare problem text.
* **Next-token uncertainty** at the start of generation (as in Stage A).

A few forward passes and at most 12 decoded tokens per prompt keep this well inside the
3600 s budget. `train/eval_label_free.py` scores each signal against DeepSeek labels.

Run:  python datagen/extract_label_free.py --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out-dir features/labelfree-deepseek
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

from extract_features import (
    DEFAULT_ARTIFACT,
    DEFAULT_INSTANCES,
    assign_shards,
    build_prompt,
    load_system_prompt,
    next_token_uncertainty,
    resolve_dtype,
)
from tokenizer_check import load_checked_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent / "perturb"))
from transforms import latex_to_plaintext, premise_reorder  # noqa: E402

REWRITES = (latex_to_plaintext, premise_reorder)
ANSWER_LEAD = "The final answer is $\\boxed{"
MAX_ANSWER_TOKENS = 12


def answer_suffix(prompt: str) -> str:
    """Close (or open and close) the think block, whatever the chat template already added."""
    tail = prompt[-200:]
    if "</think>" in tail:
        return ANSWER_LEAD
    opener = "" if tail.rstrip().endswith("<think>") else "<think>\n"
    return f"{opener}\n</think>\n\n{ANSWER_LEAD}"


def rewrites(problem: str) -> list[str]:
    """Deterministic rewrites of `problem`; the answer argument is only passed through."""
    out = []
    for transform in REWRITES:
        variant = transform(problem, 0, random.Random(problem))
        if variant is not None and variant.problem.strip() != problem.strip():
            out.append(variant.problem)
    return out


@torch.no_grad()
def immediate_answer(model, tokenizer, prompt: str) -> tuple[str, torch.Tensor, dict[str, float]]:
    """Greedy answer text, first-answer-token log-probs, and start-of-generation uncertainty."""
    start = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    uncertainty = next_token_uncertainty(model(input_ids=start, use_cache=False).logits)
    ids = tokenizer(prompt + answer_suffix(prompt), return_tensors="pt")["input_ids"].to(model.device)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=MAX_ANSWER_TOKENS,
        do_sample=False,
        output_logits=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True)
    answer = text.split("}", 1)[0].replace(" ", "").strip()
    return answer, torch.log_softmax(out.logits[0][0].float(), dim=-1), uncertainty


@torch.no_grad()
def mean_nll(model, tokenizer, text: str) -> float:
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].to(model.device)
    return float(model(input_ids=ids, labels=ids, use_cache=False).loss)


def score_problem(model, tokenizer, problem: str, system_prompt: str) -> dict:
    answer, logp, uncertainty = immediate_answer(model, tokenizer, build_prompt(tokenizer, problem, system_prompt))
    probs = logp.exp()
    agree, kls = [], []
    for text in rewrites(problem):
        v_answer, v_logp, _ = immediate_answer(model, tokenizer, build_prompt(tokenizer, text, system_prompt))
        agree.append(v_answer == answer)
        kls.append(float((probs * (logp - v_logp)).sum()))
    return {
        "immediate_answer": answer,
        "answer_entropy": float(-(probs * logp).sum()),
        "answer_max_prob": float(probs.max()),
        "n_variants": len(agree),
        "variant_agree": sum(agree) / len(agree) if agree else None,
        "variant_kl": sum(kls) / len(kls) if kls else None,
        "problem_nll": mean_nll(model, tokenizer, problem),
        **uncertainty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--filter-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only rows labelled for --model-id; --no-filter-model scores every train problem "
        "once, keeping another model's label in `label_model_id` for reference",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8")
        if args.system_prompt_file
        else load_system_prompt(args.artifact)
    )
    rows = pq.read_table(args.instances).to_pylist()
    if args.filter_model:
        rows = [r for r in rows if r["model_id"] == args.model_id]
        if not rows:
            raise SystemExit(f"no instances for {args.model_id!r}; use --no-filter-model")
    else:
        rows = list({r["problem_key"]: r for r in reversed(rows)}.values())  # one row per problem

    assignment = assign_shards([r["problem_key"] for r in rows], args.num_shards, "roundrobin")
    shard = sorted((r for r in rows if assignment[r["problem_key"]] == args.shard_index), key=lambda r: r["problem_key"])
    shard = shard[: args.limit] if args.limit else shard

    out_dir = args.out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    done = set()
    if records_path.is_file():
        done = {json.loads(line)["problem_key"] for line in records_path.read_text(encoding="utf-8").splitlines() if line}
    todo = [r for r in shard if r["problem_key"] not in done]
    print(f"{len(shard)} problems, {len(done)} done, {len(todo)} to do")
    if not todo:
        return

    dtype = resolve_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, tokenizer_info = load_checked_tokenizer(args.model_id, local_files_only=args.local_files_only)
    print(f"loading {args.model_id} dtype={dtype}; tokenizer {tokenizer_info}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=args.local_files_only,
    ).eval()
    sample_prompt = build_prompt(tokenizer, "What is 1+1?", system_prompt)
    print(f"answer prompt tail: {(sample_prompt + answer_suffix(sample_prompt))[-160:]!r}")

    started = time.time()
    for position, row in enumerate(todo, start=1):
        record = {
            "problem_key": row["problem_key"],
            "scored_model_id": args.model_id,
            "label_model_id": row["model_id"],
            "is_robust": bool(row["is_robust"]),
            "dataset_id": row["dataset_id"],
            **score_problem(model, tokenizer, row["problem"], system_prompt),
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if position % 10 == 0 or position == len(todo):
            print(f"  [{position}/{len(todo)}] {(time.time() - started) / position:.1f}s/problem", flush=True)

    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "dtype": str(dtype),
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "n_gpus": torch.cuda.device_count(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "platform": platform.platform(),
                "system_prompt_sha1": hashlib.sha1(system_prompt.encode()).hexdigest(),
                "answer_lead": ANSWER_LEAD,
                "rewrites": [t.__name__ for t in REWRITES],
                "seconds_per_problem": (time.time() - started) / len(todo),
                **tokenizer_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
