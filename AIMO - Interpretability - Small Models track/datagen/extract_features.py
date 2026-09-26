"""Shard-parallel hidden-state and uncertainty feature extraction.

Designed to run one shard per Kaggle account so the six accounts cover the whole
instance table in parallel. Stage A only: a single forward pass over the prompt, no
generation, so a shard finishes well inside a Kaggle session.

Three deliberate departures from the shipped probe baseline:

* The baseline reloads the 8B model for *every* problem (`_encode_problem` calls
  `_load_model` each call). We load once and reuse, which is the difference between
  minutes and hours for a few hundred problems.
* The baseline keeps only the last prompt token at one layer, which measured at chance
  (PLAN.md Finding 2). We keep several poolings at every layer so the pooling and layer
  choice become training-time decisions rather than baked-in guesses.
* The baseline loads `AutoTokenizer`, which under transformers 5.13 strips all whitespace
  from DeepSeek-R1-0528-Qwen3-8B prompts. We load through `tokenizer_check`, which refuses
  a tokenizer that does not round-trip (PROGRESS.md Finding 14).

Prompt construction otherwise mirrors the baseline exactly (same system prompt, read from
the probe artifact; same chat template with `add_generation_prompt=True`) so our features
stay comparable to the published probe numbers.

Example, shard 0 of 6:

    python datagen/extract_features.py \
        --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
        --num-shards 6 --shard-index 0 \
        --out-dir features/deepseek-8b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

from tokenizer_check import load_checked_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
DEFAULT_ARTIFACT = (
    ROOT
    / "getting-started"
    / "solutions"
    / "trained-probe"
    / "probe_artifacts"
    / "probe_artifact.pkl"
)

POOLINGS = ("last", "mean", "max")


def load_system_prompt(artifact_path: Path) -> str:
    """Read the baseline's system prompt from the probe artifact.

    Kept in the artifact rather than hardcoded so it cannot drift out of sync with
    the baseline we are comparing against.
    """
    if not artifact_path.is_file():
        raise SystemExit(
            f"probe artifact not found at {artifact_path}; pass --artifact or --system-prompt-file"
        )
    with artifact_path.open("rb") as handle:
        data = pickle.load(handle)
    prompt = str(data.get("system_prompt") or "")
    if not prompt:
        raise SystemExit("probe artifact contains no system_prompt")
    return prompt


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        if not torch.cuda.is_available():
            return torch.float32
        # Mirrors the starter kit's loader. Kaggle's torch reports bf16 on the T4 (emulated
        # on Turing), so runs there match the evaluation dtype; GPUs without even emulated
        # bf16 fall back to fp16. Recorded in metadata either way.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def shard_of(key: str, num_shards: int) -> int:
    """Stable content-based shard assignment, independent of row order.

    Balance is poor for small sets (138 instances split 15/34 across 6 shards), so
    prefer `assign_shards` for real runs; this remains for per-key lookups.
    """
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % num_shards


def assign_shards(keys: list[str], num_shards: int, strategy: str) -> dict[str, int]:
    """Map each problem key to a shard.

    `roundrobin` deals keys out over a hash-sorted order: deterministic, independent of
    input row order, and balanced to within one instance per shard. `hash` keeps a key's
    shard fixed even as the instance set grows, at the cost of uneven shard sizes.

    Either way, resume is keyed on the per-problem output file, so a reassignment only
    changes which account does the work, never the result.
    """
    if strategy == "hash":
        return {key: shard_of(key, num_shards) for key in keys}
    ordered = sorted(keys, key=lambda k: hashlib.sha1(k.encode("utf-8")).hexdigest())
    return {key: index % num_shards for index, key in enumerate(ordered)}


def build_prompt(tokenizer, problem: str, system_prompt: str) -> str:
    """Format a problem exactly as the probe baseline does."""
    if system_prompt and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{system_prompt}\n\n{problem}" if system_prompt else problem


def pool_hidden_states(hidden_states, attention_mask) -> tuple[dict[str, np.ndarray], int]:
    """Pool every layer's prompt tokens into fixed-size vectors.

    Returns one (n_layers, hidden_dim) float16 array per pooling strategy, plus the
    number of non-finite activations seen. Padding is masked out so `mean`/`max` are not
    polluted by pad positions.
    """
    mask = attention_mask[0].bool()
    out: dict[str, list[np.ndarray]] = {name: [] for name in POOLINGS}
    n_nonfinite = 0
    for layer in hidden_states:
        # With device_map="auto" on a multi-GPU host (Kaggle T4x2) layers live on
        # different devices, so the mask has to follow each layer.
        tokens = layer[0][mask.to(layer.device)].float()  # (n_real_tokens, hidden_dim)
        # An fp16 fallback can overflow on Qwen-family models; count rather than hide it.
        n_nonfinite += int((~torch.isfinite(tokens)).sum().item())
        out["last"].append(tokens[-1].cpu().numpy())
        out["mean"].append(tokens.mean(dim=0).cpu().numpy())
        out["max"].append(tokens.max(dim=0).values.cpu().numpy())
    return {name: np.stack(vals).astype(np.float16) for name, vals in out.items()}, n_nonfinite


def next_token_uncertainty(logits) -> dict[str, float]:
    """Cheap uncertainty signals from the next-token distribution.

    The uncertainty-profiling baseline is the only one meaningfully above chance, but
    it pays for full generations. These are the same family of signals at the cost of
    the forward pass we are already doing.
    """
    last = logits[0, -1, :].float()
    probs = torch.softmax(last, dim=-1)
    log_probs = torch.log_softmax(last, dim=-1)
    entropy = float(-(probs * log_probs).sum().item())
    top = torch.topk(probs, k=5)
    top_probs = top.values.tolist()
    sorted_logits = torch.sort(last, descending=True).values
    return {
        "next_token_entropy": entropy,
        "next_token_max_prob": float(top_probs[0]),
        "next_token_top5_prob_mass": float(sum(top_probs)),
        "next_token_logit_margin": float((sorted_logits[0] - sorted_logits[1]).item()),
        "next_token_perplexity": float(np.exp(min(entropy, 20.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="HF model id to run")
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--shard-strategy",
        default="roundrobin",
        choices=["roundrobin", "hash"],
        help="roundrobin balances shard sizes; hash pins a problem to a fixed shard",
    )
    parser.add_argument("--max-length", type=int, default=2048, help="matches the baseline's max_prompt_length")
    parser.add_argument("--limit", type=int, default=None, help="debug: stop after N instances")
    parser.add_argument(
        "--allow-lossy-tokenizer",
        action="store_true",
        help="debug only: run even if the tokenizer does not round-trip text",
    )
    parser.add_argument(
        "--filter-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep only instances whose model_id equals --model-id (default). "
            "`is_robust` is model-specific, so mixing models would mislabel features; "
            "use --no-filter-model only for pipeline smoke tests."
        ),
    )
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < --num-shards")

    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8")
        if args.system_prompt_file
        else load_system_prompt(args.artifact)
    )

    rows = pq.read_table(args.instances).to_pylist()
    if args.filter_model:
        rows = [r for r in rows if r["model_id"] == args.model_id]
        if not rows:
            available = sorted({r["model_id"] for r in pq.read_table(args.instances).to_pylist()})
            raise SystemExit(
                f"no instances for model_id {args.model_id!r}. Available: {available}"
            )
    else:
        # One vector file per problem_key, so unfiltered runs would collide across models.
        seen: set[str] = set()
        deduped = []
        for row in rows:
            if row["problem_key"] not in seen:
                seen.add(row["problem_key"])
                deduped.append(row)
        print(f"--no-filter-model: deduplicated {len(rows)} -> {len(deduped)} by problem")
        rows = deduped

    assignment = assign_shards(
        [r["problem_key"] for r in rows], args.num_shards, args.shard_strategy
    )
    shard = [r for r in rows if assignment[r["problem_key"]] == args.shard_index]
    shard.sort(key=lambda r: r["problem_key"])
    if args.limit:
        shard = shard[: args.limit]

    out_dir = args.out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: one file per instance, so an interrupted Kaggle session
    # simply continues where it stopped.
    done = {p.stem for p in vectors_dir.glob("*.npz")}
    todo = [r for r in shard if r["problem_key"] not in done]
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(shard)} instances, "
        f"{len(done)} already done, {len(todo)} to do"
    )
    if not todo:
        print("nothing to do")
        return

    dtype = resolve_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model_id} dtype={dtype} device={device}")

    tokenizer, tokenizer_info = load_checked_tokenizer(args.model_id, allow_lossy=args.allow_lossy_tokenizer)
    print(f"tokenizer: {tokenizer_info}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    records: list[dict] = []
    started = time.time()
    for position, row in enumerate(todo, start=1):
        prompt = build_prompt(tokenizer, row["problem"], system_prompt)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        pooled, n_nonfinite = pool_hidden_states(output.hidden_states, inputs["attention_mask"])
        uncertainty = next_token_uncertainty(output.logits)
        n_tokens = int(inputs["attention_mask"][0].sum().item())
        if n_nonfinite:
            print(f"  warning: {n_nonfinite} non-finite activations for {row['problem_key']} ({dtype})")

        np.savez_compressed(
            vectors_dir / f"{row['problem_key']}.npz",
            **pooled,
            n_prompt_tokens=np.asarray(n_tokens),
        )
        records.append(
            {
                "problem_key": row["problem_key"],
                "model_id": row["model_id"],
                "extracted_model_id": args.model_id,
                "is_robust": bool(row["is_robust"]),
                "dataset_id": row["dataset_id"],
                "n_prompt_tokens": n_tokens,
                "truncated": n_tokens >= args.max_length,
                "n_nonfinite_activations": n_nonfinite,
                **uncertainty,
            }
        )

        # Flush after every instance so a killed session loses nothing.
        with (out_dir / "records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(records[-1]) + "\n")

        if position % 5 == 0 or position == len(todo):
            rate = (time.time() - started) / position
            remaining = rate * (len(todo) - position)
            print(
                f"  [{position}/{len(todo)}] {rate:.1f}s/instance, "
                f"~{remaining / 60:.1f} min left",
                flush=True,
            )

        del output
        if device == "cuda":
            torch.cuda.empty_cache()

    sample = np.load(vectors_dir / f"{todo[0]['problem_key']}.npz")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "dtype": str(dtype),
                "device": device,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "poolings": list(POOLINGS),
                "n_layers": int(sample["last"].shape[0]),
                "hidden_dim": int(sample["last"].shape[1]),
                "max_length": args.max_length,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "shard_strategy": args.shard_strategy,
                "n_instances": len(shard),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "system_prompt_sha1": hashlib.sha1(system_prompt.encode()).hexdigest(),
                **tokenizer_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ndone in {(time.time() - started) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
