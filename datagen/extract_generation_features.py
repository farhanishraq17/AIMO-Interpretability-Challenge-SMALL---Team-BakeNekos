"""Stage B: generation-trace uncertainty features, shardable across the Kaggle fleet.

Stage A (`extract_features.py`) only reads the prompt. Stage B lets the model answer: one
greedy generation per problem, summarised into the uncertainty-profiling baseline's 14
trace features. That baseline is the only published method meaningfully above chance
(69.2%), so this reproduces its features on *our* consolidated instance table instead of
the 141-row aggregate it was fit on, keeping labels and grouped folds aligned with every
other experiment.

Generation goes through the starter kit's `uncertainty_profile` package unchanged (same
user-only chat template, greedy decoding, static cache, 2048/4096 token caps, batch size
8), so the features match what the baseline computes at evaluation time. We additionally
keep the generated text for answer extraction and trace-level features later.

Per-instance cost is minutes on a T4, so this is the stage the six-account fleet is for.
Shards come from `extract_features.assign_shards`, and rows are appended to
`records.jsonl` after every batch, so a session killed at Kaggle's 12-hour cap resumes
where it stopped.

Example, shard 0 of 2:

    python datagen/extract_generation_features.py \
        --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
        --num-shards 2 --shard-index 0 \
        --out-dir features/stageb-deepseek
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import transformers

DATAGEN = Path(__file__).resolve().parent
ROOT = DATAGEN.parent
UNCERTAINTY_SOLUTION = ROOT / "getting-started" / "solutions" / "uncertainty-profiling"
for import_path in (DATAGEN, UNCERTAINTY_SOLUTION):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from extract_features import DEFAULT_INSTANCES, assign_shards  # noqa: E402
from uncertainty_profile.config import FEATURE_NAMES, GenerationConfidenceConfig  # noqa: E402
from uncertainty_profile.extraction import (  # noqa: E402
    format_problem,
    get_model_input_device,
    iter_generation_feature_batches,
    load_model_and_tokenizer,
)


def select_shard(args: argparse.Namespace) -> list[dict]:
    """Rows this shard owns, with the same filtering rules as Stage A."""
    rows = pq.read_table(args.instances).to_pylist()
    if args.filter_model:
        rows = [r for r in rows if r["model_id"] == args.model_id]
        if not rows:
            available = sorted({r["model_id"] for r in pq.read_table(args.instances).to_pylist()})
            raise SystemExit(f"no instances for model_id {args.model_id!r}. Available: {available}")
    else:
        # Records are keyed by problem, so an unfiltered run must not repeat problems.
        seen: set[str] = set()
        rows = [r for r in rows if not (r["problem_key"] in seen or seen.add(r["problem_key"]))]

    assignment = assign_shards([r["problem_key"] for r in rows], args.num_shards, args.shard_strategy)
    shard = sorted(
        (r for r in rows if assignment[r["problem_key"]] == args.shard_index),
        key=lambda r: r["problem_key"],
    )
    return shard[: args.limit] if args.limit else shard


@torch.inference_mode()
def logits_are_finite(model, tokenizer, problem: str, config: GenerationConfidenceConfig) -> bool:
    """One forward pass before committing hours of generation.

    If a host falls back to fp16 (no bf16, not even emulated), Qwen-family checkpoints can
    overflow, and greedy decoding over NaN logits produces garbage traces rather than an
    error. Check once up front.
    """
    prompt = format_problem(tokenizer, problem, role=config.chat_template_role)
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=config.max_prompt_length
    )
    device = get_model_input_device(model)
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    logits = model(**encoded, use_cache=False).logits
    return bool(torch.isfinite(logits[0, -1]).all().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="HF model id to run")
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-strategy", default="roundrobin", choices=["roundrobin", "hash"])
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="debug: stop after N instances")
    parser.add_argument("--local-files-only", action="store_true", help="forbid model downloads")
    parser.add_argument(
        "--allow-nonfinite",
        action="store_true",
        help="generate even if the preflight forward pass produced inf/NaN logits",
    )
    parser.add_argument(
        "--filter-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only instances whose model_id equals --model-id (default); see extract_features.py",
    )
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < --num-shards")

    config = GenerationConfidenceConfig(
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    config.validate()

    shard = select_shard(args)
    out_dir = args.out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    done: set[str] = set()
    if records_path.is_file():
        done = {
            json.loads(line)["problem_key"]
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    todo = [r for r in shard if r["problem_key"] not in done]
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(shard)} instances, "
        f"{len(shard) - len(todo)} already done, {len(todo)} to do",
        flush=True,
    )
    if not todo:
        print("nothing to do")
        return

    started = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model_id, local_files_only=args.local_files_only)
    load_seconds = time.time() - started
    finite = logits_are_finite(model, tokenizer, todo[0]["problem"], config)

    metadata = {
        "stage": "B",
        "model_id": args.model_id,
        "dtype": str(model.dtype),
        "devices": sorted({str(d) for d in (getattr(model, "hf_device_map", None) or {}).values()}),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "generation_config": config.to_dict(),
        "feature_names": list(FEATURE_NAMES),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_strategy": args.shard_strategy,
        "n_instances": len(shard),
        "preflight_logits_finite": finite,
        "load_seconds": round(load_seconds, 1),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"loaded {args.model_id} dtype={model.dtype} in {load_seconds:.0f}s, preflight finite={finite}")
    if not finite and not args.allow_nonfinite:
        raise SystemExit(
            f"non-finite logits under {model.dtype}; rerun on a bf16-capable GPU "
            "or pass --allow-nonfinite to generate anyway"
        )

    generation_started = time.time()
    completed = 0
    for start, feature_rows in iter_generation_feature_batches(
        model=model,
        tokenizer=tokenizer,
        problem_texts=[r["problem"] for r in todo],
        config=config,
        include_generated_text=True,
    ):
        batch = todo[start : start + len(feature_rows)]
        # Flush per batch so a killed session loses at most one batch.
        with records_path.open("a", encoding="utf-8") as handle:
            for row, features in zip(batch, feature_rows):
                record = {
                    "problem_key": row["problem_key"],
                    "model_id": row["model_id"],
                    "extracted_model_id": args.model_id,
                    "is_robust": bool(row["is_robust"]),
                    "dataset_id": row["dataset_id"],
                    "hit_token_cap": int(features["generation_num_tokens"]) >= config.max_new_tokens,
                    **features,
                }
                handle.write(json.dumps(record) + "\n")
        completed += len(feature_rows)
        rate = (time.time() - generation_started) / completed
        print(
            f"  [{completed}/{len(todo)}] {rate:.0f}s/instance, "
            f"~{rate * (len(todo) - completed) / 60:.0f} min left",
            flush=True,
        )

    metadata["generation_seconds"] = round(time.time() - generation_started, 1)
    metadata["n_generated_this_session"] = completed
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\ndone in {(time.time() - started) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
