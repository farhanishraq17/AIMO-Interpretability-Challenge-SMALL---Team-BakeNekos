"""E7: sample the target model on olympiad originals and their perturbed variants.

Labels for the olympiad data program (PLAN.md §3.4) come from the model's own accuracy, the
same way the organizers' published rows do: `base_accuracy` over k samples of the original,
`permuted_accuracy` over samples of each perturbation, and robust iff the decay is <= 0.

Two phases, so GPU time only goes to problems that can yield a label:

    base      k samples per pool problem. Only problems the model always solves
              (base_accuracy == 1.0, the organizers' eligibility rule) can be labelled.
    variants  k samples per variant of each eligible problem. By default only
              `compliance == "distinct"` categories, which are labelable without organizer
              confirmation (see transforms.py).

`label.py` turns both phases into rows in the `ingest_public.py` schema.

Backends:

    vllm   Kaggle runs. Paged attention and continuous batching matter here, and only final
           answers are needed, not token-level statistics.
    hf     transformers `generate`, for local smoke tests on tiny models.

**Tokenization never goes through vLLM's own tokenizer.** Under transformers 5.13 it
resolves DeepSeek-R1-0528-Qwen3-8B to a legacy conversion that drops whitespace (PROGRESS.md
Finding 14). Prompts are encoded with the round-trip-checked tokenizer from
`tokenizer_check`, passed to vLLM as token ids, and completions are decoded with the same
tokenizer.

Output: <out-dir>/<phase>/shard_XX_of_YY/records.jsonl, one line per item, flushed per chunk
and resumable; metadata.json records sampling settings, tokenizer, versions, and throughput.

Example (Kaggle):

    python datagen/perturb/sample.py --phase base \\
        --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --backend vllm \\
        --out-dir /kaggle/working/e7
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

PERTURB = Path(__file__).resolve().parent
ROOT = PERTURB.parents[1]
for import_path in (PERTURB, PERTURB.parent):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from answers import extract_answer  # noqa: E402
from extract_features import assign_shards  # noqa: E402
from tokenizer_check import load_checked_tokenizer  # noqa: E402

DEFAULT_POOL = ROOT / "data" / "olympiad_pool" / "pool.parquet"
DEFAULT_VARIANTS = ROOT / "data" / "perturbations" / "variants.parquet"
# DeepSeek's recommended instruction for R1-family math prompts.
PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_records(pattern: str) -> dict[str, dict]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no records match {pattern}")
    records: dict[str, dict] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                records[row["item_id"]] = row
    return records


def build_items(args: argparse.Namespace) -> tuple[list[dict], dict]:
    pool = pd.read_parquet(args.pool)
    # Pools built before the pool.py fix reuse 34 AMC ids for two different problems; their records
    # could never be labelled, so do not sample them (PROGRESS.md Finding 29).
    pool = pool[~pool["pool_id"].duplicated(keep=False)]
    if args.tiers:
        pool = pool[pool["tier"].isin(args.tiers)]
    if getattr(args, "sources", None):
        pool = pool[pool["source"].isin(args.sources)]
    if getattr(args, "exclude_problem_keys", None):
        pool = pool[~pool["problem_key"].isin(set(args.exclude_problem_keys))]
    if args.problem_keys:
        # Re-sample a named set of problems, e.g. ones whose base records were lost with an
        # account. Keyed on `problem_key` rather than `pool_id` because the latter is not unique.
        wanted = set(args.problem_keys)
        pool = pool[pool["problem_key"].isin(wanted)]
        found = set(pool["problem_key"])
        if found != wanted:
            raise SystemExit(f"{len(wanted - found)} of {len(wanted)} --problem-keys are not in {args.pool}")
    info: dict = {"pool_problems": len(pool)}
    if args.phase == "base":
        items = [
            {
                "item_id": f"orig:{row.pool_id}",
                "pool_id": row.pool_id,
                "kind": "original",
                "category": None,
                "tier": row.tier,
                "problem": row.problem,
                "expected_answer": int(row.answer),
            }
            for row in pool.itertuples(index=False)
        ]
        return items, info

    if not args.base_records:
        raise SystemExit("--phase variants needs --base-records (glob of base-phase records.jsonl)")
    base = load_records(args.base_records)
    eligible = {
        r["pool_id"] for r in base.values()
        if r["n_samples"] and r["n_correct"] / r["n_samples"] >= args.min_base_accuracy
    }
    variants = pd.read_parquet(args.variants)
    info.update(base_items=len(base), eligible_problems=len(eligible), variants_total=len(variants))
    variants = variants[variants["pool_id"].isin(eligible) & variants["pool_id"].isin(set(pool["pool_id"]))]
    if not args.include_confirm:
        variants = variants[variants["compliance"] == "distinct"]
    if args.categories:
        variants = variants[variants["category"].isin(args.categories)]
    info["variants_selected"] = len(variants)
    # A stable fingerprint, so a kernel that regenerated the variants can prove it matches.
    info["variant_ids_sha1"] = hashlib.sha1("\n".join(sorted(variants["variant_id"])).encode()).hexdigest()
    items = [
        {
            "item_id": row.variant_id,
            "pool_id": row.pool_id,
            "kind": "variant",
            "category": row.category,
            "tier": row.tier,
            "problem": row.variant_problem,
            "expected_answer": int(row.expected_answer),
        }
        for row in variants.itertuples(index=False)
    ]
    return items, info


def encode_prompt(tokenizer, problem: str) -> list[int]:
    """Chat-templated prompt ids. The template already carries BOS, so add no special tokens."""
    message = [{"role": "user", "content": problem + PROMPT_SUFFIX}]
    try:
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    except (AttributeError, TypeError, ValueError):
        # Tokenizers without a chat template (smoke tests).
        return tokenizer(problem + PROMPT_SUFFIX)["input_ids"]


def stop_token_ids(tokenizer, model_id: str, local_files_only: bool) -> list[int]:
    """EOS ids from the tokenizer and the model's generation config (R1 models end on either)."""
    ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    try:
        from transformers import GenerationConfig

        eos = GenerationConfig.from_pretrained(model_id, local_files_only=local_files_only).eos_token_id
        ids |= set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
    except (OSError, ValueError):
        pass
    return sorted(ids)


class VllmBackend:
    name = "vllm"

    def __init__(self, args: argparse.Namespace, tokenizer):
        from vllm import LLM, SamplingParams, __version__

        self.version = __version__
        self.tokenizer = tokenizer
        self.stop_ids = stop_token_ids(tokenizer, args.model_id, args.local_files_only)
        self.llm = LLM(
            model=args.model_id,
            tensor_parallel_size=args.tensor_parallel,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            seed=args.seed,
            skip_tokenizer_init=True,  # vLLM's tokenizer would be the lossy one
            # Custom all-reduce crashed on some Kaggle T4 pairs (PCIe, no NVLink) with
            # "custom_all_reduce.cuh:164 invalid argument" (e14/e16, 17 Sep); NCCL works everywhere.
            disable_custom_all_reduce=True,
            # Hybrid models (Qwen3.5) need one state block per running sequence; the default 256
            # exceeded the 157 blocks a T4 holds (e15, 17 Sep).
            **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
            # Qwen3.5 checkpoints are vision-language models; text-only skips the vision tower.
            **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if args.text_only else {}),
        )
        self.params = SamplingParams(
            n=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
            stop_token_ids=self.stop_ids,
            detokenize=False,
        )

    def generate(self, prompt_ids: list[list[int]]) -> list[list[tuple[str, int, bool]]]:
        outputs = self.llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], self.params, use_tqdm=False)
        results = []
        for output in outputs:
            samples = []
            for completion in output.outputs:
                token_ids = list(completion.token_ids)
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                samples.append((text, len(token_ids), completion.finish_reason == "length"))
            results.append(samples)
        return results


class HfBackend:
    name = "hf"

    def __init__(self, args: argparse.Namespace, tokenizer):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.version = transformers.__version__
        self.args = args
        self.tokenizer = tokenizer
        cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.float16 if cuda else torch.float32,
            device_map="auto" if cuda else None,
            local_files_only=args.local_files_only,
        ).eval()
        self.stop_ids = stop_token_ids(tokenizer, args.model_id, args.local_files_only)
        torch.manual_seed(args.seed)

    def generate(self, prompt_ids: list[list[int]]) -> list[list[tuple[str, int, bool]]]:
        torch, args, tokenizer = self.torch, self.args, self.tokenizer
        stop = set(self.stop_ids)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (self.stop_ids or [0])[0]
        results = []
        for ids in prompt_ids:
            input_ids = torch.tensor([ids], device=self.model.device)
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.k,
                    eos_token_id=self.stop_ids or None,
                    pad_token_id=pad,
                )
            samples = []
            for row in output[:, len(ids):].tolist():
                length = next((i for i, token in enumerate(row) if token in stop), len(row))
                samples.append((tokenizer.decode(row[:length], skip_special_tokens=True), length, length == len(row)))
            results.append(samples)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=["base", "variants"])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--base-records", default=None, help="glob of base-phase records.jsonl (variants phase)")
    parser.add_argument("--min-base-accuracy", type=float, default=1.0)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--include-confirm", action="store_true",
                        help="also sample official-like categories retained under their historical confirm tag")
    parser.add_argument("--tiers", nargs="+", default=None, choices=["amc", "aime", "olympiad"])
    parser.add_argument("--sources", nargs="+", default=None,
                        help="restrict to named pool sources, e.g. aime-2024")
    parser.add_argument("--exclude-problem-keys", nargs="+", default=None,
                        help="exclude problem_key values, used to prevent official-label overlap")
    parser.add_argument("--problem-keys", nargs="+", default=None,
                        help="restrict to these problem_key values (unlike pool_id, they are unique)")
    parser.add_argument("--k", type=int, default=8, help="samples per item")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=None, help="default: max-new-tokens + 2048")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--dtype", default="float16", help="float16 on T4 (Finding 13)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true", help="skip CUDA graphs (saves memory on small GPUs)")
    parser.add_argument("--text-only", action="store_true", help="vLLM: do not load a vision tower (Qwen3.5)")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="vLLM concurrent sequences (default: vLLM's)")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop starting new chunks after this long, so a Kaggle session ends cleanly")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-lossy-tokenizer", action="store_true", help="debug only; see tokenizer_check.py")
    parser.add_argument("--chunk", type=int, default=32, help="items per generate call; also the flush interval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.max_model_len = args.max_model_len or args.max_new_tokens + 2048
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < --num-shards")

    items, info = build_items(args)
    assignment = assign_shards([i["item_id"] for i in items], args.num_shards, "roundrobin")
    # Hash order mixes tiers and sources, so a run stopped by --max-hours is still a fair sample.
    shard = sorted(
        (i for i in items if assignment[i["item_id"]] == args.shard_index),
        key=lambda i: hashlib.sha1(i["item_id"].encode()).hexdigest(),
    )
    if args.limit:
        shard = shard[: args.limit]

    out_dir = args.out_dir / args.phase / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    done = set()
    if records_path.is_file():
        done = {json.loads(l)["item_id"] for l in records_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [i for i in shard if i["item_id"] not in done]
    print(f"{args.phase}: {info} -> shard {args.shard_index}/{args.num_shards}: {len(shard)} items, "
          f"{len(shard) - len(todo)} done, {len(todo)} to do", flush=True)
    if not todo:
        print("nothing to do")
        return

    started = time.time()
    tokenizer, tokenizer_info = load_checked_tokenizer(
        args.model_id, allow_lossy=args.allow_lossy_tokenizer, local_files_only=args.local_files_only
    )
    print(f"tokenizer: {tokenizer_info}", flush=True)
    backend = VllmBackend(args, tokenizer) if args.backend == "vllm" else HfBackend(args, tokenizer)
    load_seconds = time.time() - started
    metadata = {
        "stage": f"E7 {args.phase}",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "backend": backend.name,
        "backend_version": backend.version,
        "prompt_suffix": PROMPT_SUFFIX,
        "stop_token_ids": backend.stop_ids,
        **tokenizer_info,
        "selection": info,
        "n_items": len(shard),
        "load_seconds": round(load_seconds, 1),
        "platform": platform.platform(),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    generated_tokens = 0
    loop_started = time.time()
    for start in range(0, len(todo), args.chunk):
        if args.max_hours and time.time() - started > args.max_hours * 3600:
            print(f"time budget of {args.max_hours} h reached; stopping (resumable)", flush=True)
            break
        chunk = todo[start : start + args.chunk]
        completions = backend.generate([encode_prompt(tokenizer, item["problem"]) for item in chunk])
        with records_path.open("a", encoding="utf-8") as handle:
            for item, samples in zip(chunk, completions):
                answers = [extract_answer(text) for text, _, _ in samples]
                tokens = [n for _, n, _ in samples]
                generated_tokens += sum(tokens)
                record = {
                    **{k: v for k, v in item.items() if k != "problem"},
                    "model_id": args.model_id,
                    "phase": args.phase,
                    "answers": answers,
                    "n_samples": len(samples),
                    "n_correct": sum(a == item["expected_answer"] for a in answers),
                    "n_no_answer": sum(a is None for a in answers),
                    "n_hit_length": sum(hit for _, _, hit in samples),
                    "completion_tokens": tokens,
                    "sample_tail": samples[0][0][-1500:] if samples else "",
                }
                handle.write(json.dumps(record) + "\n")
        elapsed = time.time() - loop_started
        finished = start + len(chunk)
        print(
            f"  [{finished}/{len(todo)}] {generated_tokens / max(elapsed, 1e-9):.0f} tok/s, "
            f"{elapsed / finished:.1f}s/item, ~{elapsed / finished * (len(todo) - finished) / 60:.0f} min left",
            flush=True,
        )
        metadata.update(
            items_done_this_session=finished,
            generated_tokens=generated_tokens,
            generation_seconds=round(elapsed, 1),
            tokens_per_second=round(generated_tokens / max(elapsed, 1e-9), 1),
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\ndone in {(time.time() - started) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
