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

Output: <out-dir>/<phase>/shard_XX_of_YY/records.jsonl, one line per item, flushed per chunk
and resumable; metadata.json records sampling settings, versions, and throughput.

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
    if args.tiers:
        pool = pool[pool["tier"].isin(args.tiers)]
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


def format_prompt(tokenizer, problem: str) -> str:
    message = [{"role": "user", "content": problem + PROMPT_SUFFIX}]
    try:
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    except (AttributeError, TypeError, ValueError):
        return problem + PROMPT_SUFFIX  # tokenizers without a chat template (smoke tests)


class VllmBackend:
    name = "vllm"

    def __init__(self, args: argparse.Namespace):
        from vllm import LLM, SamplingParams, __version__

        self.version = __version__
        self.llm = LLM(
            model=args.model_id,
            tensor_parallel_size=args.tensor_parallel,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            seed=args.seed,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(
            n=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )

    def generate(self, prompts: list[str]) -> list[list[tuple[str, int, bool]]]:
        outputs = self.llm.generate(prompts, self.params, use_tqdm=False)
        return [
            [(c.text, len(c.token_ids), c.finish_reason == "length") for c in output.outputs]
            for output in outputs
        ]


class HfBackend:
    name = "hf"

    def __init__(self, args: argparse.Namespace):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.version = transformers.__version__
        self.args = args
        cuda = torch.cuda.is_available()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.float16 if cuda else torch.float32,
            device_map="auto" if cuda else None,
            local_files_only=args.local_files_only,
        ).eval()
        torch.manual_seed(args.seed)

    def generate(self, prompts: list[str]) -> list[list[tuple[str, int, bool]]]:
        torch, args, tokenizer = self.torch, self.args, self.tokenizer
        eos = tokenizer.eos_token_id
        results = []
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.k,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos,
                )
            samples = []
            for row in output[:, encoded["input_ids"].shape[1]:].tolist():
                length = row.index(eos) if eos in row else len(row)
                samples.append((tokenizer.decode(row[:length], skip_special_tokens=True), length, eos not in row))
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
    parser.add_argument("--include-confirm", action="store_true", help="also sample categories awaiting organizer confirmation")
    parser.add_argument("--tiers", nargs="+", default=None, choices=["amc", "aime", "olympiad"])
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
    parser.add_argument("--local-files-only", action="store_true")
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
    shard = sorted((i for i in items if assignment[i["item_id"]] == args.shard_index), key=lambda i: i["item_id"])
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
    backend = VllmBackend(args) if args.backend == "vllm" else HfBackend(args)
    load_seconds = time.time() - started
    metadata = {
        "stage": f"E7 {args.phase}",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "backend": backend.name,
        "backend_version": backend.version,
        "prompt_suffix": PROMPT_SUFFIX,
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
        chunk = todo[start : start + args.chunk]
        prompts = [format_prompt(backend.tokenizer, item["problem"]) for item in chunk]
        completions = backend.generate(prompts)
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
