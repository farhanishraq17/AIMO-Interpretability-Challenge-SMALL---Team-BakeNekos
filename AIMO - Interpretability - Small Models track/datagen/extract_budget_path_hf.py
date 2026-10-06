"""E49: E37's four-budget reasoning-path features, generated in batches with `transformers`.

E37's official transfer was closed by a runtime gate measured under vLLM with at most two
sequences in flight on a T4 pair (`scripts/e37_gpu_bundle.py`). The evaluation image ships no
vLLM, so that figure described neither the path a submission would take nor the batching a
96 GB card allows (REVIEW_GENERALIZATION.md §3). This extractor is the path that would ship:

* plain `model.generate`, greedy, on left-padded batches;
* every (problem, budget) pair is its own sequence, and a per-row stopping criterion ends each
  at its own budget. The four budgets therefore stay independent generations, as E37 requires
  (E38 closed prefix reuse), while sharing one batch;
* `--max-batch-tokens` caps sequences x (prompt width + budget) per call, which is the KV-cache
  bound. A call that runs out of memory is retried at half the cap, and the retry is recorded;
* the prompt, answer forcing, features and record schema are E37's, imported unchanged, so
  `train/eval_e37_official.py` and `train/eval_e49_official.py` both read the output.

Two preflights guard the batching:

1. the first prompt's logits must be finite;
2. the shortest prompt, decoded alone and inside a batch padded to the longest prompt, must agree
   on its first tokens. This catches left-padding bugs, for example in Qwen3.5's torch fallback
   for linear attention.

Tokens, steps and seconds for every call go to `metadata.json`, so throughput at each batch shape
can be projected to other hardware. Records are written per group of problems and the run
resumes from `features.jsonl`.

Run:    python datagen/extract_budget_path_hf.py --model-id Qwen/Qwen3.5-4B --out-dir runs/e49-qwen
Smoke:  python datagen/extract_budget_path_hf.py --model-id hf-internal-testing/tiny-random-LlamaForCausalLM \
            --no-filter-model --limit 2 --smoke-divisor 256 --local-files-only --out-dir <scratch>
        (plumbing only: budgets are divided by 256 and `train/eval_e49_official.py` refuses the output)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "datagen", ROOT / "datagen" / "perturb", ROOT / "train"):
    sys.path.insert(0, str(path))

from eval_e36_budget_path import BUDGETS, trace_features  # noqa: E402
from extract_budget_path import encode_forced_prompt, extracted_answer  # noqa: E402
from extract_features import assign_shards, resolve_dtype  # noqa: E402
from sample import encode_prompt, stop_token_ids  # noqa: E402
from tokenizer_check import load_checked_tokenizer  # noqa: E402

STAGE = "E49 official budget-path extraction (transformers, batched)"
FORCED_TOKENS = 512  # E37's answer-forcing cap
PREFLIGHT_TOKENS = 32
PREFLIGHT_MIN_AGREEMENT = 8


class RowBudget(StoppingCriteria):
    """Finish each row once it has generated its own budget; `generate` pads it from then on."""

    def __init__(self, width: int, budgets: list[int]):
        self.width = width
        self.budgets = torch.tensor(budgets)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        return (input_ids.shape[1] - self.width) >= self.budgets.to(input_ids.device)


def generate(model, prompts: list[list[int]], budgets: list[int], pad_id: int, stop_ids: list[int]):
    """Greedy decoding of left-padded prompts. Returns, per row, (tokens, hit_budget), and call stats."""
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, prompt in enumerate(prompts):
        ids[row, width - len(prompt):] = torch.tensor(prompt)
        mask[row, width - len(prompt):] = 1
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids.to(model.device), attention_mask=mask.to(model.device), do_sample=False,
            max_new_tokens=max(budgets), eos_token_id=stop_ids or None, pad_token_id=pad_id,
            stopping_criteria=StoppingCriteriaList([RowBudget(width, budgets)]),
        )
    seconds = time.time() - started
    stop = set(stop_ids)
    results = []
    for tokens, budget in zip(output[:, width:].tolist(), budgets):
        # Cut at the budget before looking for a stop token: finished rows are padded, and the pad
        # id is often an EOS id.
        tokens = tokens[:budget]
        end = next((i for i, token in enumerate(tokens) if token in stop), None)
        results.append((tokens if end is None else tokens[:end], end is None))
    stats = {
        "sequences": len(prompts), "width": width, "max_budget": max(budgets),
        "steps": int(output.shape[1] - width),
        "generated_tokens": sum(len(tokens) for tokens, _ in results),
        "seconds": round(seconds, 2),
    }
    return results, stats


def cost(batch: list[tuple]) -> int:
    """KV-cache tokens a call holds: every row is padded to the widest prompt and runs to the longest budget."""
    return len(batch) * (max(len(job[2]) for job in batch) + max(job[1] for job in batch))


def batches(jobs: list[tuple], cap: int) -> list[list[tuple]]:
    """Group (key, budget, prompt) jobs, longest budget first, so that each call's cost <= cap."""
    ordered = sorted(jobs, key=lambda job: (-job[1], len(job[2]), job[0]))
    groups, current = [], []
    for job in ordered:
        trial = current + [job]
        if current and cap and cost(trial) > cap:
            groups.append(current)
            current = [job]
        else:
            current = trial
    return groups + [current] if current else groups


def run_jobs(model, jobs: list[tuple], cap: int, pad_id: int, stop_ids: list[int], calls: list[dict]) -> dict:
    """Generate every job, halving the cap on out-of-memory. Returns {(key, budget): (tokens, hit)}."""
    results = {}
    pending = batches(jobs, cap)
    while pending:
        batch = pending.pop(0)
        try:
            outputs, stats = generate(model, [j[2] for j in batch], [j[1] for j in batch], pad_id, stop_ids)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            cap = min(cap or cost(batch), cost(batch)) // 2
            calls.append({"oom_retry": True, "sequences": len(batch), "new_cap": cap})
            print(f"  out of memory at {len(batch)} sequences; retrying at max_batch_tokens={cap}", flush=True)
            pending = batches(batch, cap) + pending
            continue
        calls.append(stats)
        print(f"  {stats['sequences']} seqs, budget {stats['max_budget']}: {stats['generated_tokens']} tokens "
              f"in {stats['seconds']:.0f}s ({stats['generated_tokens'] / max(stats['seconds'], 1e-9):.0f} tok/s)",
              flush=True)
        for job, output in zip(batch, outputs):
            results[(job[0], job[1])] = output
    return results


def preflight(model, prompts: list[list[int]], pad_id: int, stop_ids: list[int]) -> dict:
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([prompts[0]], device=model.device)).logits[0, -1]
    if not torch.isfinite(logits).all():
        raise SystemExit("preflight: non-finite logits on the first prompt (fp16 overflow?)")
    short, long = min(prompts, key=len), max(prompts, key=len)
    (alone, _), = generate(model, [short], [PREFLIGHT_TOKENS], pad_id, stop_ids)[0]
    (padded, _), _ = generate(model, [short, long], [PREFLIGHT_TOKENS] * 2, pad_id, stop_ids)[0]
    agreement = next((i for i, (a, b) in enumerate(zip(alone, padded)) if a != b), min(len(alone), len(padded)))
    required = min(PREFLIGHT_MIN_AGREEMENT, len(alone), len(padded))
    if agreement < required:
        raise SystemExit(f"preflight: padded decoding diverges after {agreement} tokens; left padding is broken")
    return {"logits_finite": True, "padding_agreement_tokens": agreement, "padding_width_gap": len(long) - len(short)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--rows", type=Path, default=ROOT / "data" / "train_main" / "rows.parquet")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dtype", default="float16", choices=["auto", "float16", "bfloat16", "float32"],
                        help="float16 on T4s (emulated bf16 is ~3x slower, Finding 13); bfloat16 on cards with native bf16")
    parser.add_argument("--max-batch-tokens", type=int, default=0,
                        help="cap on sequences x (prompt width + budget) per generate call; 0 = one call per group")
    parser.add_argument("--group-size", type=int, default=0, help="problems per written group; 0 = all at once")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=None, help="start no new group after this long")
    parser.add_argument("--no-filter-model", action="store_true",
                        help="plumbing only: take every labelled small-track problem, whatever model labelled it")
    parser.add_argument("--smoke-divisor", type=int, default=1, help="plumbing only: generate budget // divisor tokens")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    started = time.time()

    rows = pd.read_parquet(args.rows)
    rows = rows[rows["is_small_track"].astype(bool) & rows["is_labelled"].astype(bool)]
    if not args.no_filter_model:
        rows = rows[rows["model_id"] == args.model_id]
    rows = rows.drop_duplicates("problem_key")
    if rows.empty:
        raise SystemExit(f"no labelled small-track rows for {args.model_id}")
    assignment = assign_shards(list(rows["problem_key"]), args.num_shards, "roundrobin")
    rows = rows[rows["problem_key"].map(assignment) == args.shard_index].sort_values("problem_key")
    rows = rows.head(args.limit) if args.limit else rows

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "features.jsonl"
    done = set()
    if records_path.is_file():
        done = {json.loads(line)["problem_key"] for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = rows[~rows["problem_key"].isin(done)].reset_index(drop=True)
    print(f"{args.model_id}: {len(rows)} rows in shard, {len(done)} done, {len(todo)} to do", flush=True)
    if todo.empty:
        return

    cuda = torch.cuda.is_available()
    dtype = resolve_dtype(args.dtype) if cuda else torch.float32
    tokenizer, tokenizer_info = load_checked_tokenizer(args.model_id, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=dtype, device_map="auto" if cuda else None, local_files_only=args.local_files_only,
    ).eval()
    stops = stop_token_ids(tokenizer, args.model_id, args.local_files_only)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (stops or [0])[0]
    prompts = {str(row.problem_key): encode_prompt(tokenizer, str(row.problem)) for row in todo.itertuples(index=False)}
    session = {
        "started_unix": round(started), "load_seconds": round(time.time() - started, 1),
        "preflight": preflight(model, list(prompts.values()), pad_id, stops), "calls": [],
    }

    metadata_path = args.out_dir / "metadata.json"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    metadata = {
        "stage": STAGE,
        "model_id": args.model_id,
        "rows_filtered_to_model": not args.no_filter_model,
        "rows_in_shard": int(len(rows)),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "budgets": list(BUDGETS),
        "smoke_divisor": args.smoke_divisor,
        "forced_tokens": FORCED_TOKENS,
        "decoding": "greedy",
        "dtype": str(dtype),
        "max_batch_tokens": args.max_batch_tokens,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "tokenizer": tokenizer_info,
        "stop_token_ids": stops,
        "pad_token_id": pad_id,
        "sessions": previous.get("sessions", []) + [session],
    }

    def write_metadata() -> None:
        session["seconds"] = round(time.time() - started, 1)
        session["generated_tokens"] = sum(call.get("generated_tokens", 0) for call in session["calls"])
        session["generation_seconds"] = round(sum(call.get("seconds", 0.0) for call in session["calls"]), 1)
        if cuda:
            session["peak_memory_gb"] = round(sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30, 2)
        metadata["rows_written"] = len(done)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_metadata()
    size = args.group_size or len(todo)
    for start in range(0, len(todo), size):
        if args.max_hours and time.time() - started > args.max_hours * 3600:
            print(f"--max-hours {args.max_hours} reached; stopping (resumable)", flush=True)
            break
        group = todo.iloc[start:start + size]
        keys = [str(key) for key in group["problem_key"]]
        jobs = [(key, budget, prompts[key]) for key in keys for budget in BUDGETS]
        scaled = {budget: max(1, budget // args.smoke_divisor) for budget in BUDGETS}
        primary = run_jobs(model, [(k, scaled[b], p) for k, b, p in jobs], args.max_batch_tokens, pad_id, stops, session["calls"])

        traces = {}
        for key, budget, _ in jobs:
            tokens, hit = primary[(key, scaled[budget])]
            trace = tokenizer.decode(tokens, skip_special_tokens=True)
            traces[(key, budget)] = (trace, extracted_answer(trace), hit, len(tokens))
        problems = dict(zip(keys, group["problem"].astype(str)))
        need = [(key, budget) for (key, budget), (_, answer, hit, _) in traces.items() if hit or not answer]
        forced_budget = max(1, FORCED_TOKENS // args.smoke_divisor)
        forced_jobs = [(f"{key}@{budget}", forced_budget, encode_forced_prompt(tokenizer, problems[key], traces[(key, budget)][0]))
                       for key, budget in need]
        forced = run_jobs(model, forced_jobs, args.max_batch_tokens, pad_id, stops, session["calls"]) if forced_jobs else {}

        with records_path.open("a", encoding="utf-8") as handle:
            for row in group.itertuples(index=False):
                key = str(row.problem_key)
                collected, summaries = [], []
                for budget in BUDGETS:
                    trace, answer, hit, count = traces[(key, budget)]
                    was_forced = (key, budget) in need
                    if was_forced:
                        forced_answer = extracted_answer(tokenizer.decode(forced[(f"{key}@{budget}", forced_budget)][0], skip_special_tokens=True))
                        answer = forced_answer or answer
                    collected.append({"token_budget": budget, "predicted_answer": answer, "answer_forced": was_forced, "trace": trace})
                    summaries.append({"budget": budget, "answer": answer, "answer_forced": was_forced,
                                      "hit_length": hit, "tokens": count, "trace_tail": trace[-500:]})
                handle.write(json.dumps({
                    "problem_key": key,
                    "model_id": args.model_id,
                    "label_model_id": str(row.model_id),
                    "reasoning_effort": str(row.reasoning_effort),
                    "is_robust": bool(row.is_robust),
                    **trace_features(pd.DataFrame(collected)),
                    "budget_summaries": summaries,
                }) + "\n")
                done.add(key)
        write_metadata()
        print(f"[{len(done)}/{len(rows)}] {(time.time() - started) / 60:.1f} min, "
              f"{session['generated_tokens']} generated tokens this session", flush=True)


if __name__ == "__main__":
    main()
