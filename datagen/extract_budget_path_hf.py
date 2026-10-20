"""E49: E37's four-budget reasoning-path features, generated in batches with `transformers`.

E37's official transfer was closed by a runtime gate measured under vLLM with at most two
sequences in flight on a T4 pair (`scripts/e37_gpu_bundle.py`). The evaluation image ships no
vLLM, so that figure described neither the path a submission would take nor the batching a
96 GB card allows (REVIEW_GENERALIZATION.md §3). This extractor is the path that would ship:

* plain `model.generate`, greedy, on left-padded batches;
* every (problem, budget) pair is its own sequence, and a per-row stopping criterion ends each
  at its own budget. The four budgets therefore stay independent generations, as E37 requires
  (E38 closed prefix reuse), while sharing one batch;
* on GPU, calls are sized from **measured** memory, not from config arithmetic. Hand-set caps
  ran out of memory late in Qwen's and Olmo's first E49 runs, after most of a call's work was done.
  - Three short probe generations fit a per-GPU model:
    peak = overhead + sequences x (per sequence + per position x cache length).
  - Each call is filled while every GPU's predicted peak stays under 85% of its free memory.
  - An out-of-memory error halves that margin for the rest of the session and re-splits the call.
  - Predicted and measured peaks are recorded for every call, so each run checks the model.
  - `--max-batch-tokens` remains an optional extra cap, and is the only limit on CPU;
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


def generate(model, prompts: list[list[int]], budgets: list[int], pad_id: int, stop_ids: list[int], **extra):
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
            stopping_criteria=StoppingCriteriaList([RowBudget(width, budgets)]), **extra,
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


def positions(batch: list[tuple]) -> int:
    """Cache length every row of a call reaches: padded to the widest prompt, run to the longest budget."""
    return max(len(job[2]) for job in batch) + max(job[1] for job in batch)


def cost(batch: list[tuple]) -> int:
    """Cached positions a call holds, summed over its rows."""
    return len(batch) * positions(batch)


# Memory calibration: (sequences, decode steps) of three short generations. The first two differ
# only in length and give the per-position cost; the first and third differ only in batch size and
# separate the per-sequence cost from fixed overhead.
PROBES = ((2, 64), (2, 448), (6, 64))
SAFETY = 0.85  # share of each GPU's free memory a call may be predicted to fill


def fit_memory(peaks: dict[tuple[int, int], list[int]], width: int) -> list[tuple[float, float, float]]:
    """Per device (overhead, per sequence, per position), in bytes, from the probes' peak memory.

    A call's peak above idle is modelled as overhead + sequences x (per_sequence + per_position x
    positions). Measuring it covers what a config-based estimate misses: allocator overhead,
    Qwen3.5's linear-attention state, Olmo's sliding-window layers, and how `device_map` splits
    the layers between GPUs.
    """
    (n1, t1), (_, t2), (n3, _) = PROBES
    model = []
    for short, long, wide in zip(peaks[PROBES[0]], peaks[PROBES[1]], peaks[PROBES[2]]):
        per_position = max(0.0, (long - short) / (n1 * (t2 - t1)))
        per_row = max(0.0, (wide - short) / (n3 - n1))  # one row at width + t1 positions
        overhead = max(0.0, short - n1 * per_row)
        model.append((overhead, max(0.0, per_row - per_position * (width + t1)), per_position))
    return model


class Sizer:
    """Decides whether a call fits: an optional cap on cached positions and, on GPU, measured memory."""

    def __init__(self, cap: int = 0, memory: list[tuple[float, float, float]] | None = None,
                 available: list[int] | None = None):
        self.cap, self.memory, self.available, self.safety = cap, memory, available, SAFETY

    def predict(self, batch: list[tuple]) -> list[float]:
        return [o + len(batch) * (k + s * positions(batch)) for o, k, s in self.memory]

    def fits(self, batch: list[tuple]) -> bool:
        if self.cap and cost(batch) > self.cap:
            return False
        return not self.memory or all(p <= self.safety * a for p, a in zip(self.predict(batch), self.available))

    def shrink(self, batch: list[tuple]) -> str:
        """After an out-of-memory error, tighten the limit that applies, for the rest of the run."""
        if self.memory:
            self.safety /= 2
            return f"memory safety {self.safety:.3f}"
        self.cap = min(self.cap or cost(batch), cost(batch)) // 2
        return f"max_batch_tokens {self.cap}"


def calibrate(model, prompt: list[int], pad_id: int) -> tuple[Sizer, dict]:
    """Fit the memory model on this checkpoint and these GPUs with the three probes."""
    devices = range(torch.cuda.device_count())
    peaks = {}
    for n, steps in PROBES:
        torch.cuda.empty_cache()
        idle = [torch.cuda.memory_allocated(d) for d in devices]
        for d in devices:
            torch.cuda.reset_peak_memory_stats(d)
        # min_new_tokens holds EOS back, so every row really reaches `steps`.
        generate(model, [prompt] * n, [steps] * n, pad_id, [], min_new_tokens=steps)
        peaks[(n, steps)] = [torch.cuda.max_memory_allocated(d) - i for d, i in zip(devices, idle)]
    torch.cuda.empty_cache()
    available = [torch.cuda.mem_get_info(d)[0] for d in devices]
    memory = fit_memory(peaks, len(prompt))
    record = {"probes": [list(p) for p in PROBES], "probe_width": len(prompt), "safety": SAFETY, "devices": [
        {"available_gb": round(a / 2**30, 2), "overhead_gb": round(o / 2**30, 3),
         "per_sequence_mb": round(k / 2**20, 2), "per_position_kb": round(s / 2**10, 2)}
        for a, (o, k, s) in zip(available, memory)
    ]}
    return Sizer(memory=memory, available=available), record


def batches(jobs: list[tuple], fits) -> list[list[tuple]]:
    """Group (key, budget, prompt) jobs, longest budget first, filling each call while `fits` allows."""
    ordered = sorted(jobs, key=lambda job: (-job[1], len(job[2]), job[0]))
    groups, current = [], []
    for job in ordered:
        trial = current + [job]
        if current and not fits(trial):
            groups.append(current)
            current = [job]
        else:
            current = trial
    return groups + [current] if current else groups


def run_jobs(model, jobs: list[tuple], sizer: Sizer, pad_id: int, stop_ids: list[int], calls: list[dict]) -> dict:
    """Generate every job; on out-of-memory, tighten `sizer` and re-split. Returns {(key, budget): (tokens, hit)}."""
    results = {}
    cuda = torch.cuda.is_available()
    devices = range(torch.cuda.device_count()) if cuda else range(0)
    pending = batches(jobs, sizer.fits)
    while pending:
        batch = pending.pop(0)
        idle = [torch.cuda.memory_allocated(d) for d in devices]
        for d in devices:
            torch.cuda.reset_peak_memory_stats(d)
        try:
            outputs, stats = generate(model, [j[2] for j in batch], [j[1] for j in batch], pad_id, stop_ids)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            limit = sizer.shrink(batch)
            calls.append({"oom_retry": True, "sequences": len(batch), "positions": positions(batch), "new_limit": limit})
            print(f"  out of memory at {len(batch)} sequences; retrying with {limit}", flush=True)
            pending = batches(batch, sizer.fits) + pending
            continue
        if cuda:
            stats["measured_peak_gb"] = [round((torch.cuda.max_memory_allocated(d) - i) / 2**30, 2) for d, i in zip(devices, idle)]
        if sizer.memory:
            stats["predicted_peak_gb"] = [round(p / 2**30, 2) for p in sizer.predict(batch)]
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
                        help="optional cap on sequences x (prompt width + budget) per call; on GPU, calls are "
                             "also sized from a measured memory model; 0 = no cap")
    parser.add_argument("--group-size", type=int, default=0, help="problems per written group; 0 = all at once")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0, help="skip this many rows of the sorted shard (split one model over kernels)")
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
    rows = rows[rows["problem_key"].map(assignment) == args.shard_index].sort_values("problem_key").iloc[args.offset:]
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
    # One sizer for the whole session, so a limit tightened after out-of-memory stays tightened.
    sizer = Sizer(args.max_batch_tokens)
    if cuda:
        try:
            sizer, session["memory_model"] = calibrate(model, max(prompts.values(), key=len), pad_id)
            sizer.cap = args.max_batch_tokens
            print(f"memory model: {session['memory_model']['devices']}", flush=True)
        except Exception as exc:  # sizing is an optimisation; never lose a run to it
            torch.cuda.empty_cache()
            session["memory_model"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"memory calibration failed ({exc}); sizing by --max-batch-tokens only", flush=True)

    metadata_path = args.out_dir / "metadata.json"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    metadata = {
        "stage": STAGE,
        "model_id": args.model_id,
        "rows_filtered_to_model": not args.no_filter_model,
        "rows_in_shard": int(len(rows)),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "offset": args.offset,
        "limit": args.limit,
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
        primary = run_jobs(model, [(k, scaled[b], p) for k, b, p in jobs], sizer, pad_id, stops, session["calls"])

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
        forced = run_jobs(model, forced_jobs, sizer, pad_id, stops, session["calls"]) if forced_jobs else {}

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
