"""Extract E37's four-budget reasoning-path features for official AIMO rows.

The target model is run greedily at 2k/4k/8k/16k.  If a trace reaches its cap or has no boxed
answer, a short deterministic final-answer request is made, matching the public archive's
documented answer-forced convention.  Only architecture-independent E36 features are saved.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "datagen", ROOT / "datagen" / "perturb", ROOT / "train"):
    sys.path.insert(0, str(path))

from eval_e36_budget_path import BUDGETS, trace_features  # noqa: E402
from answers import last_boxed  # noqa: E402
from sample import PROMPT_SUFFIX, encode_prompt, stop_token_ids  # noqa: E402
from tokenizer_check import load_checked_tokenizer  # noqa: E402


SMALL_MODELS = {
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}
FORCE_REQUEST = "Stop reasoning and return only your final answer within \\boxed{}."


def extracted_answer(text: str) -> str:
    return last_boxed(str(text).rsplit("</think>", 1)[-1]) or last_boxed(str(text)) or ""


def encode_forced_prompt(tokenizer, problem: str, trace: str) -> list[int]:
    # A bounded tail keeps the follow-up comfortably inside the 18k context even when the
    # primary trace used the whole 16k budget.  The same helper will ship at inference time.
    content = (
        problem + PROMPT_SUFFIX + "\n\nHere is the end of your unfinished work:\n" + trace[-8_000:]
        + "\n\n" + FORCE_REQUEST
    )
    messages = [{"role": "user", "content": content}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    except (AttributeError, TypeError, ValueError):
        return tokenizer(content)["input_ids"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, choices=sorted(SMALL_MODELS))
    parser.add_argument("--instances", type=Path, default=ROOT / "data" / "train_main" / "rows.parquet")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams, __version__ as vllm_version

    all_rows = pd.read_parquet(args.instances)
    rows = all_rows.loc[
        (all_rows["model_id"] == args.model_id)
        & all_rows["is_small_track"].astype(bool)
        & all_rows["is_labelled"].astype(bool)
    ].copy()
    if rows.empty:
        raise SystemExit(f"no labelled small-track rows for {args.model_id}")
    if rows["problem_key"].duplicated().any():
        raise SystemExit("duplicate problem_key within model")
    rows = rows.sort_values("problem_key").reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "features.jsonl"
    done = set()
    if records_path.is_file():
        done = {
            json.loads(line)["problem_key"]
            for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
    rows = rows.loc[~rows["problem_key"].isin(done)].reset_index(drop=True)
    print(f"{args.model_id}: {len(done)} done, {len(rows)} remaining", flush=True)
    if rows.empty:
        return

    started = time.time()
    tokenizer, tokenizer_info = load_checked_tokenizer(args.model_id, local_files_only=args.local_files_only)
    stops = stop_token_ids(tokenizer, args.model_id, args.local_files_only)
    engine = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel,
        dtype=args.dtype,
        max_model_len=18_944,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        skip_tokenizer_init=True,
        disable_custom_all_reduce=True,
        **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
        **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if args.text_only else {}),
    )
    metadata = {
        "stage": "E37 official budget-path extraction",
        "model_id": args.model_id,
        "budgets": list(BUDGETS),
        "seed": args.seed,
        "temperature": 0.0,
        "prompt_suffix": PROMPT_SUFFIX,
        "force_request": FORCE_REQUEST,
        "vllm_version": vllm_version,
        "tokenizer": tokenizer_info,
        "stop_token_ids": stops,
        "platform": platform.platform(),
        "load_seconds": round(time.time() - started, 1),
    }
    metadata_path = args.out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    generated = 0
    for start in range(0, len(rows), args.chunk):
        chunk = rows.iloc[start : start + args.chunk]
        collected = {str(row.problem_key): [] for row in chunk.itertuples(index=False)}
        summaries = {str(row.problem_key): [] for row in chunk.itertuples(index=False)}
        prompt_ids = [encode_prompt(tokenizer, str(row.problem)) for row in chunk.itertuples(index=False)]
        for budget in BUDGETS:
            params = SamplingParams(
                n=1, temperature=0.0, max_tokens=budget, seed=args.seed,
                stop_token_ids=stops, detokenize=False,
            )
            outputs = engine.generate([{"prompt_token_ids": ids} for ids in prompt_ids], params, use_tqdm=False)
            forced_requests = []
            forced_positions = []
            primary = []
            for position, (row, output) in enumerate(zip(chunk.itertuples(index=False), outputs)):
                completion = output.outputs[0]
                tokens = list(completion.token_ids)
                trace = tokenizer.decode(tokens, skip_special_tokens=True)
                answer = extracted_answer(trace)
                hit = completion.finish_reason == "length"
                primary.append((str(row.problem_key), trace, answer, hit, len(tokens)))
                generated += len(tokens)
                if hit or not answer:
                    forced_positions.append(position)
                    forced_requests.append(encode_forced_prompt(tokenizer, str(row.problem), trace))
            forced_answers = {}
            if forced_requests:
                forced_params = SamplingParams(
                    n=1, temperature=0.0, max_tokens=512, seed=args.seed,
                    stop_token_ids=stops, detokenize=False,
                )
                forced_outputs = engine.generate(
                    [{"prompt_token_ids": ids} for ids in forced_requests], forced_params, use_tqdm=False
                )
                for position, output in zip(forced_positions, forced_outputs):
                    tokens = list(output.outputs[0].token_ids)
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    generated += len(tokens)
                    forced_answers[position] = extracted_answer(text)
            for position, (problem_key, trace, answer, hit, token_count) in enumerate(primary):
                was_forced = position in forced_answers
                if was_forced and forced_answers[position]:
                    answer = forced_answers[position]
                collected[problem_key].append({
                    "token_budget": budget,
                    "predicted_answer": answer,
                    "answer_forced": was_forced,
                    "trace": trace,
                })
                summaries[problem_key].append({
                    "budget": budget,
                    "answer": answer,
                    "answer_forced": was_forced,
                    "hit_length": hit,
                    "tokens": token_count,
                    "trace_tail": trace[-500:],
                })

        with records_path.open("a", encoding="utf-8") as handle:
            for row in chunk.itertuples(index=False):
                key = str(row.problem_key)
                features = trace_features(pd.DataFrame(collected[key]))
                record = {
                    "problem_key": key,
                    "model_id": args.model_id,
                    "reasoning_effort": str(row.reasoning_effort),
                    "is_robust": bool(row.is_robust),
                    **features,
                    "budget_summaries": summaries[key],
                }
                handle.write(json.dumps(record) + "\n")
        finished = start + len(chunk)
        elapsed = time.time() - started
        print(f"[{finished}/{len(rows)}] {elapsed / 60:.1f} min, {generated} generated tokens", flush=True)
        metadata.update(rows_done_this_session=finished, generated_tokens=generated, seconds=round(elapsed, 1))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
