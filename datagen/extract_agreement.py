"""E14: test-time answer agreement. Does the model's answer survive rewrites of the problem?

The label is defined by exactly this: a case is robust when the model's accuracy does not
drop under perturbation. Here the model reasons in full (unlike E12's immediate answers) on
the original problem and on a few rewrites of it. The fraction of rewrite samples whose
final answer matches the original's modal answer then estimates the decay directly,
without knowing the correct answer. If this predicts DeepSeek's labels, it can run at
evaluation time for every model, labels or not.

Rewrites (unlabelled, used only as inference-time probes, which the extra-data rule allows):

* `latex_to_plaintext`, `premise_reorder` from `perturb/transforms.py`;
* `typos`: two interior letters swapped in three words outside math;
* `distract`: one irrelevant sentence appended.

The last two mimic official perturbation types on purpose. They must never become labelled
training data (COMPETITION.md, extra-data rule), so they live here and not in `transforms.py`.

Prompts, sampling, and backends are E7's (`perturb/sample.py`), so agreement measured here
matches how self-generated labels are produced.

Run:  python datagen/extract_agreement.py --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out-dir features/agreement-deepseek
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "perturb"))

from answers import last_boxed, normalize_integer  # noqa: E402
from extract_features import DEFAULT_INSTANCES, assign_shards  # noqa: E402
from sample import HfBackend, VllmBackend, encode_prompt  # noqa: E402
from tokenizer_check import load_checked_tokenizer  # noqa: E402
from transforms import _MATH_SPAN, latex_to_plaintext, premise_reorder  # noqa: E402

_LONG_WORD = re.compile(r"\b[A-Za-z]{6,}\b")
DISTRACTORS = (
    "Note that this problem appeared in a regional competition held in the spring.",
    "A student once spent 45 minutes on a similar question.",
    "Recall that 2024 is divisible by 11.",
    "Some solvers find it helpful to draw a diagram first.",
    "The number 7 is the fourth prime number.",
)


def typos(problem: str, rng: random.Random) -> str | None:
    math = [m.span() for m in _MATH_SPAN.finditer(problem)]
    words = [m for m in _LONG_WORD.finditer(problem) if not any(a <= m.start() < b for a, b in math)]
    if len(words) < 3:
        return None
    pieces, last = [], 0
    for m in sorted(rng.sample(words, 3), key=lambda m: m.start()):
        word, i = m.group(), rng.randrange(1, len(m.group()) - 2)
        pieces += [problem[last : m.start()], word[:i] + word[i + 1] + word[i] + word[i + 2 :]]
        last = m.end()
    return "".join(pieces) + problem[last:]


def distract(problem: str, rng: random.Random) -> str:
    return f"{problem.rstrip()} {rng.choice(DISTRACTORS)}"


def rewrites(problem: str) -> dict[str, str]:
    """kind -> rewritten problem, deterministic per problem; kinds that do not apply are absent."""
    candidates = {
        "latex_to_plaintext": getattr(latex_to_plaintext(problem, 0, random.Random(problem)), "problem", None),
        "premise_reorder": getattr(premise_reorder(problem, 0, random.Random(problem)), "problem", None),
        "typos": typos(problem, random.Random(problem)),
        "distract": distract(problem, random.Random(problem)),
    }
    return {k: v for k, v in candidates.items() if v and v.strip() != problem.strip()}


def final_answer(text: str) -> str | None:
    """Normalised final boxed answer; integers compare as integers, anything else as tidy LaTeX."""
    raw = last_boxed(text.rsplit("</think>", 1)[-1]) or last_boxed(text)
    if raw is None:
        return None
    number = normalize_integer(raw)
    if number is not None:
        return str(number)
    tidy = re.sub(r"\\left|\\right|\\[!,;]|\s|\$", "", raw)
    return tidy.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").rstrip(".")


def signals(outputs: dict[str, dict]) -> dict:
    """Agreement features from {kind: {"answers", "tokens", "hit"}}; kind "original" is the anchor."""
    original = outputs["original"]
    answered = [a for a in original["answers"] if a is not None]
    modal = Counter(answered).most_common(1)[0][0] if answered else None
    kinds = [k for k in outputs if k != "original"]
    rewrite_answers = [a for k in kinds for a in outputs[k]["answers"]]
    rewrite_tokens = [t for k in kinds for t in outputs[k]["tokens"]]
    all_hits = [h for o in outputs.values() for h in o["hit"]]

    def agree(answers: list) -> float | None:
        return None if modal is None or not answers else sum(a == modal for a in answers) / len(answers)

    per_kind = {f"agree_{k}": agree(outputs[k]["answers"]) for k in kinds}
    defined = [v for v in per_kind.values() if v is not None]
    return {
        "n_rewrites": len(kinds),
        "orig_answered": len(answered) / len(original["answers"]),
        "orig_self_agree": agree(original["answers"]),
        "rewrite_agree": agree(rewrite_answers),
        "rewrite_min_agree": min(defined) if defined else None,
        "orig_tokens": sum(original["tokens"]) / len(original["tokens"]),
        "rewrite_token_ratio": (sum(rewrite_tokens) / len(rewrite_tokens)) / max(sum(original["tokens"]) / len(original["tokens"]), 1)
        if rewrite_tokens else None,
        "hit_length_frac": sum(all_hits) / len(all_hits),
        **per_kind,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--filter-model", action=argparse.BooleanOptionalAction, default=True,
                        help="keep only rows labelled for --model-id; --no-filter-model scores every train problem once")
    parser.add_argument("--k", type=int, default=2, help="samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--chunk", type=int, default=8, help="problems per generate call; also the flush interval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.max_model_len = args.max_model_len or args.max_new_tokens + 2048

    rows = pq.read_table(args.instances).to_pylist()
    if args.filter_model:
        rows = [r for r in rows if r["model_id"] == args.model_id]
        if not rows:
            raise SystemExit(f"no instances for {args.model_id!r}; use --no-filter-model")
    else:
        rows = list({r["problem_key"]: r for r in reversed(rows)}.values())
    assignment = assign_shards([r["problem_key"] for r in rows], args.num_shards, "roundrobin")
    shard = sorted((r for r in rows if assignment[r["problem_key"]] == args.shard_index), key=lambda r: r["problem_key"])
    shard = shard[: args.limit] if args.limit else shard

    out_dir = args.out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    done = set()
    if records_path.is_file():
        done = {json.loads(l)["problem_key"] for l in records_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in shard if r["problem_key"] not in done]
    print(f"{len(shard)} problems, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    started = time.time()
    tokenizer, tokenizer_info = load_checked_tokenizer(args.model_id, local_files_only=args.local_files_only)
    backend = VllmBackend(args, tokenizer) if args.backend == "vllm" else HfBackend(args, tokenizer)
    metadata = {
        "stage": "E14 test-time agreement",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "backend": backend.name,
        "backend_version": backend.version,
        "load_seconds": round(time.time() - started, 1),
        "platform": platform.platform(),
        **tokenizer_info,
    }
    generated = 0
    for start in range(0, len(todo), args.chunk):
        if args.max_hours and time.time() - started > args.max_hours * 3600:
            print(f"time budget of {args.max_hours} h reached; stopping (resumable)", flush=True)
            break
        chunk = todo[start : start + args.chunk]
        prompts = [(row, kind, text) for row in chunk
                   for kind, text in {"original": row["problem"], **rewrites(row["problem"])}.items()]
        completions = backend.generate([encode_prompt(tokenizer, text) for _, _, text in prompts])
        outputs: dict[str, dict[str, dict]] = {}
        for (row, kind, _), samples in zip(prompts, completions):
            outputs.setdefault(row["problem_key"], {})[kind] = {
                "answers": [final_answer(text) for text, _, _ in samples],
                "tokens": [n for _, n, _ in samples],
                "hit": [bool(h) for _, _, h in samples],
            }
            generated += sum(n for _, n, _ in samples)
        with records_path.open("a", encoding="utf-8") as handle:
            for row in chunk:
                record = {
                    "problem_key": row["problem_key"],
                    "scored_model_id": args.model_id,
                    "label_model_id": row["model_id"],
                    "is_robust": bool(row["is_robust"]),
                    "dataset_id": row["dataset_id"],
                    **signals(outputs[row["problem_key"]]),
                    "outputs": outputs[row["problem_key"]],
                }
                handle.write(json.dumps(record) + "\n")
        elapsed = time.time() - started
        print(f"  [{start + len(chunk)}/{len(todo)}] {generated / elapsed:.0f} tok/s overall, "
              f"{elapsed / (start + len(chunk)):.0f}s/problem", flush=True)
        metadata.update(generated_tokens=generated, seconds=round(elapsed, 1), problems_done=start + len(chunk))
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
