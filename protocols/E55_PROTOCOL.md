# E55: one-pass features for every candidate problem, and per-problem timing

Status: **pre-registered 16 October 2026, before any E55 code or run exists.** Planned code:
`datagen/build_e55_instances.py`, `scripts/e55_gpu_bundle.py`, `datagen/time_per_problem.py`. Any
change after a kernel is pushed is an amendment, dated in §7.

## Motivation

The collaborator's run (PLAN §12.6-12.8) will produce self-labels at the organizers' 100k budget for
a subset of the problems below. A per-problem predictor trained on them needs one-pass features for
the same (model, problem) pairs. E55 extracts those features now, on Kaggle T4s, so training can
start the day the labels arrive. It also measures what each candidate costs per call, because since
15 Oct every call carries one problem and the whole run gets 1 h 10 min (Finding 79). E54 made the
budget-path predictor a candidate (Finding 80); its cost is generation, so E55 times that too.

E55 reads **no labels** and has **no accuracy gate**.

## 1. Problems

The union, deduplicated by `problem_key` (whitespace-normalised SHA-1, `ingest_public.problem_key`):

| source | revision | problems |
| --- | --- | --- |
| `aimo-interp/train-main-v2` | `23791d9b` | 36 |
| `BeyondHsueh/ReliableMath`, `solve/aime.json` and `solve/amc.json` | `c059d6e6` | 30 + 83 |
| our olympiad pool, `data/olympiad_pool/pool.parquet` | as committed | 505 |

Every problem is paired with all four small-track models in an unlabelled instance table,
`data/e55/instances.parquet`.

## 2. Feature extraction (four kernels, one model each)

- `datagen/extract_familiarity.py`: bare-text mean NLL and the E23 statistics (the inputs of `e27k`).
- `datagen/extract_features.py`: Stage A, the probe-baseline system prompt and chat template,
  last/mean/max pooling at every layer; merged on Kaggle by `datagen/merge_features.py`, with the
  per-problem vectors deleted afterwards.
- `--dtype auto` (bf16, emulated on T4), the same setting as E23 and E52, so features stay comparable.
- Bundle and environment: E52's features kernel, unchanged apart from the instance table.

## 3. Per-problem timing (one kernel)

For each model, `datagen/time_per_problem.py` loads the checkpoint once and calls a one-problem
function repeatedly, as the evaluator now does:

1. **One-pass path:** mean NLL plus mean-pooled hidden states for 40 problems. Record load seconds,
   per-call seconds, and peak GPU memory.
2. **Budget-path path:** E37's four budgets (2k, 4k, 8k, 16k tokens with answer forcing, via E49's
   batched `transformers` extractor) for 5 problems. Record per-problem seconds and generated tokens.

The timing problems are the first 40 and the first 5 official problems by sorted `problem_key`.

## 4. Success criteria (infrastructure only)

- Features for at least 95% of (model, problem) pairs, zero non-finite activations, and provenance
  in every `metadata.json` (model, dtype, GPUs, library versions, prompt SHA-1).
- A timing table for all four models and both paths.

## 5. Decisions fixed now

- **E56** (budget-path features at scale, for the budget-path candidate) runs only if a per-problem
  submission using it would fit: projected time for **60 one-problem calls across the four models**
  on the evaluation card, taken as **one third of the measured T4 time**, must be at most **50 minutes**
  (leaving 20 of the 70 for loading and fallbacks). The one-third factor is an assumption about
  Blackwell against a T4 pair, stated here so it cannot be tuned later.
- Otherwise the budget path is dropped, and the per-problem predictor uses one-pass features only.

## 6. Not done

No label of any kind is joined to E55's outputs inside E55. No leaderboard upload. No threshold is
chosen here: thresholds belong to the predictor trained on the self-labels.

## 7. Amendments

**1 (17 Oct, before any E55 code or kernel existed): no separate timing kernel; §5 is decided by E49's
record.** E49 already measured E37's budget-path extraction with batched `transformers` on T4 pairs
(Finding 74, `kernels/runs/e49-official/result.json`): 734 s per row for Qwen, 859 for Skywork, 2,785
for DeepSeek and 1,568 for Olmo. Batched rows are faster than one-problem calls, so this is a lower
bound. Sixty one-problem calls take at least 24.8 h on T4; one third of that is **496 minutes against
the 50-minute rule**, so **E56 is not run** and the budget-path candidate of Finding 80 cannot ship on
cost. Re-timing five problems per model would spend about 10 GPU-hours to confirm a tenfold miss.

The one-pass timing of §3 comes from the extraction kernels themselves, which load each model once and
score problems one at a time, the evaluator's new pattern: `extract_familiarity.py` records seconds per
problem, and each kernel records every step's wall time (model load included) in `execution.json`.
`datagen/time_per_problem.py` is not written.
