# E49: E37's frozen budget-path score, generated on the evaluation path, scored once on the official rows

Status: **pre-registered 6 October 2026, before any E49 extraction.** No E49 feature exists, and no
one has computed an official-row accuracy for the budget-path signal under any protocol. §6.4
needs the project owner's sign-off before the first extraction is read.

## Motivation

E36/E37 hold the only signal in this repository that transferred across independent sources:

| step | population | result |
| --- | --- | --- |
| E36 holdout | Putnam-AXIOM, 206 configurations | AUC **0.765 [0.687, 0.839]**; 0.68–0.81 within every model size |
| E37 confirmation | GSM-Plus, 793 configurations | AUC **0.673 [0.599, 0.744]**; every pre-registered gate passed |

E37's official transfer was then closed by its runtime gate: 7,526 s for Qwen and Skywork alone.
That figure was measured under vLLM with at most two sequences in flight on a T4 pair
(`scripts/e37_gpu_bundle.py`: `--max-num-seqs 2 --chunk 2`). The evaluation image ships no vLLM
(`getting-started/Dockerfile.competition`), so the figure described neither the path a submission
takes nor the batching one 96 GB card allows (`REVIEW_GENERALIZATION.md` §3). The official
accuracy was never measured: 23 of 34 rows were recovered and none was scored.

**E37 stays closed as recorded.** E49 is a new experiment with its own gates, written after E37's
runtime failure but before anyone has seen this signal's official-row accuracy. Read its result
with that history in view.

It changes three things:

1. **Extraction:** batched `transformers` `generate`, the only engine a submission can use (§3).
2. **Runtime is recorded, not gated** (§4). The submission E49 would feed is anytime: it cannot
   exceed the time limit, and runtime matters only on an accuracy tie.
3. **Gates sized to the data** (§5). E37 asked for a template-bootstrap lower bound above 0.5, and
   20 labelled problems cannot meet that. E49's gates are calibrated by simulation, and their
   error rates are stated.

Everything else is E37's, unchanged: the predictor, its 18 features, the four budgets, the
prompt, the answer forcing and the decision rule.

## 1. Holdout hygiene

- **The existing E37 official features are never scored, by anyone, under any protocol.** They
  sit in Kaggle outputs: Qwen 12/12, Skywork 3/3, DeepSeek 8/10. E49 re-extracts all 34 rows with
  its own extractor.
- The 34 labelled small-track `train-main-v2` rows are scored **exactly once**, all four models
  together, by `train/eval_e49_official.py` as committed with this protocol. The result records
  the commit.
- Before that one scoring run, extraction may be inspected only through `metadata.json`,
  `runner.log`, `execution.json` and token counts. Never join `features.jsonl` to labels.
- **Frozen once this protocol is committed:** budgets, features, model coefficients, prompt,
  answer forcing, decision rule, rank fraction, gates and thresholds.
  - Batch settings (`--max-batch-tokens`, `--group-size`, account) may change only for memory or
    session limits. They never read labels, and every change is recorded in `metadata.json`.
- Smoke runs (`--smoke-divisor` > 1, or `--no-filter-model`) may be used to calibrate memory and
  throughput on real checkpoints. The scorer refuses their output.

## 2. Frozen predictor and decision

- **Training data:** E37's four MATH-Perturb shards from
  `ssubhnil/Qwen3-Math-Eval@f64c82f5ce9249155ad7c70991a9711d92963cce`.
- **Training:** `build_examples(..., "math_perturb")`, then `feature_columns`, then `make_model()`:
  median imputation, standardisation, class-balanced logistic regression with `C = 1`, seed 36.
  All three come from `train/eval_e36_budget_path.py`, unchanged.
- **Amendment, 7 Oct, before any extraction:** the archive now answers 401 to anonymous
  downloads, so `train/prepare_e37_model.py` cannot fetch it.
  - E35 kept the same four files in `runs/e35-external/math_perturb/`. Their SHA-256 match the
    confirmation run's manifest (`kernels/runs/e37-confirmation-Arifin/result.json`).
  - The scorer refuses any other bytes (`verify_archive`).
  - Reading those shards whole gives the same 1,001 training examples (769 robust) and
    bit-identical coefficients as `load_partition`'s filtered read. That was checked on 7 Oct.
- **Decision:** within each `(model_id, reasoning_effort)` batch, the higher-scoring `ceil(n/2)`
  rows are robust, using `train/eval_e37_official.py` `half_rank`. There are no per-model tilts
  (`REVIEW_GENERALIZATION.md` §4.3).

## 3. Extraction

`datagen/extract_budget_path_hf.py`, checked by `datagen/test_extract_budget_path_hf.py` and a
CPU smoke run on `hf-internal-testing/tiny-random-LlamaForCausalLM`.

- **Prompt:** `sample.encode_prompt` (the problem plus E37's `PROMPT_SUFFIX`, chat template, a
  checked tokenizer).
- **Decoding:** greedy. Budgets 2,048, 4,096, 8,192 and 16,384 tokens, each an independent
  generation (E38 closed prefix reuse). Every (problem, budget) pair is one row of a left-padded
  batch, and a per-row stopping criterion ends each row at its own budget.
- **Answer forcing:** E37's `encode_forced_prompt` and `FORCE_REQUEST`, at most 512 greedy
  tokens, whenever a trace hits its budget or has no boxed answer.
- **Features:** E37's `trace_features`, unchanged. The record schema is E37's, plus
  `label_model_id`.
- **Preflights (abort on failure):**
  - the first prompt's logits are finite;
  - the shortest prompt decoded alone and inside a batch padded to the longest prompt agree on
    at least 8 of their first 32 tokens.

**Fleet settings** (T4 pair, fp16 as E37's official run used). The batch cap is in tokens:
sequences × (prompt width + budget).

| model | KV cache per token | `--max-batch-tokens` | `--group-size` |
| --- | --- | --- | --- |
| Qwen3.5-4B | ~32 KB (8 of 32 layers full attention) | 400,000 | all |
| Skywork-OR1-Math-7B | ~57 KB | 150,000 | all |
| Olmo-3-7B-Think | ~128 KB + a 4,096-token window on 24 layers | 50,000 | 3 |
| DeepSeek-R1-0528-Qwen3-8B | ~147 KB | 60,000 | 5 |

- **Amendment, 7 Oct, during extraction (execution only, recorded before any slice runs):**
  - A Kaggle session's output can only be downloaded once the whole session ends. DeepSeek's first
    run (TASHIN) projects to about 8.5 h and Olmo's (Labib) to about 5 h.
  - Both are therefore re-run as row slices on parallel kernels: DeepSeek as 5 × 2 rows and Olmo
    as 3 × 3 rows. These are the `e49-<model>-r<offset>` jobs in `scripts/e49_gpu_bundle.py`, using
    the batch caps the first runs held.
  - **For DeepSeek and Olmo, the scored features are the slices'.** The first runs are fallbacks,
    used for a row only if its slice fails. Qwen and Skywork keep their first runs.
  - The extractor now also sizes calls from a memory model measured on the GPU at start-up. If
    that calibration fails, the caps alone apply.
  - Nothing here touches §1's frozen list.
- **Launch:** `python scripts/kaggle_followup.py push e49-<qwen|skywork|olmo|deepseek> --account <ALIAS>`.
  It refuses read-only accounts (RSD). Do not use the excluded `farhanishraqq` GPU account.
- **Kernel environment:** the `Dockerfile.competition` pins with Kaggle's own torch, no vLLM.
  The process guard stops at 11 h, and the extractor starts no new group after 10 h.
- **Precision caveat:** greedy traces depend on dtype (Finding 13: 0 of 8 traces identical
  between fp16 and bf16). The evaluation card runs bf16. The features are aggregates over traces,
  so how much this moves the score is unknown. §6.2 and §6.3 are where it would show.

## 4. Runtime: recorded, not gated

- **Why not gated:** §6.1's submission checks a wall-clock deadline (`generate(max_time=...)`)
  and falls back per batch, so it cannot exceed the limit. No fleet account has hardware like the
  evaluation card, and E37 showed that a T4 measurement stands in for it badly.
- **Recorded per model** (`runtime_record` in the result): rows, generated tokens, tokens per
  row, generation seconds, tokens/s at every batch shape, peak memory, out-of-memory retries.
- **Projection, stated as one:** test cost ≈ tokens per row × test rows ÷ evaluation-card
  throughput. The record supplies the first factor; nobody here has measured the third.

## 5. Gates

Computed by `train/eval_e49_official.py`:

1. **Coverage:** all 34 labelled small-track rows and all four models, from clean E49 runs.
2. **Problem-level AUC ≥ 0.65.** This is E37's definition: row scores averaged per problem, labels
   pooled over every labelled configuration. 20 problems carry small-track rows.
3. **Half-rank balanced accuracy > 0.55** over the 34 rows.

**Error rates** (`python train/eval_e49_official.py --simulate-gates`: binormal row scores on
these exact rows, batches and label pooling, 4,000 draws each):

| true row AUC | P(pass gates 2 and 3) |
| --- | --- |
| 0.50 (no signal) | **0.114** |
| 0.60 | 0.365 |
| 0.65 | 0.525 |
| 0.70 | 0.708 |
| 0.75 | 0.845 |

About one useless ranking in nine passes. This is not hypothetical: a plumbing dry run of the
scorer on random synthetic features passed both gates, at problem AUC 0.80. So passing §5
authorizes §6, and §6.3 is a second, independent filter. Neither alone is evidence of a working
model.

**Reported, not gated:**

- the template-bootstrap CI;
- row AUC and per-model AUCs;
- E37's original gates, for the record;
- the shipped familiarity ranking on the same rows and batches. It scores 0.353 accuracy, and
  any half split beats that here, so it would be a vacuous gate.

## 6. If §5 passes

1. **Build an anytime `e49` ZIP.** For each batch:
   - compute `e27k`'s familiarity ranking first (seconds);
   - run the budget-path extraction with `generate(max_time=...)` set to the time left before a
     3,000 s deadline, counted from the first `are_robust` call;
   - use the half-rank E49 decision if all four budgets finished. Otherwise use the familiarity
     ranking at `ceil(n/2)`. On any exception, use E11 v2's constants.
   - Use bf16 where the card supports it natively, otherwise fp16.
2. **Checks before upload:**
   - `submission/docker_load_test.py` (contract);
   - a Kaggle GPU check of the ZIP on the 34 official problems. Its decisions are compared with
     the §5 decisions. More than 4 of 34 disagreeing blocks the upload pending investigation.
3. **One Codabench upload,** a sanity check rather than a selection. It must show coverage 1.0,
   0 invalid predictions, and a score of at least 7/14 (0.50, the all-False score). A lower score
   means E49 does not freeze. Investigate; do not tune.
4. **Freeze rule for 27 October (needs owner sign-off before extraction is read).** If 6.1–6.3
   pass, freeze `e49`. Its leaderboard score is recorded but does not have to beat `e27k`'s 0.86:
   that score is calibrated to these 14 cases (Findings 63, 67), and E49 was chosen on independent
   data. Otherwise `e27k` stays.

    Sign-off: ______________________  date: __________

## 7. If §5 fails

E49 closes and `e27k` remains the incumbent. Record the result as a Finding, with the runtime
record. Nothing in §1's frozen list changes, and the old E37 features stay unscored.

## 8. Schedule

| when | step |
| --- | --- |
| 6–8 Oct | commit this protocol and its code; optional smoke calibration on one real checkpoint |
| 8–12 Oct | four extractions (expected under 3 GPU-hours each on a T4 pair) |
| by 13 Oct | the one scoring run, `--math-dir runs/e35-external/math_perturb` |
| 14–24 Oct | §6 if it passes |
| 27 Oct | freeze under §6.4 |
