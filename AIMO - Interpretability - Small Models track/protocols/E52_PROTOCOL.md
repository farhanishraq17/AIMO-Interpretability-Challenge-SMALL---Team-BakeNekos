# E52: do our models notice an impossible premise, and can one forward pass predict which ones?

Status: **closed 13 October 2026: gate 3 failed with no robust-like pair** (Qwen detection rate 0.021,
Skywork 0.000; PROGRESS Finding 77). The §3 audit agreed 30 of 30. `e27k` stays; E53 is not planned.
Pre-registered 13 October 2026, before any E52 record existed. Code:
`datagen/perturb/reliablemath_pool.py`, `scripts/e52_gpu_bundle.py`, `train/eval_e52.py` (grader
checked by `train/test_eval_e52.py`). Any change after the first record is read is an amendment,
dated and listed in §8.

## Motivation

Finding 76: on olympiad problems a model has solved, the five surface families almost never cost
25 points (0-3% of eligible rows in the organizers' sample). `expert_no_solution` does, in 43%, and
sets 10 of the 18 spurious labels there. It adds one premise that contradicts the rest, so the
right answer is that there is none. Our only large label source (DeepSeek, Hendrycks MATH) is
driven by the `domain` family instead, which is why nothing trained on it transferred.

No experiment in this repository has measured the deciding family on our models. E52 does, on
public data, and asks whether a signal cheap enough to ship predicts it.

## 1. Population and holdout hygiene

- **Problems:** `BeyondHsueh/ReliableMath` at revision `c059d6e6dd612c0dd21c2a4844d508b482c0d4b2`
  (CC-BY-4.0): the 75 AMC 2023 originals with an integer answer and at least one `contradict`
  rewrite, and all 164 of those rewrites (1-6 per problem). `remove` rewrites are excluded: the
  official family adds a premise rather than deleting one.
- **No official data is touched.** None of the 75 problems is a `train-main-v2` problem (checked
  by `problem_key`). The 34 official rows stay closed (Finding 74), and the 7 leaked validation
  cells of Finding 75 are never read.
- **Extra-data rule:** labelled data is allowed (organizers, 19 and 26 Sep; Finding 59). The
  dataset and its licence go in the report.

## 2. Models, decoding, and runs

| item | value |
| --- | --- |
| models | Qwen3.5-4B, Skywork-OR1-Math-7B. DeepSeek and Olmo-3-Think join only after gate 3 passes |
| decoding | the organizers' (FAQ): temperature 1.0, top-k 40, top-p 0.95; chat template; the E7 prompt suffix |
| base phase | k = 2 samples of each original, 16,384-token cap |
| rewrite phase | k = 2 samples of every contradict rewrite of each eligible problem, same cap |
| engine | vLLM 0.20.0 in its own venv after the CUDA linker check (E37's recipe). Qwen: one engine per T4, `--max-num-seqs 64`. Skywork: tensor parallel over both T4s |
| split | each model's problems are split in two by sorted `problem_key` (even and odd positions); each half is one kernel that runs both phases, so eligibility and rewrites live in the same session |
| features | one kernel, all four models: `extract_familiarity.py` (bare-text mean NLL, as `e27k`) and `extract_features.py` (Stage A: the probe-baseline system prompt and chat template, pooled at every layer), merged on Kaggle |

The 16k cap is far below the organizers' 100k. Reasoning models think at length on ill-posed
problems (MiP-Overthinking, arXiv 2504.06514), so some samples that would have abstained late will
truncate. They are counted as not detected, and the rate without them is reported beside it.

## 3. Grading (fixed; `train/eval_e52.py`)

One rewrite sample is:

- `truncated` if it used the whole cap: not detected;
- `answered` if it finished with an integer final answer (`answers.extract_answer`): not detected.
  The subset equal to the original's answer is reported as `kept_original`;
- `detected` if it finished with no integer answer and its ending matches the no-solution pattern
  in `eval_e52.NO_SOLUTION`;
- `other` otherwise: not detected, counted.

**Audit before reading any result:** 30 rewrite samples drawn at random (seed 0) are graded by hand
from their stored endings. If hand and script disagree on more than 3, the pattern is fixed and the
fix is recorded as an amendment before any table is read.

## 4. Labels

A (model, problem) pair is **eligible** iff both base samples solved the original. Its **detection
rate** is detected over all its rewrite samples. Against a base accuracy of 1.0 the official rule
becomes: `robust_like` iff rate >= 0.90, `spurious_like` iff rate <= 0.75.

## 5. Questions and gates

1. **Per-model detection rate** on eligible pairs, with a problem-bootstrap 95% CI. Reported, not
   gated. This is the first independent evidence on Finding 67's bet: the validation probes make
   Skywork spurious-leaning and Qwen robust-leaning.
2. **Is detection a property of the problem?** Spearman of the two models' rates and Cohen's kappa
   of `robust_like`, on problems eligible for both. Reported.
3. **Does one forward pass on the original predict `robust_like`?** Three signals:
   - **probe (primary):** per model, out-of-fold probabilities from 5-fold stratified CV (seed 0):
     one standardised L2 logistic regression (C = 0.01) per layer on mean-pooled Stage A states,
     averaged over every layer (Finding 11's layer ensemble);
   - mean NLL, fixed direction familiar = robust (`e27k`'s);
   - problem length, fixed direction shorter = robust.

   **Gate 3 passes iff** there are at least 60 labelled pairs, both classes have at least 5 pairs
   for each model used, and the probe's pooled AUC has a problem-bootstrap 95% CI lower bound
   above 0.5. The two fixed-direction scalars are reported, not gated, so there is one test.

## 6. Decisions

- **Gate 3 passes:** E53 is planned, as a separate pre-registration: DeepSeek and Olmo are added,
  the probe is fitted on all E52 pairs, and a one-pass submission is built with `e27k`'s ranking as
  the per-batch fallback. It gets one leaderboard read and the Docker test before the 27 October
  freeze.
- **Gate 3 fails:** `e27k` is the submission. Questions 1 and 2 and the failed gate go in the report
  beside Finding 76.
- **Either way, question 1 informs the counts decision** recorded in PLAN §12.3 and Finding 67. It
  does not change the counts by itself.

## 7. Not done

- No threshold, layer set, C, rewrite subset or grading pattern is chosen after reading results.
- No leaderboard submission comes out of E52.
- The 34 official rows are not used for selection or scoring.

## 8. Amendments

None. One grader false positive was found after the audit (a Qwen sample that answered 2.4 and
called the dimensions "physically impossible"); it is reported in Finding 77 and the script is unchanged.

## 9. E52b: Olmo-3-Think on the same problems, question 1 only

Status: **stopped 14 October 2026 with no rewrite sample** (Finding 78): vLLM 0.20 died with
`Cannot get 1 free blocks from the pool` in every kernel. The reading below was not made.

Pre-registered 14 October 2026, after E52 closed and before any Olmo record exists.

**Why.** `e27k` tilts two models: Skywork and Olmo-3-Think each call one fewer problem robust
(Finding 60). E52 gave the Skywork tilt mechanistic support (Finding 77): on the deciding family,
Skywork answers through the contradiction. Olmo has no such evidence, and the official split puts
it at 7 of 9 robust, against the tilt (Finding 67).

**What is the same.** The 75 problems and 164 rewrites, decoding, k, eligibility, the grader, the
labels and every Q1 statistic (§1-§4). The audit of §3 is not repeated: the grader is a fixed
function of the ending, and its audit covered the same code.

**What differs, fixed now:**

- Olmo's problems are split four ways (every fourth sorted `problem_key`), one kernel each, because
  Olmo runs few sequences at once on a T4 pair (Finding 26).
- Engine: vLLM 0.20.0 with `transformers` 5.5.3, the pin E37's config check verified for Olmo's
  rope parameters; tensor parallel 2, `--gpu-memory-utilization 0.95`, 8 problems per call.
- Time guards of 2.5 h (base) and 4 h (rewrites) per kernel bound the quota. Whatever finishes is
  reported with its count.
- **Cap:** 16k, as E52. If vLLM cannot allocate the cache for it (Finding 26 measured 12k as the
  most a T4 pair held under an older vLLM), every Olmo kernel reruns at 12k and the Q1 table states
  the cap. Truncation is judged against each run's own cap (`eval_e52.run_cap`).

**Analysis.** `train/eval_e52.py` adds Olmo to the Q1 and per-model tables. Q2 and Q3 stay on the
two gate models; gate 3 stays closed.

**Reading, fixed now.** No gate and no automatic change to the submission:

- **Olmo like Skywork** (detection rate on eligible problems at most 0.05, and at least 60% of
  rewrite samples answered): the Olmo tilt gains the same mechanistic support as Skywork's.
- **Olmo like Qwen or better** (detection rate excluding truncated samples at least 0.107, Qwen's
  E52 value): it does not, and the freeze note records that the Olmo tilt rests on the validation
  balance alone.
- Anything between is reported as inconclusive.

**Amendment, 14 October, before any Olmo record existed.** The pilot (`e52-olmo-a` on Mashrafi)
failed at engine start: weights took 6.86 GiB a GPU and vLLM's CUDA-graph memory profiling reserved
4.72 GiB more, leaving 1.61 GiB of cache, too little for one 16k request ("EngineCore failed to
start"). Every Olmo kernel therefore adds `--enforce-eager`, which skips graph capture and frees that
memory. It is an engine setting: decoding, k, cap and grading are unchanged. The cap stays 16k. The
12k fallback above was written for a cache that cannot hold the cap at all, and lowering the cap
would not have fixed this failure; it still applies if eager mode cannot start at 16k.

**Second amendment, 14 October, still before any Olmo record.** With eager mode the engine
allocated the cache but ran out of memory in the sampler warm-up (394 MiB requested, 111 MiB free at
`--gpu-memory-utilization 0.95`; the warm-up sizes its buffers for vLLM's default 256 sequences over
Olmo's 100k vocabulary). Olmo kernels now use utilisation 0.92 and `--max-num-seqs 16`; the cache
holds only a few 16k sequences anyway. The 12k rule is not invoked: this failure is warm-up headroom,
which the cap does not change. Engine settings only; decoding, cap and grading are unchanged.

**Third amendment, 14 October, still before any Olmo record.** Pilot 3 failed the same way, and its
traceback places the 394 MiB request in FlashInfer's prefill workspace (`flashinfer.py`,
`_get_workspace_buffer`), the attention backend vLLM uses on a T4. It is allocated after the cache is
sized and is not in vLLM's memory budget, so the second amendment's diagnosis (sampler warm-up) was
wrong. Utilisation drops to 0.85, leaving room for the workspace while keeping about 5 GiB of cache
a GPU. Engine settings only; decoding, cap and grading are unchanged.
