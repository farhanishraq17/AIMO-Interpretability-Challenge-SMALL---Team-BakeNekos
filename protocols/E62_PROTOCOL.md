# E62: does Qwen miss added contradictions on the problems it finds familiar?

Status: **pre-registered 21 October 2026, before any E62 variant or sample exists.** Report experiment: it
decides no submission and has no gate. Code: `datagen/perturb/e62_pool.py`, `datagen/perturb/premise_variants.py`
(generation), the E60 sampling kernels (`scripts/e58_gpu_bundle.py` pattern), `train/eval_e62.py` (to be written
before the first sample is read). Any change after the first sample is read is an amendment, dated in §8.

## Motivation

`e27k`'s direction, familiar problems are robust, holds on DeepSeek's MATH rows and on the hidden validation set,
and reverses on the official olympiad rows, now with an interval below 0.5 (Finding 86). Finding 76 found what
decides olympiad labels: whether the model notices an added premise that contradicts the rest
(`expert_no_solution`). One mechanism would explain the reversal: a model that recognises a problem answers from
memory and skips the added premise. If so, **the more familiar a problem is to the model, the less often it flags
an added contradiction.** E62 measures that relation directly, on problems with no official label, so its result
is independent of the 34 rows every labeller has already been read on.

## 1. Problems

`data/e62/pool.parquet` from `datagen/perturb/e62_pool.py`: E55's problems that carry a reference answer in
`data/olympiad_pool/pool.parquet` and are not `train-main-v2` rows (497 problems: OlympiadBench, AMC, AIME), with
E55's problem text, in `problem_key` order. Generation and sampling both proceed in that order, so a run cut short by
quota still covers a hash-random prefix.

## 2. Variants

E60's generator and checker, prompts, insertion rule, keep rule and single retry round, unchanged except for the
models: **`gemini-2.5-pro`** writes two contradicting conditions per problem (temperature 1.0) and
**`gemini-2.5-flash`** checks each variant and the original blind (temperature 0). A variant is kept iff the checker
calls it contradictory and the original solvable. The checker is from the generator's family, unlike E60's; this is
recorded as a limitation, and 30 kept variants are audited by hand before any sample is read (pass: at least 27 are
genuine contradictions).

## 3. Sampling (Qwen3.5-4B only)

E60's recipe on T4 pairs, vLLM, one engine per GPU: the prompt that can decline (`--allow-no-solution`), the
organizers' decoding (temperature 1.0, top-k 40, top-p 0.95), **32k-token cap**.

1. Base: k = 2 on the original. Eligible iff both samples are correct.
2. Contradictions: k = 4 on each kept variant of every eligible problem, so up to 8 samples a problem.

A sample is **flagged** when `train/eval_e52.grade` scores it as abstaining. A truncated sample is not flagged, and
the cap-hit rate is reported.

## 4. Measurement

For every eligible problem with two kept variants: **detection rate** = flagged / 8. Its familiarity is Qwen's E55
`mean_nll` (lower = more familiar).

- **Primary:** Spearman correlation between `mean_nll` and detection rate, with a 95% problem-bootstrap interval
  (2,000 draws, seed 62). The memorisation mechanism predicts it is **positive** (less familiar, more detection).
- **Secondary:** the same within length terciles (character count), averaged with equal weight, to separate
  familiarity from length; and detection rate by `mean_nll` quintile.
- Reported whatever the sign or width. Fewer than 40 qualifying problems: reported as underpowered, no reading.

## 5. Compute and dates

Generation on the free Gemini tier, paced and resumable, from 21 Oct. Sampling on fresh Kaggle quota (Nipun,
Tamim2, Tamim3, Mazharul, Simanta01, Simanta02 this week; the whole fleet after the 24 Oct reset), in problem-key
slices, stopping by 30 Oct. Whatever is complete then is the population.

## 6. Not done

No submission uses E62. No official row is sampled. Nothing about the threshold, cap, k, grader or population
changes after a sample is read. Skywork, DeepSeek and Olmo are not sampled (Skywork never flags, DeepSeek rarely,
Olmo cannot be batched on T4s: Findings 77, 78, 85).

## 7. Report use

Positive interval: the mechanism behind the reversal has direct support. Interval spanning zero or negative: the
reversal stays unexplained, and the publication hypothesis (Finding 69) loses its best mechanism.

## 8. Amendments

None yet.
