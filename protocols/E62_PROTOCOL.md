# E62: does Qwen miss added contradictions on the problems it finds familiar?

Status: **pre-registered 21 October 2026, before any E62 variant or sample exists.** Report experiment: it
decides no submission and has no gate. Code: `datagen/perturb/e62_pool.py`, `datagen/perturb/premise_variants.py`
(generation), the E60 sampling kernels (`scripts/e58_gpu_bundle.py` pattern), `train/eval_e62.py` (to be written
before the first sample is read). Any change after the first sample is read is an amendment, dated in §8.

**Result, 31 Oct (Finding 95): positive. Spearman(mean NLL, detection rate) 0.256 [0.112, 0.386] on 177 problems;
within length terciles 0.142 [-0.015, 0.282].**

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

**1 (21 Oct, before any variant was kept): the models in §2.** `gemini-2.5-pro` answers 404, "no longer available
to new users", although the key's model list includes it, so the first run stopped after six failures with no
variant generated (one original had been checked by `gemini-2.5-flash`; that record is set aside in
`data/e62/superseded/`). Every Pro-class model the key lists (`gemini-3.1-pro-preview`, `gemini-pro-latest`,
`gemini-4-argon`) answers 429 at once: this key has no free quota for them. Flash-class models answer. So:

- generator: **`gemini-3.8-flash`**, the newest model the key can call, temperature 1.0;
- checker: **`gemini-3-flash-preview`**, a different model from the generator, temperature 0.

Everything else in §2 stands, including the 30-variant hand audit, which matters more with a Flash generator.

**2 (21 Oct, 00:20 UTC, before any variant was sampled): E60's models, and generation for eligible problems only.**
The free Gemini tier allows 20 requests per model per day (the 429 names the quota
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, value 20); the checker spent its 20 on the first four problems.
At about five requests a problem that is four problems a day. So:

- **generator and checker are E60's, unchanged:** `nvidia/nemotron-3-ultra-550b-a55b:free` writes,
  `minimax/minimax-m3.1-flash:free` checks, through OpenRouter's free tier with both keys (about 100 requests a day).
  This restores E60's cross-family check, which amendment 1 had given up. The 8 variants Gemini kept for 4 problems
  are set aside in `data/e62/superseded/`; one generator writes every variant used;
- **variants are generated only for problems the base phase found eligible**, in `problem_key` order, once base
  records exist (before that, in `problem_key` order over the whole pool). Ineligible problems are never sampled, so
  this changes no measurement; it spends the daily requests where they can count.

**3 (22 Oct, 13:30 UTC, before any contradiction was sampled): a new checker.** OpenRouter ended the free variant of
`minimax/minimax-m3.1-flash` on 22 Oct (HTTP 404). Of the free models listed that day, `google/gemma-4-31b-it:free` and
`gemma-4-26b-a4b-it:free` were rate-limited upstream and `thinkingmachines/inkling:free` refused API use;
`dots-studio/dots-3-note-preview:free` answered and, on a test problem built to be contradictory, said so. It checks
every variant from here on; it is from a different family than the generator, as E60 required. Variants already kept
under the MiniMax checker (problems done by 22 Oct) stay; the analysis reports the checker beside each problem, and
the hand audit of §2 samples both.

**4 (28 Oct, 08:30 UTC, before any variant from the new models exists and before any contradiction was sampled):
generation on Kaggle, regenerated for every eligible problem.** At OpenRouter's free rate (about 50 requests a day
after amendment 3) generation had covered 88 of the 300 eligible problems (`data/e62/eligible.parquet`) in a week,
while the fleet had more than 300 GPU-hours of unused quota this week. So:

- **generator: `Qwen/Qwen3-14B-AWQ`** (Qwen's own 4-bit release), thinking on, temperature 1.0; **checker:
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`**, quantised to 4 bits in flight by bitsandbytes, temperature 0.6, its
  model card's setting (greedy decoding repeats itself). Each runs under vLLM 0.20.0 on one T4 of a Kaggle pair, with
  a 16,384-token context and a 12,000-token completion budget; every other sampling setting is the model's own
  generation config. The checker is Llama weights from another family than the generator, as E60 required. The
  generator is a Qwen model, like the model sampled in §3: recorded as a limitation.
- **Fallback, fixed now:** if the server log of `e62g-smoke` (two problems, run first, its variants never used) shows
  the 14B cannot serve one 16k context on a T4, **`Qwen/Qwen3-8B-AWQ`** replaces it. That is decided from the server
  log alone; of the smoke's output only request counts, parse failures and keep counts are read.
  **Applied 28 Oct, 08:50 UTC:** the first `e62g-smoke` (Arifin) stopped at start-up, before any request. Its
  server log gives the 14B 1.68 GiB of KV cache on a T4 against the 2.5 GiB one 16k sequence needs (longest
  servable context 10,976 tokens), so **`Qwen/Qwen3-8B-AWQ` is the generator** and the smoke is rerun with it.
- **All 300 eligible problems are regenerated** by these two models, so one generator and one checker produce every
  variant used. The 160 OpenRouter variants (82 problems; Nemotron with the MiniMax or dots checker) move to
  `data/e62/superseded/openrouter/` and are never sampled. Of them only counts and keep rates were read.
- Prompts, insertion rule, keep rule, retry round and two variants a problem are unchanged
  (`datagen/perturb/premise_variants.py`, which gains local endpoints, sharding and these two settings). The hand
  audit of 30 kept variants (pass: at least 27 genuine contradictions) stands and comes before any sample is read.
- Kernels: `scripts/e62_gen_bundle.py`, `e62g-a` to `e62g-j`, each running four generator processes on disjoint
  slices of the eligible problems, on accounts with verified GPUs (never RSD, MUBAMAN or an account E67 holds).
  Sampling (§3) is unchanged. **§5's stop date moves to 4 November 2026, 12:00 UTC:** whatever is complete then is the
  population.
- **Rerun, 29 Oct (only counts read):** seven parts finished their 30 problems in 7.0-8.9 hours; `e62g-c`, `e62g-h`
  and `e62g-i` reached the 9-hour guard with 27, 26 and 29 done. Their 8 unfinished problems are rerun, with the same
  settings, by `e62g-rest` (four processes of two problems each); every finished record stands as written.
- **Sampling resumed, 30 Oct (only record counts read):** `e62v-qwen-b`, `-e`, `-f` and `-g` reached the 9-hour
  guard with 44 of 57, 32 of 57, 41 of 51 and 32 of 62 variants recorded; the other four slices finished. Each of
  the four is resumed from its own downloaded records (`-resume`, saving every 8 items instead of 16), sampling
  settings unchanged. `train/eval_e62.py` looked for contradiction records under `e62-qwen-*`, where there are none;
  it now reads `e62v-qwen-*`, a path fix made before any sample was read.
