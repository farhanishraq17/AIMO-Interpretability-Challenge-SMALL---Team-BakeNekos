# Review: how each experiment was built, and a more general final model

6 October 2026. A full pass over the small-track repository: every experiment's implementation,
what limited its accuracy, and what a final model that generalises past the 14-case leaderboard
should look like. New numbers come from `train/review_generalization.py` (CPU, about 3 minutes on
first run); everything else cites the Finding or `RESULTS.md` section it came from.

## 0. Summary

1. **`e27k`'s 0.86 is mostly calibration to the validation set, and the private test is a
   different draw.** About one of its two cases over `e27e` is count arithmetic (Finding 63), and
   the ranking under it sits at p = 0.055. On a ~30-problem test the per-model tilts shift at most
   one call per model. What decides the private score is the test prevalence and whether the
   ranking keeps its direction on olympiad problems (§2.4).
2. **The shipped ranking is mostly problem length.** Mean NLL correlates −0.39 to −0.80 with
   token count. On DeepSeek's MATH rows, length alone (AUC 0.744) beats the NLL ranking
   (0.66–0.68), and length-residualised NLL keeps only 0.587 (§2.1–2.2).
3. **Length changes sign between populations, and Finding 46 reads it backwards.** On the
   official problems, the raw AUC for "longer is robust" is **0.335**. E22's 0.635 is a fitted
   model whose coefficient is **negative** (−0.84), so shorter problems are the robust ones there.
   Shorter also leans robust on MATH-Perturb (0.437) and Putnam-AXIOM (0.425). DeepSeek's MATH
   rows are the outlier. Most of the "unresolved contradiction" of `PLAN.md` §12.3 is this length
   axis.
4. **On two populations nobody had scored familiarity against, it is near chance:** MATH-Perturb
   0.548 and Putnam-AXIOM 0.529. Those labels already sit in E35's data. A 0.6B proxy reproduces
   the competition models' rankings (Spearman 0.84–0.90).
5. **The one signal in the repository that transferred across independent sources was rejected
   on a runtime figure that does not describe the evaluation machine.** E36/E37's budget-path
   score reached:
   - **Putnam-AXIOM:** AUC **0.765 [0.687, 0.839]**, 0.68–0.81 within each model size;
   - **GSM-Plus:** AUC **0.673 [0.599, 0.744]**.

   Its runtime gate was measured with **at most two sequences in flight on a T4 pair**
   (`scripts/e37_gpu_bundle.py:54`: `--max-num-seqs 2 --chunk 2`). Its official-row accuracy has
   never been measured.
6. **Recommendation:** pre-register E49, which is E37's frozen predictor with batched
   generation, scored once on all 34 official rows. Package it as an *anytime* submission that
   falls back to `e27k`'s ranking per model when its wall-clock deadline approaches (§4).

## 1. What ships, mechanically

`submission/e27d-familiarity/solution.py` with `ROBUST_HALF = "familiar"` and
`TILT = {Skywork: -1, Olmo: -1}`, built by `submission/make_e27.py`:

1. Load the scored model in bf16.
2. Run one forward pass per bare problem (no chat template, truncated at 2,048 tokens) and take
   the mean token NLL.
3. Call the `ceil(n/2) + tilt` lowest-NLL problems in the batch robust.
4. On any failure, fall back to constants.

The ingestion program calls `are_robust` **once per `model_id` with that model's whole batch**
(`components/ingestion_program/ingestion.py:130-143`), so the test batch is every test case for
that model.

The implementation itself is sound. E39 showed it is deterministic, insensitive to fp16 versus
bf16, and fast (334 s for 4 × 36 on T4s). E42 showed it runs in the competition image. The
problems are what it measures and how its counts were chosen, not code defects.

## 2. New measurements (`train/review_generalization.py`)

### 2.1 Mean NLL is mostly problem length

| scoring model | Spearman(mean NLL, tokens), official | same, MATH | Spearman with the 0.6B proxy |
| --- | --- | --- | --- |
| Qwen3.5-4B | −0.58 | −0.39 | 0.892 |
| Skywork-OR1-Math-7B | −0.80 | −0.77 | 0.836 |
| Olmo-3-7B-Think | −0.68 | −0.55 | 0.901 |
| DeepSeek-R1-0528-Qwen3-8B | −0.70 | −0.48 | 0.865 |

The last column extends Finding 68: a 0.6B model that is not in the evaluation cache reproduces
each checkpoint's ranking. The proxy's agreement with the four models' consensus is 0.935. The
model-specific content of the ranking is small.

### 2.2 The shipped direction on every labelled population we hold

AUC oriented so that above 0.5 means the shipped direction (longer, or more familiar, is robust).
All five rows are scored by `Qwen/Qwen3-0.6B`. On the two rows that also have competition-model
scores, the proxy matches them: MATH 0.681 against DeepSeek's 0.662, and official 0.35–0.41
against 0.31–0.36.

| population | n | robust | longer | familiar (mean NLL) | suffix NLL | NLL, length-residualised | NLL within length quintiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek, Hendrycks MATH | 142 | 0.27 | **0.744** | 0.681 | 0.590 | 0.587 | 0.597 |
| official, problem level | 33 | 0.61 | **0.335** | 0.350 | 0.392 | 0.481 | 0.391 |
| official, small-track rows | 34 | 0.71 | 0.471 | 0.412 | 0.379 | 0.512 | 0.270 |
| E35 MATH-Perturb (Qwen3 labels) | 233 | 0.82 | 0.437 | 0.548 | 0.536 | 0.551 | 0.577 |
| E35 Putnam-AXIOM (Qwen3 labels) | 48 | 0.48 | 0.425 | 0.529 | 0.506 | 0.525 | 0.509 |

How to read it:

- **Length is population-specific.** It is strong on the population the policy was developed on
  and reversed on the other three.
- **The length-free part of familiarity is weak but mostly positive:** 0.51–0.59 on four of five
  rows. On official rows it is negative, but only when length is controlled by quintile bins.
- The E35 labels come from Qwen3 models under MATH-Perturb and Putnam variations, which are not
  the official families. Treat those two rows as evidence about *problems*, not as replicas of
  the target. That relies on Findings 38 and 68.

### 2.3 Finding 46's length signal has the opposite sign

| measurement on the 33 official problems | value |
| --- | --- |
| raw AUC, longer = robust | **0.335** |
| E22's leave-one-template-out fitted model | 0.635 |
| coefficient on log length, all 33 problems | **−0.844** |

Finding 46 says length "points the same way" as on MATH. It does not. The fitted model learned
"shorter is robust". `e27c`, which shipped that fitted model, scored 8/14. `e27e`, whose ranking
leans "longer is robust", scored 10/14. So the leaderboard and the official rows disagree on the
length axis. That is the same contradiction §12.3 records, seen from the side where it is
largest.

### 2.4 What the robust count is worth at test prevalence

The policy calls the top fraction `c` of a batch robust. The table gives expected accuracy under
a binormal ranking of the stated AUC (§4 of the script).

| ranking AUC | test prevalence | c = 0.5 | c = 0.6 | c = 0.7 | all `True` |
| --- | --- | --- | --- | --- | --- |
| 0.4 | 0.5 / 0.7 | 0.429 / 0.440 | 0.431 / 0.483 | 0.438 / 0.530 | 0.500 / 0.700 |
| 0.5 | 0.5 / 0.7 | 0.500 / 0.500 | 0.500 / 0.540 | 0.500 / 0.580 | 0.500 / 0.700 |
| 0.6 | 0.5 / 0.7 | **0.571** / 0.560 | 0.569 / 0.599 | 0.562 / 0.634 | 0.500 / **0.700** |
| 0.7 | 0.5 / 0.7 | **0.645** / 0.620 | 0.640 / 0.661 | 0.624 / 0.692 | 0.500 / **0.700** |

The two sources disagree about prevalence: the leaderboard is 50% robust and `train-main-v2`'s
small-track rows are 71% robust. If the test resembles the train split, half-splitting each batch
loses to all-`True` at any AUC up to 0.7. If the familiarity ranking reverses there as it does on
the official rows (AUC about 0.4), half-splitting scores below every constant.

The only statement the organizers made about evaluation prevalence is the `train-main-v2`
card's: "the validation set is close to balanced. Calibrating a decision threshold on this
prevalence may not transfer." That supports a half split, so keep `c = 0.5`. But the main
lever is then the ranking's AUC on olympiad problems, and the familiarity family has no
out-of-sample evidence above about 0.55 there.

## 3. Experiment-by-experiment

**Verdict** is about the implementation. **Accuracy lever** says what, if anything, would have
made the measurement more accurate or more general.

| Exp. | How it was built | Verdict | Accuracy lever |
| --- | --- | --- | --- |
| E0 / E0b | Char tf-idf (3–5, `char_wb`) logistic regression, 5 folds grouped by `problem_key`; within-dataset confound check | Clean | None; the signal is MATH-specific (Findings 30, 39) |
| E1 → E1b | One forward pass; last/mean/max pooling at all 37 layers; L2 probes at C = 0.01; layer ensembles to avoid selection; nested selection reported | Clean after the tokenizer fix (Finding 14) | Did not transfer (official AUC 0.476, Finding 32) |
| E2 → E2f | Greedy generation, 4k cap, fp16, the starter kit's 14 features | Valid but confounded: 55% of traces hit the cap, so the features largely measure truncation | E36's budget-path features are the well-designed successor, and they transferred |
| E3, E13 | Plumbing | Clean | — |
| E4 | Teacher-forced trace states; hook pooling matched exactly to the final-norm convention (Finding 17) | Careful work | Signal sits in the prompt state |
| E7 / e7b / E15–E20 / E24 / E34a | vLLM fp16, k = 4, temperature 0.6, caps 8k–24k, eligibility = 4/4 correct; `answer_modulus` variants (E34a: two surface families) | Pipeline clean, but its settings are not the official ones | The official FAQ uses temperature 1.0, top-k 40, top-p 0.95, up to 100k tokens, and labels by `max_drop` without a 4/4 gate (Finding 57). A 16k cap truncated 58–62% of Qwen's samples. `answer_modulus` changes the requested output, which official families never do. G-V and G-V2 measured *this* pipeline, not official-family self-labelling |
| E10 | Nested fusion: inner-fold standardisation and thresholds, fused combinations fixed in advance, paired CIs | Clean | — |
| E11 | Refit on all data; arrays instead of pickled probes; GPU-verified to 1.3e-9 | Clean | Its threshold was learned at 27% robust, so it is an absolute, prior-bound cut |
| E12 | Next-token scalars and problem NLL; within-quintile control (after the Finding 24 fix) | Clean | Showed early that the signals are length proxies |
| E14 | DeepSeek, k = 2, 8k cap, four mechanical rewrites | Clean | The rewrites barely bite (all samples agree for 78.5% of problems), so this measured weak perturbations. Rewrites from an LLM in the official families would test the idea properly. Its "infeasible" verdict rests on E47's T4 rate (below) |
| E21 | E11 on the 10 official DeepSeek rows; leave-one-model-out on next-token scalars | Clean | n = 3–12 per model |
| E22 | Problem-level harness: labels pooled across all configurations (gpt-oss supplies 37 of 82 rows), leave-one-template-out, template bootstrap | Clean harness | **Interpretation error on length** (§2.3). Pooling with gpt-oss is defensible given 83/85 agreement, but a small-track-rows view should sit beside it |
| E23 | One bare-text forward pass; seven membership statistics; orientation fixed on MATH within-length | Clean and well pre-registered | Mean NLL is −0.4 to −0.8 correlated with length (§2.1) |
| E27 / E28 / E29 | Batch rank by an E23 statistic; counts from constant probes | Implementation clean | The ranking is mostly length (§2.2). Min-K gap, chosen on official rows (offline 0.808), scored 4/14: a second case where official rows and the leaderboard disagree |
| E30 | 12 norm/cosine trajectory scalars; max-T correction | Clean | Reversed on MATH (0.348) |
| E31 / E32 | Last-token state of the templated prompt, original against up to four rewrites; five pre-registered reductions | Clean, with one measurement weakness: each score averages over **whichever rewrites apply** (official rows 2–4 kinds, MATH mostly 2), so different rows mix different kinds | Per-kind standardisation, or one fixed kind set, would compare like with like. It is the only one-pass signal with the **same sign on both populations** (official 0.735, MATH 0.545), and it was never read on the leaderboard |
| E35 | Text model trained on external labels; source-disjoint holdout | Clean | Fails (0.609). Its labels are reused in §2.2 |
| **E36 / E37** | Greedy traces at four budgets; 18 architecture-independent features (budget agreement, natural completion, growth, markers); frozen model trained on MATH-Perturb | **Clean, and the best result in the repo** (Putnam 0.765, GSM-Plus 0.673, every model size ≥ 0.63) | Killed by a runtime gate measured with at most two sequences in flight on T4s, under vLLM, which the evaluation image does not contain. Official accuracy never measured (23/34 rows recovered) |
| E38 | Can the archive's shorter traces be replaced by prefixes of the longest one? | Clean, and it failed | It concerns the archive's generation process. A prefix scheme for our own inference would need its own check |
| E39 – E44 | GPU audit, batch arithmetic, re-probe, Docker test, prior sensitivity, cross-model ranking | Clean. E40 and E43 are correct arithmetic | E44's conclusion reproduced with a 0.6B proxy (§2.1) |
| E45 / E46 / E48 | Publication split, continuous `max_drop`, power | Clean and honestly framed | — |
| **E47** | Cost = forward passes × 2.1 s + tokens ÷ **90 tok/s**, the aggregate T4 rate over 8 streams | Its conclusion is overstated | One 96 GB RTX PRO 6000 with 16–48 sequences in flight is plausibly 10–30× that rate (an estimate, not a measurement). "Infeasible at any accuracy" for E14 and E36/E37 does not follow. What does follow is that these methods need batched generation and a deadline guard |

## 4. Toward a more general final model

### 4.1 E49: give the transferring signal a fair runtime test (highest value)

**Now pre-registered in `protocols/E49_PROTOCOL.md`**, with code in
`datagen/extract_budget_path_hf.py`, `train/eval_e49_official.py` and `scripts/e49_gpu_bundle.py`.
The protocol supersedes the sketch below where they differ:

- the gates became problem AUC ≥ 0.65 and balanced accuracy > 0.55, after simulation put the
  sketch's false-pass rate at 23%;
- the familiarity comparison is reported, not gated, because any half split beats it on these
  rows.

The original sketch:

1. **Frozen and unchanged:** E37's predictor (`train/prepare_e37_model.py`), its 18 features,
   four budgets, prompt suffix, answer-forcing request, and within-model half-rank decisions.
2. **Coverage first.** Finish the official extraction: the 2 remaining DeepSeek rows and all 9
   Olmo rows. Batch everything: every problem × every budget submitted at once, with
   `max_num_seqs` as large as memory allows (not 2).
   - Qwen (12/12) and Skywork (3/3) features already exist.
   - **Decide now that all 34 rows are scored together once.** Peeking at the 15 complete rows
     first would spend the only untouched olympiad holdout piecemeal.
3. **Accuracy gate on the 34 official rows**, fixed in advance. At n = 34 a CI gate cannot pass,
   so the gate is comparative:
   - point AUC above 0.5;
   - half-rank balanced accuracy above always-robust (0.50);
   - accuracy above the familiarity split on the same rows (0.353, Finding 58).
4. **Runtime is measured on the path that ships.** The evaluation image has no vLLM
   (`getting-started/Dockerfile.competition`), so port extraction to `transformers` `generate`
   with left padding and a batch per budget.
   - Measure tokens/s on a T4 pair at that batch, then on the fastest card the fleet can reach.
   - Olmo-3-Think keeps a full KV cache: 4.5 GiB per GPU for one 18k-token sequence at TP = 2
     (Finding 26). It is the binding model; budget its batch separately.
5. **Leaderboard:** one upload, read as a sanity check rather than a selection.

### 4.2 Make the submission anytime, so runtime cannot fail the run

- Score `e27k`'s ranking for the batch first; it takes seconds.
- Then compute E49 features model by model under a wall-clock deadline (for example 3,000 s from
  the first call, kept in a module-level start time).
- If a model cannot finish, return the familiarity ranking for that batch.

This turns E37's pass/fail runtime gate into a graded cost. Runtime only matters on an accuracy
tie, where it is the tie-break.

### 4.3 Counts: make the choice knowingly

At test batch sizes the tilts move about one call per model. They encode 14-case per-model
priors that the official split contradicts for exactly the two tilted models (Finding 67). For
the generalised candidate, use the plain `ceil(n/2)`, the prevalence the organizers' card
implies, and keep `e27k` as the board-calibrated alternative. This is a judgement call between
two weak sources; §2.4 shows the stakes.

### 4.4 Lower priority

- **Length-neutral familiarity** (rank by NLL residualised on log tokens within the batch)
  removes the component that flips sign across populations. Its remaining signal is
  0.48–0.59 (§2.2), so it is a hedge for the fallback, not a primary signal.
- **E31 with fixed rewrite kinds** is cheap: five forward passes. It is the only one-pass signal
  with a consistent sign. Worth one leaderboard read if E49 fails.
- **Labels remain the real ceiling.** Finding 72 puts it at about 320 rows. Official-family
  rewrites (rephrase, rename, domain, distract) generated by an LLM, and sampled under the
  official decoding settings, are now explicitly allowed (Finding 59). The cost, not the rule,
  is what makes this impractical before 1 November.

### 4.5 Do not

- Search counts or thresholds further on the 14-case board.
- Switch the familiarity score to the chat template (Finding 62).
- Ship anything selected on the 34 official rows without an independent read. Min-K gap (0.808
  offline, 4/14 on the board) is the precedent.

## 5. Corrections the record needs before the report

1. **Finding 46 and `PLAN.md`:** on the official problems, shorter problems lean robust (raw AUC
   0.335, coefficient −0.84). This is the opposite of MATH.
2. **Finding 71 / E47:** the 90 tok/s rate is the T4 rate. The verdicts for E14 and E36/E37
   should say "infeasible at T4 throughput without batching", not "at any accuracy".
3. **Finding 64 (E37 runtime):** record that the 7,526 s was measured at `max_num_seqs = 2` under
   vLLM, which the evaluation image does not ship.
4. **Findings 48 and 56 (G-V2, E34a):** record that sampling ran at temperature 0.6 with 12k–24k
   caps and a 4/4 eligibility gate, against the official temperature 1.0 and 100k cap. The
   failures bound this pipeline, not official-family self-labelling.
5. **`PLAN.md` §12.3:** most of the contradiction is the length axis. Length-controlled
   familiarity on the official rows is 0.48–0.51 when residualised and 0.27–0.39 by quintile
   bins: unresolved at n = 34, with the length part explained.

Reproduce §2: `python train/review_generalization.py`. It needs `Qwen/Qwen3-0.6B` in the local
Hugging Face cache, the E23 and E35 run outputs under `runs/`, and caches proxy scores to
`runs/review-generalization/`.
