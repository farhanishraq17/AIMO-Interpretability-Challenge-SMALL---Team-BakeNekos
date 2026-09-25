# Experiment log

Every experiment run for the Small Models Track, in the order it was run. Configurations
live in `datagen/kaggle_launch.py` (`EXPERIMENTS`); the kernels as pushed, with each run's
manifest, are in `kernels/`; measurements live in `RESULTS.md`; the reasoning behind each
result is the numbered Finding in `PROGRESS.md`.

Last updated: 25 September 2026.

**Reading the status column:** Done means the run finished and was analysed. Superseded means
the result exists but a later run replaced it. Failed means it produced nothing usable.

---

## 1. Summary

| ID | Question | Where | Status | Verdict |
| --- | --- | --- | --- | --- |
| E0 | How far does the problem text alone go? | local | Done | 0.6732 balanced accuracy: the floor everything else must beat |
| E0b | Is that just recognising the source dataset? | local | Done | No: 0.64–0.67 within a single dataset |
| E1 | Prompt hidden states, every layer | Kaggle | **Superseded** | Ran on a whitespace-stripping tokenizer |
| E1b | Same, with the tokenizer fixed | Kaggle | Done | Layer ensembles AUC 0.749; picking one layer overfits |
| E2 | Reasoning traces + uncertainty features | Kaggle | **Superseded** | bf16 on T4 is emulated: 617 s/problem, then out of memory |
| E2-bench | bf16 vs fp16 on T4 | Kaggle | Done | fp16 is 3.3× faster; the two produce different traces |
| E2f | Trace features, in fp16 | Kaggle | Done | The published uncertainty features are near chance here |
| E3 | Does Olmo-3-7B-Instruct run? | Kaggle | Done | Yes; 33 layers, no numeric problems |
| E4 | Hidden states *during* reasoning | Kaggle | Done | Weaker than prompt states; the signal is in the prompt |
| E7 smoke | Olympiad sampling at a 4k cap | Kaggle | **Superseded** | 4k truncates nearly everything; lossy tokenizer |
| E7b | Olympiad sampling at 8k/16k caps | Kaggle | Done | About 25% of problems are labellable; 300–1,250 s each |
| E10 | Fuse text + hidden states + traces | local | Done | Text + prompt states is the model; traces dilute it |
| E11 | Package that as a submission | local + GPU check | Done | v1 0.47, **v2 0.64** on the leaderboard |
| Probes | What is in the hidden validation set? | Codabench | Done (re-measured 18 Sep) | About 14 cases; v2's defaults are the best constants (Finding 27) |
| E12 | Label-free signals from one forward pass | Kaggle | Done | They are problem-text proxies, not model-specific |
| E13 | Do the three unlabelled models run? | Kaggle | Done | All three run on T4s under the pins |
| E14 | Does the answer survive rewrites at test time? | Kaggle | Done | **No.** At chance, and too expensive. Approach dropped |
| E15 | Qwen3.5-4B: which problems can be labelled? | Kaggle | **Stopped** (3 sessions) | 84 of 270 eligible; labels unusable (Finding 31) |
| E16 | Skywork-OR1-Math-7B: same | Kaggle | **Stopped** (2 sessions) | 143 of 232 eligible; labels unusable (Finding 31) |
| E17 | Olmo-3-7B-Think: same | Kaggle | **Stopped**, output recovered 20 Sep | 18 of 32 eligible; slowest model at 30 tok/s; labels unusable (Findings 26, 31, 42) |
| E18 | Qwen3.5-4B: sample the variants of eligible problems | Kaggle | Done | 89 of 91 variants; **47 problems labelled**, 19 robust; 26 clear-cut (Finding 29) |
| E19 | Skywork-OR1-Math-7B: same | Kaggle | **Stopped** | Labels set aside after E20 (Finding 31) |
| E20 | Do self-labels agree with official labels? (DeepSeek, 66 MATH problems) | Kaggle | Done | **No.** kappa -0.14, drops correlate -0.25. Self-labelling stops (Finding 31) |
| **E21** | Does the DeepSeek model transfer to olympiad problems? Is there a cross-model signal? | Kaggle + local | Done | **No to both.** AUC 0.429 on 10 unseen olympiad rows; no held-out model clears AUC 0.58 (Finding 32) |
| Gapfill | Base samples for problems whose variants were already paid for (Qwen, Skywork) | Kaggle (`nakib_v2`) | Done | 26 Qwen + 23 Skywork self-labels; invalidated by E20 (Finding 37) |
| Stage A olympiad | Do prompt states carry olympiad signal where text does not? | Kaggle (`nakib_v2`) | Done | AUC 0.844 on 12 official Qwen rows, strict control p = 0.073, self-labelled training set (Finding 41) |
| Official-label tests | Do models agree? Does the text block reach olympiad rows? | local (`nakib_v2`) | Done | Models agree **20/20**, p = 0.0005 (Finding 38); text block AUC 0.617, CI spans 0.5 (Finding 39) |
| E22 | Problem-level acceptance harness on `train-main-v2`, template-grouped | local | Done | Nothing clears chance; problem length best at 0.635, CI [0.42, 0.84] (Finding 46) |
| E23 | Do familiarity/contamination scores predict robustness? | Kaggle | Done | **G-M fails**; on olympiad problems familiar problems lean spurious, not significant (Finding 47) |
| E24 | Do `answer_modulus` self-labels agree with official labels on *olympiad* problems? | Kaggle | Done | **G-V2 fails**, but the sign flips from E20: kappa +0.29 on 15 pairs vs 0.72 chance agreement (Finding 48) |
| E27 | Per-problem predictions, to beat the 0.64 constant ceiling | local + Codabench | Done | Official-label models 8/14; the familiarity split runs the other way up (10/14), and with per-model robust counts **12/14 = 0.86**, the best score so far (Findings 49, 58, 60) |
| E28 | Rank-based familiarity successors: suffix NLL and Min-K tail shape | local + Kaggle + Codabench | Done | E28a ties the best at **0.71**; the exact E28d retry scores **4/14 = 0.29**, rejecting Min-K gap despite its offline stability |
| E29 | Apply the stable Min-K gap only where E27 found hidden-set gain | local + Codabench | Done | DeepSeek-only tail gap, constants elsewhere; submission 942281 scored **8/14 = 0.57**, so the transfer is rejected |
| E30 | Architecture-invariant activation geometry | local | Done | Family-wise test fails and the selected signal reverses on independent MATH data |
| E31 | Prompt-state sensitivity to answer-preserving rewrites | Kaggle + local | Done | Best AUC 0.735, but template CI and max-T test fail; Qwen is near chance |
| E32 | Independent confirmation of E31 on DeepSeek MATH | Kaggle + local | Done | AUC 0.545, CI [0.428, 0.658]; lane closed |
| **E34a** | Can official-like surface perturbations yield a usable new AIME label set? | Kaggle (`tashintahir`) | **Done — failed gate** | Only 1/52 originals was 4/4 correct; one robust label, no spurious label; E34b closed and no submission |
| **E35** | Do independently published robustness labels support a source-transferring problem-text model? | local | **Done — failed gate** | Putnam holdout AUC 0.609, CI [0.442, 0.769]; official transfer not opened |
| Official data | `aimo-interp/train-main-v2`, published 17 Sep | local | Ingested 20 Sep | Labels for all four models; their label rule differs from ours; MATH->olympiad transfer is below chance (Finding 30) |

---

## 2. Running now

**E35 completed and failed its pre-registered external gate.** The frozen MATH-Perturb to
Putnam-AXIOM transfer produced AUC 0.609 with CI [0.442, 0.769]. No official labels were scored
and no submission was made.

**E34a completed and failed its pre-registered gate.** Both base shards processed all 52
problems, but only one original was correct in all four samples (1.92%). It yielded one robust
`latex_to_plaintext` label, no `premise_reorder` label and no spurious label. This misses every
yield/class-balance requirement in `E34_PROTOCOL.md`; E34b is closed and no submission follows.

**E27's decomposition is complete (26 Sep).** The rebuilt per-model ZIPs scored constants 9/14,
Qwen-only 9/14, DeepSeek-only 10/14, Skywork-only 9/14 and **Olmo-only 9/14**, the cell Finding 50
left queued; all four splits together score 10/14. The deltas sum to +1 as they must, and since
per-model contributions are additive, **10/14 is the ceiling of the familiarity family**. The same
split scores 0.353 on the official rows, where always-robust reaches 0.706 (Finding 58).

**E28-E29 complete.** E28a (`suffix_nll`, submission 942047) scored 0.71, tying rather than
improving E27e. E28d's first run (942257) failed at ingestion, but an exact retry (942361)
finished and scored **4/14 = 0.29**; the official local path still reports coverage 1.0 and
zero invalid predictions. A stability check over
only the pre-registered statistics leaves out each scoring model in turn: `min_k_gap` retains
problem AUC 0.731-0.835, while mean and suffix NLL remain below 0.37. Its official DeepSeek AUC
is 0.762, and E27's hidden decomposition says the DeepSeek split supplies the entire one-case
gain. E29a therefore used Min-K gap for DeepSeek only and constants elsewhere. Submission
942281 scored **8/14 = 0.57**, two cases below E27's 10/14 best. Since both the all-model and
DeepSeek-only tests fail, the tail-gap branch is closed. No cutoff or ensemble tuning follows
on this 14-case validation set.

All E28 real-checkpoint checks ran on TASHIN and RSD; no `farhanishraqq` GPU was used.

E24b finished on 23 Sep (15.8 GPU-hours over four accounts) and gate G-V2 failed, so the
self-labelling lane stays closed and E25 is dropped (Finding 48).

**E24a.** `train-main-v2` ships no answers, so Skywork sampled all 20 labelled problems
8 times (`e24a-answer-key-skywork`, ISHRAQ; its first push went to Eshad, which ran without a
GPU and needs phone verification). A reference answer is kept only when all 8 samples agree:
that happened on 3 of the 6 problems whose answers were already known, matching all 3, and on 8
of the 14 unknown ones. **14 problems answered, 28 compliant variants, 27 (model, problem)
pairs to test**: Qwen 9, Olmo 8, DeepSeek 7, Skywork 3.

**Why it stopped.** E20 compared self-generated `answer_modulus` labels with official ones on
66 DeepSeek problems: agreement 0.29, Cohen's kappa -0.135, and our measured drops correlate
with the official drops at -0.25 (Finding 31). The category the then-current rule let us label
does not measure the robustness the competition scores.

**What replaced it.** `aimo-interp/train-main-v2` (published 17 Sep, ingested 20 Sep) carries
official labels for every evaluated configuration, including the first for Qwen3.5-4B,
Skywork-OR1-Math-7B and Olmo-3-7B-Think (Finding 30).

| Source | DeepSeek | Qwen3.5-4B | Olmo-3-7B-Think | Skywork | Llama-3.1-8B |
| --- | --- | --- | --- | --- | --- |
| Train-ready instances | 153 (46 robust) | 12 (8) | 9 (7) | 3 (2) | 49 (0) |

**Kept for the report:** the E15-E19 and gapfill base and variant records, as a documented
negative result, and the Qwen and Skywork self-labels (`Self-generated olympiad labels` in
`RESULTS.md`), which are not used for training.

**What the official labels then bought us: nothing yet.** E21 spent them on the two questions
they made answerable, and both came back negative (Finding 32, below).

---

## 3. Completed experiments

### E0: text-only floor (local, 14–15 Sep)

Char and word tf-idf, handcrafted surface features, and gradient boosting over the 138
DeepSeek instances, out-of-fold across the five problem-grouped folds.

- **Char tf-idf: 0.6732 balanced accuracy, 0.6686 AUC.** Every model-internals method is
  measured against this.
- Always-spurious scores 0.7391 raw accuracy, which is why raw accuracy is not the
  development metric.
- Not a dataset artifact: 0.670 within `hendrycks-math-train-reference-500` (n=84) and 0.644
  within `hendrycks-math-test-reference-100` (n=53).
- Findings 9 and the `RESULTS.md` text sections.

### E1 / E1b: prompt hidden states (ISHMAM1, 15–16 Sep, 0.1 h each)

One forward pass per problem; every layer's prompt tokens pooled three ways (last, mean, max),
37 layers × 4096 dimensions, bf16 on 2× T4.

- **E1 is invalid.** It ran through `AutoTokenizer`, which strips whitespace from this
  checkpoint under transformers 5.13 (Finding 14). 135 of 138 prompts were wrong.
- **E1b** re-extracted everything with the checked tokenizer.
- Picking a layer per fold overfits: inner AUC 0.75–0.85 falls to 0.640 held out.
- Averaging all layers instead reaches **AUC 0.749**, and fused with text gives balanced
  accuracy **0.7173** (Findings 11, 16).
- Cost: 1.0 s per problem.

### E2 / E2-bench / E2f: reasoning traces (ISHRAQ, DIDHITI, TASHIN, DESIKALAKAR, 15–16 Sep)

Greedy generation plus the starter kit's 14 uncertainty features.

- **E2 (bf16) failed the budget:** 617 s per problem, and shard 1 hit out-of-memory after 16
  problems.
- **The benchmark explains it:** the T4 has no native bf16, so Kaggle emulates it. fp16 runs
  at 48.4 tokens/s against 14.7, and the two dtypes produce different traces, so they are
  never mixed (Finding 13).
- **E2f (fp16)** completed both shards in 3.7 h and 4.1 h, at 190–211 s per problem.
- **Result:** the published uncertainty features are near chance on our instances (AUC
  0.54–0.55). Longer traces lean robust, but the 4k cap confounds that (Finding 15).

### E3: Olmo-3-7B-Instruct plumbing (TASHIN, 15 Sep, 0.1 h)

Loads under the pins on 2× T4 in bf16, 33 hidden states × 4096, 1.0 s per problem, no numeric
problems. Labels in that run belong to DeepSeek, so the features are plumbing only
(Finding 12).

### E4: hidden states during reasoning (SALAM, 16 Sep, 0.4 h)

Teacher-forced pass over prompt + the model's own E2f trace, pooled at five positions.

- **The hypothesis failed.** `prompt_last` is the strongest pooling (AUC 0.739) and positions
  late in the reasoning are the weakest (0.60).
- Fusing it in raises AUC to 0.739 but not balanced accuracy, and it costs a full generation
  at inference, so it stays out of the shipped model (Finding 18).
- An out-of-memory failure on the first attempt led to hook-based pooling, which had to
  reproduce the final-norm convention exactly (Finding 17).

### E7 smoke / E7b: olympiad sampling (ISHMAM1, TASHIN, 16 Sep, 0.9 h and 4.4 h)

vLLM on 2× T4, fp16, k=4 samples per problem.

| Run | Cap | Eligible | Cost per problem |
| --- | --- | --- | --- |
| E7 smoke | 4,096 | Almost none | — (also used the lossy tokenizer) |
| E7b, AMC | 8,192 | 4 of 16 | 311 s |
| E7b, AIME | 16,384 | 2 of 8 | 1,258 s |

Labelling the whole pool for DeepSeek would cost 45–175 GPU-hours, which is why E15–E17 run
the same phase for the models that have no labels at all (Finding 19).

### E10: fusion (local, 16 Sep)

Nested grouped cross-validation over text, prompt states, trace states and trace features,
with thresholds learned inside each fold.

| Combination | Balanced acc. | AUC |
| --- | --- | --- |
| Text alone | 0.6389 | 0.6481 |
| **Text + prompt states (shipped)** | **0.7173** | 0.7230 |
| Text + prompt states + trace states | 0.7116 | 0.7391 |
| All four blocks | 0.6667 | 0.7176 |

Text + prompt states is the model in E11; the trace block dilutes it (Findings 16, 18).

### E11: the submission (local + `e11-gpu-check` on ISHMAM1, 16–17 Sep)

- The packaged ZIP was run on real GPUs before uploading: 138 problems, 0 disagreements with
  the reference decisions, recomputed probabilities matching to 1.3e-9, 176.8 s total
  (Finding 20).
- Three bugs were caught that way: a scikit-learn version mismatch, a text-block failure that
  would have invalidated a whole batch, and a silent fallback when the model was not cached.
- **Leaderboard: v1 0.47, v2 0.64** (v2 defaults Qwen3.5-4B to True).

### Leaderboard probes (Codabench, 16–17 Sep)

Constant-prediction ZIPs from `submission/make_probes.py`, to find out which models the hidden
set contains.

| Submission | Score |
| --- | --- |
| all False | 0.47 |
| True for DeepSeek only | 0.47 |
| True for Qwen3.5-4B only | 0.53 |
| True for Skywork only | 0.47 |
| True for Olmo-3-Think only | 0.47 |

- Only the Qwen default moved the score, which pointed to a small, Qwen-heavy set (Finding 21).
- **These need redoing.** E11 v2 scored 0.64 where these numbers predict 0.53, and an older
  all-false upload now shows a different score, so the set or the task changed (Finding 23).
- Four uploads were also rejected outright by the Main Track task, which refuses `small.txt`.

### E12: label-free signals from one forward pass (ISHMAM1, 17 Sep, 0.3 h)

Immediate-answer confidence, answer stability under rewrites, problem predictability, and
next-token uncertainty, for 138 problems at 5.3 s each.

| Signal | AUC | Within length quintiles |
| --- | --- | --- |
| Next-token entropy | 0.736 | 0.587 |
| Problem predictability (`problem_nll`) | 0.324 | 0.419 |
| Immediate answer survives a rewrite | 0.507 | 0.497 |
| Problem length (reference) | 0.734 | — |

**Verdict:** what works is reading the problem, not the model. Another model computes
`problem_nll` just as well against DeepSeek's labels (Finding 22).

### E13: do the three unlabelled models run? (TASHIN, ISHRAQ, DESIKALAKAR, 17 Sep, 0.2–0.3 h)

| Model | Loads as | Layers × width | Non-finite activations | Label-free scoring |
| --- | --- | --- | --- | --- |
| Qwen3.5-4B | `Qwen3_5ForCausalLM` (saved as a vision-language model) | 33 × 2560 | 0 of 20 | 3.3 s/problem |
| Skywork-OR1-Math-7B | `Qwen2ForCausalLM` | 29 × 3584 | 0 of 20 | 3.8 s/problem |
| Olmo-3-7B-Think | `Olmo3ForCausalLM` | 33 × 4096 | 0 of 20 | 4.8 s/problem |

Every tokenizer round-trips. Qwen3.5's linear-attention layers fall back to plain PyTorch
here and in the evaluation image (Finding 22).

### E14: test-time answer agreement (ISHRAQ, DIDHITI, 17 Sep, 4.5 h + 5.8 h)

The idea: robustness is defined as "the answer survives perturbation", so measure that
directly at inference. DeepSeek reasoned in full (k=2, 8k cap) on each problem and on up to
four rewrites: LaTeX to plain text, reordered premises, typos, and an appended irrelevant
sentence. 447 prompts, 3.86M generated tokens.

| Signal | AUC [95% CI] | Within length quintiles |
| --- | --- | --- |
| Rewrite answers match the original | 0.458 [0.36, 0.55] | 0.543 |
| Worst rewrite type | 0.468 [0.38, 0.55] | 0.566 |
| Original's samples agree | 0.455 [0.37, 0.52] | 0.484 |
| Tokens spent reasoning | 0.652 [0.56, 0.74] | 0.497 |

- **The rewrites barely bite:** every rewrite sample matched the original answer for 78.5% of
  problems, spurious ones included.
- **Reasoning length is problem length** once controlled.
- Adding it to the shipped model changes balanced accuracy 0.7173 → 0.6928.
- **Cost:** 28,000 generated tokens per problem, far beyond the 3600 s evaluation budget.
- **Decision (gate G-D): dropped** (Finding 24).
- It also exposed a measurement bug of ours: the old length control turned heavily tied
  signals into length itself, which flattered them by 0.1–0.2 AUC.

### E21: olympiad transfer and cross-model signal (DIDHITI, SALAM, TASHIN, ISHRAQ, 20 Sep, ~0.3 h each)

Stage A prompt states over the official split for all four models: DeepSeek's 152 train
instances (the 8 olympiad ones from `train-main-v2` included), Qwen's 12, Olmo-3-Think's 9,
Skywork's 3. Four small kernels, one per model, because each needs its own checkpoint.

**Question 1: does the shipped DeepSeek model survive the shift to olympiad problems?** The 10
`train-main-v2` rows are held out by construction: they were unlabelled when the artifact was
built. (Re-measured after the label fix of Finding 44; the first pass saw 8 of them.)

| Method | Accuracy | Balanced acc. | ROC-AUC |
| --- | --- | --- | --- |
| E11 fused, shipped artifact | 0.700 | 0.595 | 0.429 |
| its text block alone | 0.500 | 0.452 | 0.476 |
| layer ensemble refit on the 142 MATH rows | 0.500 | 0.548 | 0.476 |
| all-robust | 0.700 | 0.500 | 0.500 |

**Question 2: is there anything model-agnostic to borrow?** The five next-token uncertainty
scalars are the only features whose width does not depend on the model. Pooled over the 34
official rows, standardised within each model, one model held out at a time: no fold clears
AUC 0.58, and the balanced accuracies (0.464-0.750) swing on one or two rows each.

- **Decision (gate G-T): both fail.** The signal we have is specific to MATH-style problems and
  to DeepSeek. **E11 v2 stands; there is no v3 to build from a model** (Finding 32).
- Sample sizes are 3 to 12 rows, so these are go/no-go signals rather than measurements.
- Cost: about 1.2 GPU-hours in total, most of it downloading checkpoints.

### Ported from the `nakib_v2` branch (teammate's checkout and accounts, 19 Sep)

The branches are deliberately not merged; scripts were copied file by file and the findings
renumbered onto our sequence (Findings 33-43). Three runs and four local analyses came across.

| Run / analysis | Where | Result |
| --- | --- | --- |
| `e20-base-qwen35-gapfill` (their E23; the slug kept its original number) | NAKIB, 2.4 h | Base samples for 19 problems whose variants were already paid for; 13 eligible, Qwen reaches 26 labels |
| `e22-base-skywork-gapfill` (their E22) | NAKIB, 1.4 h | The same for 25 Skywork problems; 24 eligible, 23 labels |
| `e21-stagea-qwen35-olympiad` (their E21) | MR, 142 s | Stage A probes on self-labels score AUC 0.844 on 12 official Qwen rows; strict control p = 0.073 |
| `scripts/inspect_train_main_v2.py` | local | 82 rows, label rule verified at 0 violations, **29 templates from 36 problem strings** |
| `train/eval_train_main_v2.py` | local | **Models agree 20/20 on shared problems, p = 0.0005**; text block AUC 0.617, CI [0.391, 0.833] |
| `train/eval_stagea_olympiad.py` | local | The controls behind the 0.844 above |
| `datagen/plan_labelling_cost.py` | local | Labelling the whole pool for Qwen = 113 GPU-h, 16% of the remaining budget |

- **The two self-label results are superseded by E20** (Finding 31), which invalidated the label
  source. They stay in the record as measurements.
- **The one result that stands on official data is the 20/20 model agreement:** robustness is a
  property of the problem, not of the model, which is why per-model constants do as well as they
  do and why our cross-model *feature* test still failed (Finding 32).
- **Fleet correction.** That checkout's aliases are NAKIB (`nakibsaleh`), CHOTON
  (`gmnoorlabib` = the registry's `Labib`), MEDIUM (`rsdaiyan` = `RSD`) and MR
  (`mohsinreza1971`), so `CHOTON`/`MEDIUM` share a quota with `Labib`/`RSD` rather than adding
  one. Their audit found every AIMO kernel downloaded once `e17-base-olmo-think` was recovered
  - it had completed on 17 Sep and was never pulled. **18.5 GPU-hours are reachable from that
  machine; 46.0 across 20 runs are not**, including `e16-base-skywork` and
  `e15-base-qwen35-a` (Finding 42).
- **A bug it exposed here:** our ingest was aggregating the official split with the older tables
  for the same problem, which flipped one DeepSeek label and dropped another. Fixed; the
  official rows now decide any pair they cover, and DeepSeek's olympiad evaluation set grows
  from 8 rows to 10 (Finding 44).

---

### E30–E31: architecture-invariant and counterfactual activation geometry (local + TASHIN/RSD, 24 Sep)

E30 reduced the existing E21 all-layer archives to 12 norm/cosine trajectory scalars. Its
best official-data result (`mean_growth`, AUC 0.813) did not survive the 12-way max-T test
(p=0.2049) and reversed on 142 untouched DeepSeek MATH rows (AUC 0.348).

E31 then tested a stronger causal proxy: how much the labelled model's prompt state changes
under deterministic answer-preserving rewrites. Five measures and a lower-is-robust direction
were fixed before extraction. All 34 official rows completed without truncation. All-layer
mean cosine distance was best (AUC 0.735), but its 19-template CI [0.380, 0.938], corrected
p=0.0667 and Qwen AUC 0.531 fail the locked gate. It is not a submission. E32 is the single
independent check of that already-selected signal on DeepSeek MATH; no layer or perturbation
search is reopened. E32 completed all 142 rows without truncation and rejected the signal:
AUC 0.545, CI [0.428, 0.658], within-length AUC 0.510 and deployable median-split balanced
accuracy 0.527. The counterfactual lane is closed without packaging or a submission.

Neither run used the excluded `farhanishraqq` GPU account.

## 4. Failures worth remembering

| What broke | Where | Fix |
| --- | --- | --- |
| `AutoTokenizer` strips whitespace from DeepSeek prompts | E1, E7 smoke | `datagen/tokenizer_check.py` refuses any tokenizer that fails a round trip (Finding 14) |
| Emulated bf16 on T4: 617 s per problem, then out of memory | E2 | fp16, batch 4, expandable CUDA segments (Finding 13) |
| `output_hidden_states=True` holds every layer at once | E4 | Hook-based pooling, matching the final-norm convention exactly (Finding 17) |
| Kaggle ships scikit-learn 1.6.1, which cannot read our artifact | E11 check | Kernels pin 1.8.0, as the evaluation image does (Finding 20) |
| The solution silently fell back to text-only with no cached model | E11 check | The check pre-downloads the checkpoint (Finding 20) |
| Four probe uploads rejected: `small.txt` on the Main Track | Codabench | Submit to the Small Model Track task (Finding 21) |
| vLLM's custom all-reduce crashes on some T4 pairs | E14, E16 | NCCL instead (`disable_custom_all_reduce`) (Finding 23) |
| Qwen3.5 needs one state block per sequence; the default 256 exceeds the 157 available | E15 | `--max-num-seqs 64` (Finding 23) |
| Two Kaggle tokens stopped authenticating | 17 Sep | Awaiting new tokens from the account owners (Finding 23) |
| One problem's base accuracy scored another's variants, silently | E15-E19 | `pool_id` is not unique; disambiguated at the source, and one record of each colliding pair had been discarded before that (Findings 29, 36) |
| The official split was merged with older tables for the same problem | Ingest | Different label rules; the official rows now win outright (Finding 44) |

---

## 5. Where this leaves the approach

- **DeepSeek has a working model on MATH problems:** text + prompt hidden states, 0.7173
  balanced accuracy under grouped CV. E21 shows it does not transfer to olympiad problems
  (AUC 0.429), which is the distribution the competition scores.
- **Every shortcut has now failed:** label-free forward-pass signals (E12), test-time answer
  agreement (E14), self-generated labels (E20), and cross-model transfer (E21). Each is a
  measured negative result for the report.
- **Labels are the binding constraint.** 34 official small-track rows are not a training set.
  The 19 Sep rule change permits new official-family labels, but their symbolic source remains
  inaccessible (Finding 54).
- **E11 v2 (0.64) is the submission** until more labels exist. The open ask is to the
  organizers: is more training data coming?
