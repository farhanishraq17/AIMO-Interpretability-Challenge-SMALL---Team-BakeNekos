# Experiment log

Every experiment run for the Small Models Track, in the order it was run. Configurations
live in `datagen/kaggle_launch.py` (`EXPERIMENTS`); the kernels as pushed, with each run's
manifest, are in `kernels/`; measurements live in `RESULTS.md`; the reasoning behind each
result is the numbered Finding in `PROGRESS.md`.

Last updated: 30 September 2026. **The 27 Oct freeze is decided: ship `e27k`** (Findings 63, 64). E37 and E38 are closed and no new candidate is eligible; E39, E40 and E41 audited the incumbent instead, and it survives all three.

### E55: done 17 Oct, every success criterion met (Finding 81)

One-pass features (mean NLL and Stage A) for 548 problems x 4 models: the 36 official problems,
ReliableMath AIME and AMC, and our olympiad pool, so a per-problem predictor can train the day the
collaborator's self-labels arrive. No labels read. Amendment 1: E56 is not run - E49's recorded
budget-path runtime puts 60 one-problem calls at 496 minutes against the 50-minute rule.

| Kernel | Account | Status |
| --- | --- | --- |
| `e55-features-qwen` | Mashrafi | complete, 17 min |
| `e55-features-skywork` | Nasif | complete, 25 min |
| `e55-features-olmo` | Samin | complete, 27 min |
| `e55-features-deepseek` | Labib | complete, 25 min |

### E52b: stopped 14 Oct with no rewrite sample (protocol `protocols/E52_PROTOCOL.md` §9, committed `4297317` before launch)

**Outcome:** every Olmo kernel died with vLLM 0.20's `Cannot get 1 free blocks from the pool`. On a T4 pair
Olmo's cache holds 20,288 tokens; two samples of one problem at a 16k cap need about 33k, and this vLLM
crashes instead of preempting a sibling. Saved: base records for 16 of 75 problems (a 8, c 8), 0 rewrite
samples, so section 9's reading cannot be made. About 7.5 session-hours including the failed pilots.

Olmo-3-Think on E52's 75 problems, question 1 only: does it answer through a contradiction like
Skywork (support for its `e27k` tilt) or reason about it like Qwen? Four kernels, one per quarter
of the sorted problem keys; vLLM 0.20.0 with `transformers` 5.5.3, TP=2.

| Kernel | Account | Status |
| --- | --- | --- |
| `e52-olmo-a` | Mashrafi | v1-v3 failed at engine start (amendments `700e1fa`, `2f29554`, `7ec5390`); v4 sampled at 56 tok/s, saved 8 base records, then died (block pool) |
| `e52-olmo-b` | Nasif | v2 died in base after 41 min (block pool), 0 records |
| `e52-olmo-c` | Samin | v2 finished 8 base records, then both engines died (block pool) |
| `e52-olmo-d` | Labib | v2 died in base after 29 min (block pool), 0 records |

Final Olmo engine: eager, utilisation 0.85, at most 16 sequences, TP=2. vLLM 0.20 sizes Olmo's cache as
full attention on every layer (20,288 tokens), so only about 1.5 full-length requests fit at once.

### E52: closed 13 Oct, gate 3 failed with no positive class (Finding 77)

**Result:** within 16k tokens neither model flags a contradiction. Skywork answered 217 of 274 rewrite
samples (44% with the original problem's answer) and flagged none; Qwen truncated 103 of 130 and flagged 2.
No eligible pair was robust-like, so no one-pass signal could be tested. `e27k` stays.

Protocol `protocols/E52_PROTOCOL.md`, committed `1d636c6` before launch.

Do our models notice an impossible premise (Finding 76), and does one forward pass predict which
problems they miss? ReliableMath AMC 2023, 75 originals and 164 contradiction rewrites, organizers'
decoding, 16k cap. Analysis: `python train/eval_e52.py` once every output below is downloaded.

| Kernel | Account | Work | Status |
| --- | --- | --- | --- |
| `e52-qwen-a` | Mashrafi | Qwen3.5-4B, 38 problems, both phases, one engine per T4 | complete, 3.4 h |
| `e52-qwen-b` | Nasif | Qwen3.5-4B, the other 37 | complete, 2.8 h |
| `e52-skywork-a` | Samin | Skywork, 38 problems, TP=2 | complete, 2.8 h |
| `e52-skywork-b` | DIDHITI | Skywork, the other 37. TASHIN refused (both GPU slots busy); MUBAMAN's session got no GPU (`nvidia-smi` missing, quota) | complete, 2.7 h |
| `e52-features` | Labib | mean NLL and Stage A states, all four models | complete, 0.4 h |

Download: `python scripts/kaggle_followup.py output e52-<job> --account <alias>`.

### E49: closed 8 Oct, failed its gate (Finding 74)

E37's frozen budget-path score, re-extracted with batched `transformers` (the path a submission
would take) and scored once on all 34 official rows. Protocol: `protocols/E49_PROTOCOL.md`; review
behind it: `REVIEW_GENERALIZATION.md`.

**Result:** problem AUC 0.659 passed gate 2 (≥ 0.65). Half-rank balanced accuracy 0.521 failed
gate 3 (> 0.55). Row AUC was 0.571, against 0.358 for the familiarity ranking on the same rows.
**`e27k` stays.** Result: `kernels/runs/e49-official/result.json`.

| Kernel | Account | Rows | Outcome |
| --- | --- | --- | --- |
| `e49-qwen` | Mashrafi | 12 | complete, 4.2 h; **scored** |
| `e49-skywork` | Nasif | 3 | complete, 0.8 h; **scored** |
| `e49-deepseek-r0/r2/r4/r8` | Nasif, Nasif, Samin, DIDHITI | 2 each | complete, 1.8–2.5 h; **scored** |
| `e49-deepseek-r6` | Samin | 2 | ERROR: out of memory on one ~7k-token forced prompt; rows taken from `e49-deepseek` (fallback) |
| `e49-olmo-r0/r3/r6` | DIDHITI, SALAM, Labib | 3 each | complete, 1.3–1.7 h; **scored** |
| `e49-deepseek` | TASHIN | 10 | complete, 8.4 h; fallback only (rows 6–7 scored) |
| `e49-olmo` | Labib | 9 | complete, 5.5 h; fallback, unused |

### Previous execution: E37 (closed)

| Retry | Account | Status | Result |
| --- | --- | --- | --- |
| `aimo-small-e37-retry-qwen` | MUBAMAN (`mksadman`) | Complete | Full model coverage; runtime exceeds the whole-run gate by itself |
| `aimo-small-e37-retry-skywork` | Arifin (`joblesssamurai`) | Complete | Full model coverage; outputs recovered |
| `aimo-small-e37-retry-deepseek` | Nipun (`nowshintabassumnipun`) | Error | Process guard stopped at its time limit; partial outputs recovered |
| `aimo-small-e37-retry-olmo` | Tamim3 (`abrarrafi1552`) | Access denied | Identity verified; notebook state and output remain unknown |

- E37 fails its frozen runtime gate. The exact coverage, durations and output hashes are
  recorded in `kernels/runs/e37-runtime-result.json` and the generated RESULTS section.
  Incomplete official coverage is not scored for accuracy. No further GPU retry or
  Codabench upload follows; E27k remains the incumbent.
- All four old `aimo-small-e37-official-*` jobs on Arifin (Qwen), Nipun (DeepSeek),
  Tamim2 (Skywork) and MUBAMAN (Olmo) are confirmed `CANCEL_ACKNOWLEDGED`. Their worker
  processes had failed CUDA linking while API status remained `RUNNING`. The user stopped
  those sessions. Compact failure evidence is preserved in `kernels/runs/e37-status.json`.
- The retry bundle fixes the missing `libcuda.so` linker name using the installed driver,
  verifies compilation and `cuInit`, and guards the model process against fatal worker
  errors and excessive runtime. Tamim3's standalone CUDA test passed. The earlier Olmo
  `rope_theta` error was separately fixed by Transformers 5.5.3, preserving its published
  rotary and attention settings. Features, budgets and statistical gates stayed frozen.
- E38a completed on Tamim3 and failed every exact-prefix comparison. E38b is closed without
  a GPU pilot. Source and results are in `kernels/runs/e38-prefix-audit-Tamim3/`;
  `train/report_e38.py` regenerates its table. No correctness labels were read by the audit.
- E37's GSM-Plus confirmation passed. Its exact result and launched source are preserved
  in `kernels/runs/e37-confirmation-Arifin/`; `train/report_e37.py` regenerates its table.
  `train/prepare_e37_model.py` rebuilt the predictor from only the pinned MATH archive,
  verifying source hashes against the confirmation run.
- GPU capability checks previously passed for Arifin, Nipun, Tamim2, MUBAMAN's replacement
  identity `mksadman`, and Tamim3's second check. Tamim1 (`fehazulislamsawmiq`) is distinct
  from Tamim2 (`duneak47`); its latest completed check received neither GPU nor internet.
  Tamim3's current notebook permission issue is unresolved and does not establish whether
  GPU capability changed. No more GPU computation is needed for these closed experiments.
- The first Qwen/DeepSeek attempts failed before inference because Kaggle lacks working
  `ensurepip`; the launcher now uses system pip with an isolated environment. Previously
  rejected concurrent-session launches produced no data and remain marked in manifests.

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
| E27 | Per-problem predictions, to beat the 0.64 constant ceiling | local + Codabench | Done | Official-label models 8/14; the familiarity split runs the other way up (10/14), and with per-model robust counts **12/14 = 0.86**, the best score so far (Findings 49, 58, 60, 61) |
| L0-L1 | Main Track: does Kaggle give an H100, and can a proxy panel stand in for gpt-oss-120b? | Kaggle | L1 complete, ungraded | See the Main Track repository, `https://github.com/farhanishraq17/AIMO-Interpretability-Challenge-LARGE---Team-BakeNekos` (Findings LF-1 to LF-10) |
| E28 | Rank-based familiarity successors: suffix NLL and Min-K tail shape | local + Kaggle + Codabench | Done | E28a ties the best at **0.71**; the exact E28d retry scores **4/14 = 0.29**, rejecting Min-K gap despite its offline stability |
| E29 | Apply the stable Min-K gap only where E27 found hidden-set gain | local + Codabench | Done | DeepSeek-only tail gap, constants elsewhere; submission 942281 scored **8/14 = 0.57**, so the transfer is rejected |
| E30 | Architecture-invariant activation geometry | local | Done | Family-wise test fails and the selected signal reverses on independent MATH data |
| E31 | Prompt-state sensitivity to answer-preserving rewrites | Kaggle + local | Done | Best AUC 0.735, but template CI and max-T test fail; Qwen is near chance |
| E32 | Independent confirmation of E31 on DeepSeek MATH | Kaggle + local | Done | AUC 0.545, CI [0.428, 0.658]; lane closed |
| E39 | Is the 0.86 incumbent stable and fast enough to freeze? | Kaggle | Done | **Yes on mechanism.** 334 s for all four models, exactly repeatable, **0 flips under fp16**; but the chat template moves 2-10 of 36, so the bare-text prompt is load-bearing (Finding 62) |
| E40 | Is 12/14 an artifact of small-batch rounding? | local | Done | **No.** `e27k` calls the true robust count for every model in 56/56 consistent compositions and `e27e` in 0/56, so the tilts are forced by the probes; the ranking on top sits at p = 0.055 (Finding 63) |
| E41 | Do the probe-measured balances still hold? | Codabench | Done | **Yes, 5 for 5.** All four balances reproduce exactly twelve days on, so the set has not been rebuilt and the freeze is decided: **ship `e27k`** (Finding 64) |
| E42 | Does the shipped ZIP run in the competition image? | Docker (collaborator) | Done | **Yes.** Imports, contract and fallback all clean under the exact pins; the two reported failures were both bugs in the test harness, now fixed (Finding 65). The fixed test passes end to end from a rebuilt image on a second machine (Finding 66) |
| E43 | How far can the private prior drift before the tilts are wrong? | local | Done | **Crossover at a 0.5 robust share**, and only Skywork and Olmo matter. The official split puts both at 0.667/0.778, the other side of it, so the two-case gain is a symmetric bet (Finding 67) |
| E44 | Is the familiarity ranking model-specific at all? | local | Done | **No.** Mean pairwise Spearman 0.821 across the four models, 0.748 within length deciles; the ranking is a problem property and the count is the only model-specific part (Finding 68) |
| E45 | Does publication status explain the direction reversal? | local | Done | **No support.** Both halves of the official split reverse (AUC 0.417 mirrored, 0.299 not), so they do not straddle 0.5; underpowered and the indicator is one-sided (Finding 69) |
| E46 | Does the reversal hold against continuous `max_drop`? | local | Done | **Consistent in sign, never significant.** Spearman +0.247 overall, positive for every model, +0.225 within length sextiles, every CI spanning zero (Finding 70) |
| E47 | Could the methods we rejected have shipped at all? | local | Done | **Two could not**, by 10-11x: E14 and E36/E37 exceed the 3600 s whole-run budget regardless of accuracy. A per-pair cost gate would have caught both for free (Finding 71) |
| E48 | How many labels would each closed lane have needed? | local | Done | **63 to 320 rows; we had 33 to 61.** Power was 14-53%, so the experiments could not have returned a positive result even if their hypotheses were true (Finding 72) |
| E50 | What exactly is the hidden set, given every score we have? | local | Done | **Solved to two states**, both Qwen 5, DeepSeek 3, Skywork 3, Olmo 3. `e27k` is perfect on the last three; its two misses are one swapped Qwen pair. Ranking null p = 0.048 on that composition. A public pre-withholding `train-main-v2` revision exposes 7 validation labels; unused (Finding 75) |
| E51 | Which perturbation families make a label spurious? | local | Done | **Different ones per population.** MATH: domain (0.86 of rows drop >= 0.25). Olympiad: `expert_no_solution` (0.43), surface families <= 0.03. MATH-trained models learned the wrong mechanism (Finding 76) |
| E55 | One-pass features for every candidate problem, and per-problem cost | Kaggle | Done | **Complete**: 548 problems x 4 models, 100% coverage, zero non-finite; one pass is 0.55-0.89 s per problem on T4. Budget path closed on cost (Finding 81) |
| E54 | Does any held signal beat the constants as a per-problem rule? | local | Done | **Only the budget path, by the rule** (Putnam 0.598, GSM-Plus 0.564), and on the official rows its threshold calls every problem spurious. A candidate needing in-population re-calibration; the constants ship meanwhile (Finding 80) |
| E52b | Does Olmo answer through a contradiction like Skywork? | Kaggle | Stopped | **No data.** vLLM 0.20 cannot hold two 16k Olmo samples in a T4 pair's 20,288-token cache and crashed in every kernel; 16 base records, 0 rewrite samples. Olmo tilt keeps only its leaderboard support (Finding 78) |
| E52 | Do our models notice an impossible premise, and does one forward pass predict it? | Kaggle | Done, gate failed | **Almost never, so there is nothing to predict.** Skywork 0 of 274 rewrite samples flagged (44% gave the original answer); Qwen 2 of 130, with 79% out of budget. No robust-like pair; `e27k` stays (Finding 77) |
| **E34a** | Can official-like surface perturbations yield a usable new AIME label set? | Kaggle (`tashintahir`) | **Done — failed gate** | Only 1/52 originals was 4/4 correct; one robust label, no spurious label; E34b closed and no submission |
| **E35** | Do independently published robustness labels support a source-transferring problem-text model? | local | **Done — failed gate** | Putnam holdout AUC 0.609, CI [0.442, 0.769]; official transfer not opened |
| Official data | `aimo-interp/train-main-v2`, published 17 Sep | local | Ingested 20 Sep | Labels for all four models; their label rule differs from ours; MATH->olympiad transfer is below chance (Finding 30) |

---

## 2. Running now

**Nothing, on either track** (checked 26 September: `kaggle_launch.py status` and
`klarge.py status` both report zero RUNNING or QUEUED kernels).

**The submission that stands is `e27k`, 12/14 = 0.86** on the small-track leaderboard: within
each model's batch, the more familiar problems are called robust, and how many are called robust
comes from that model's class balance as the probes measured it. `e27e` (0.71) is the prior-free
version of the same ranking, and E11 v2 (0.64) is the zero-GPU fallback. Every ZIP rebuilds with
`python submission/make_e27.py`.

**The leaderboard series, in full** (Small Model Track, 14 cases, so one case is 0.07). `submission/README.md` carries the same series **ordered by score**, with each ZIP's name, what it does and how to rebuild it:

| Submission | Score | What it was |
| --- | --- | --- |
| all-False probe | 0.50 | 7 of 14 cases are spurious |
| E11 v2 | 0.64 | text + hidden-state fusion for DeepSeek, constants elsewhere |
| `e27a`/`e27b`/`e27c` | 0.57 | models fitted on the official labels: they carry nothing |
| `e27d` | 0.29 | familiarity split, *unfamiliar* half robust - far below chance, so inverted |
| `e27e` | 0.71 | the same split the other way up |
| `e27f` | 0.64 | best constants alone, confirming the constants ceiling |
| `e27g`/`e27i`/`e27j` | 0.64 | split for Qwen / Skywork / Olmo alone |
| `e27h` | 0.71 | split for DeepSeek alone: the whole gain of `e27e` |
| **`e27k`** | **0.86** | `e27e` with Skywork and Olmo calling one fewer problem robust |
| `e27m`/`e27n`/`e27o`/`e27p` | 0.79 | every single-step neighbour of `e27k`'s counts |

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
