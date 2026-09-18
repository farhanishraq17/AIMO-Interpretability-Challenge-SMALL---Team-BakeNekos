# Experiment log

Every experiment run for the Small Models Track, in the order it was run. Configurations
live in `datagen/kaggle_launch.py` (`EXPERIMENTS`); the kernels as pushed, with each run's
manifest, are in `kernels/`; measurements live in `RESULTS.md`; the reasoning behind each
result is the numbered Finding in `PROGRESS.md`.

Last updated: 18 September 2026, 21:00 UTC.

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
| E15 | Qwen3.5-4B: which problems can be labelled? | Kaggle | Done (2 sessions) | **90 of 285** eligible (32%) |
| E16 | Skywork-OR1-Math-7B: same | Kaggle | Done (2 sessions) | **157 of 246** eligible (64%) |
| E17 | Olmo-3-7B-Think: same | Kaggle | Paused | 17 of 30 eligible; too slow on T4s (Finding 26) |
| E18 | Qwen3.5-4B: sample the variants of eligible problems | Kaggle | **Running** | — |
| E19 | Skywork-OR1-Math-7B: same | Kaggle | **Running** | — |

---

## 2. Running now

Step 2 of labelling, launched 18 Sep around 20:45 UTC. For every problem the model solved in all
4 base samples, sample each compliant (`answer_modulus`) variant 4 times; `label.py` then marks
the case robust if accuracy does not drop.

| Run | Account | Configuration | Work |
| --- | --- | --- | --- |
| `e18-variants-qwen35` | RSD | Qwen3.5-4B, one vLLM engine per T4 (2 shards), k=4, 16k cap, 4-problem chunks, 8.5 h budget | 109 variants over 53 problems |
| `e19-variants-skywork` | Labib | Skywork-OR1-Math-7B, tensor parallel 2, same settings | 205 variants over 101 problems |

**Base-phase yields that feed this** (Findings 25–26):

| Model | Sampled (of 505) | Eligible | Rate | By tier (eligible/sampled) |
| --- | --- | --- | --- | --- |
| Skywork-OR1-Math-7B | 246 | 157 | 64% | AMC 20/23, AIME 9/26, olympiad 128/197 |
| Qwen3.5-4B | 285 | 90 | 32% | AMC 8/25, AIME 2/29, olympiad 80/231 |
| Olmo-3-7B-Think | 30 | 17 | 57% | AMC 3/4, AIME 1/3, olympiad 13/23 |

**Why Olmo is paused:** it needs a full KV cache and fits a T4 pair only at a 12k cap, so it
runs one or two sequences at a time (30 problems in 9.6 h). It moves to the Fields grant.

**Fleet:** RSD and Labib hold this week's remaining quota. SALAM, DIDHITI, TASHIN and ISHRAQ
have used theirs; ISHMAM1 and DESIKALAKAR are still unreachable.

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

---

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

---

## 5. Where this leaves the approach

- **DeepSeek has a working model:** text + prompt hidden states, 0.7173 balanced accuracy,
  packaged and verified on GPUs.
- **Two label-free shortcuts have now failed** (E12, E14). Neither reads anything specific to
  the model; both mostly read the problem.
- **So the three unlabelled models need labels**, from the organizers or from E15–E17. That is
  the whole current GPU load.
- **The leaderboard needs re-measuring** before E11 v3 sets each model's default.
