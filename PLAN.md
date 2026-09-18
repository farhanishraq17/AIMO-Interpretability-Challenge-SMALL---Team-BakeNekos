# Small Models Track: Execution Plan

**Scope:** AIMO Interpretability Challenge, Small Models Track (open-weight, <10B parameters).

**Key dates:**

| Date | Event |
| --- | --- |
| 17 Sep 2026 | Plan last updated |
| 1 Nov 2026, 12:55 UTC | Codabench submissions close (about 6.5 weeks away) |
| 15 Nov 2026 | Technical report and open-source code due |
| 11/12 Dec 2026 | NeurIPS workshop |

**Companion documents:**

- `COMPETITION.md`: rules, contract, leaderboard mechanics.
- `PROGRESS.md`: what was built and measured, with numbered Findings.
- `RESULTS.md`: generated metric tables.
- `CLAUDE.md`: working conventions and commands.

**Repository:** https://github.com/farhanishraq17/AIMO-Interpretability-Challenge---Small-Model-Track

---

## 0. Objective and success criteria

Build `are_robust(model_id, problems) -> list[bool]` that beats the published baselines on the
private small-track test set, then write it up.

| Bar | Leaderboard accuracy |
| --- | --- |
| Chance | 50% |
| Linear-probe baseline | 58.4% ± 7.5 (averaged honestly, at chance: Finding 2) |
| Uncertainty-profiling baseline | 69.2% ± 10.1 |
| **Minimum goal** | **>72%, coverage 1.0, 0 invalid predictions** |
| Stretch (prize contention) | 78%+ with fast runtime (ties break on speed) |

**How results are measured** (COMPETITION.md §5):

- Local `run_local.py` on `val-sample` checks the contract only; it has 1 small-model case.
- The live Codabench leaderboard scores the hidden validation set: 10 submissions a day, 100
  in total.
- The private test set decides prizes after 1 Nov.

Development decisions therefore use **nested grouped CV on our own instances**: balanced
accuracy and ROC-AUC, with every difference from text reported as a **paired bootstrap CI**.

| Development gate | Grouped CV, DeepSeek-8B |
| --- | --- |
| Floor: char tf-idf over problem text, 0.5 threshold | 0.6732 bal. acc. / 0.6686 AUC |
| A block earns a place in the fusion | Lifts fused balanced accuracy and AUC, with CI lower bounds above 0 |
| Lane B default | ≥ 0.72 balanced accuracy, in-fold threshold, and holds on olympiad labels |

**Secondary goal:** a report for the $5,000 writeup pool, which rewards generalization,
actionable interpretability, negative results, and efficiency. Report material already exists:

- the probe baseline is at chance (Finding 2);
- layer selection is what makes probes look good or bad (Finding 11);
- the `AutoTokenizer` whitespace bug (Finding 14);
- the published uncertainty features do not transfer to our setup (Finding 15).

---

## 1. Where we stand (17 Sep)

| Area | State |
| --- | --- |
| Rules, contract, leaderboard mechanics | Done: `COMPETITION.md` |
| Training set | 187 instances, **138 DeepSeek-8B (36/102)**; **no labels for the other three small-track models** |
| Text floor; E1b; E2f; E4; E10 | Done (Findings 9–18). Text + stagea is the DeepSeek model |
| E11 submission | **v2 scores 0.64 on the leaderboard** (v1: 0.47) |
| Leaderboard probes | Done (Finding 21) |
| E12 label-free signals; E13 plumbing for the three new models | Done (Finding 22) |
| E14–E17 | **Running** on Kaggle (§4) |
| Kaggle accounts | **ISHMAM1 and DESIKALAKAR tokens stopped authenticating on 17 Sep**; four accounts usable |
| Organizer questions | Drafted 17 Sep, **not sent** |

### Evidence that shapes the plan

Details are in `PROGRESS.md`:

1. **Text alone scores 0.6732 balanced accuracy** on DeepSeek; the signal is real, not
   dataset detection (Finding 9).
2. **Prompt hidden states help only when averaged and fused.** Text + stagea reaches
   balanced accuracy 0.7173 and AUC 0.7230, with both gains over text established
   (Findings 11, 16). Trace features add little (Findings 15, 18).
3. **The tokenizer bug** (Finding 14) affects the shipped probe and the organizers' baselines.
4. **Near-total difficulty shift.** There is one olympiad instance in training; the test set
   is olympiad.
5. **The leaderboard.**

   | Submission | Score |
   | --- | --- |
   | all `False` | 0.47 |
   | `True` for Qwen3.5-4B only | 0.53 |
   | E11 v1 (DeepSeek fusion, others `False`) | 0.47 |
   | **E11 v2 (v1 + Qwen `True`)** | **0.64** |

   Yesterday's probes predicted 0.53 for v2. The extra 0.11 means the hidden set, or how it
   is scored, changed overnight: an older all-false submission now also shows 0.50. **The
   16 Sep per-model conclusions need re-measuring** (§9).
6. **No label-free shortcut so far.** Forward-pass signals are problem-text proxies, and
   immediate-answer stability is noise (Finding 22). A model-specific signal needs the model
   to *reason* (E14), or labels (E15–E17).
7. **All four models run under the pins on T4s** (Findings 12, 22). Qwen3.5-4B is the
   cheapest: it fits one T4 in fp16, with a small hybrid-attention KV cache.

## 2. Models

### 2.1 Evaluated (fixed by organizers)

**Updated 16 Sep (upstream README `995f992`):** the small track evaluates four models, passed
verbatim with no aliases. `Olmo-3-7B-Instruct` was warm-up only.

| Priority | `model_id` | Params | Labels | Plumbing (Finding 22) | E11 v2 handling | Label source in progress |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | `Qwen/Qwen3.5-4B` | 4B | **none** | 33 hidden states × 2560; saved as a VLM, loads as `Qwen3_5ForCausalLM` | all `True` | E15 |
| P0 | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 8B | 138 train-ready | 37 × 4096; **needs `PreTrainedTokenizerFast`** | text + stagea fusion | E14 (signal validation) |
| P1 | `Skywork/Skywork-OR1-Math-7B` | 7B | **none** | 29 × 3584 (`Qwen2`) | all `False` | E16 |
| P1 | `allenai/Olmo-3-7B-Think` | 7B | **none** | 33 × 4096 (`Olmo3`) | all `False` | E17 |

Qwen3.5-4B moved to the top: it is the only model whose default visibly changed the
leaderboard score (Finding 21).

**Largest new gap: three of the four scored models have no labels anywhere public.** Labelling
them ourselves is limited to non-official perturbation categories (§3.3), and those may not
measure the same robustness the test labels do. Ask the organizers whether a train split for
these models is coming.

Every artifact ships per `model_id`, plus a safe fallback of `[False] * len(problems)`.

### 2.2 Development stand-ins

- `hf-internal-testing/tiny-random-LlamaForCausalLM` (locally cached): CPU smoke tests of every
  GPU pipeline.
- `Qwen/Qwen3-1.7B` and `allenai/OLMo-2-0425-1B-Instruct`: cheap GPU dry runs if needed.

---

## 3. Data

### 3.1 What we have

- **Public:** `data/small_track/`, with 138 DeepSeek train instances. Curation rules are in
  `CLAUDE.md`.
- **Olympiad pool:** `datagen/perturb/pool.py`, pinned revisions.

  | Source | Problems |
  | --- | --- |
  | AIME 2024 + 2026 | 60 |
  | AMC | 83 |
  | OlympiadBench | 362 |

  All are integer answers, with escape-sequence corruption repaired. The pool rebuilds
  identically on Kaggle.
- **Variants:** `datagen/perturb/generate.py`, 1,249 in total.

  | Category | Variants | Compliance |
  | --- | --- | --- |
  | `answer_modulus` | 649 | **distinct** |
  | `latex_to_plaintext` | 393 | *confirm* |
  | `premise_reorder` | 207 | *confirm* |

### 3.2 Gaps, ranked

1. **Olympiad difficulty (critical).** 137 of 138 DeepSeek instances are Hendrycks MATH, and
   even the E10 candidate is untested on olympiad problems.
2. **Sample size.** At n=138 most differences have CIs spanning zero, and every adopted
   ablation carries selection risk.
3. **No labels for Qwen3.5-4B, Skywork or Olmo-3-Think.** Ask the organizers (§10) while E15–E17
   build our own.
4. **No symbolic chains.** Transforms must be answer-aware by construction (done) or verified.
5. **Class imbalance.** 26% robust; thresholds must be learned in-fold.

### 3.3 Compliance with the extra-data rule

- No extra labelled data for the official types: `rephrase`, `rename`, `domain`, `distract`,
  `typos`, `expert_perturbations`, `expert_no_solution`.
- Self-generated labels use **`answer_modulus` only** until the organizers confirm the
  `confirm` categories.
- `label.py` carries the compliance tag, so an official-plus-distinct-only training set can
  always be rebuilt.

### 3.4 E7 labelling program

`datagen/perturb/sample.py` then `label.py`:

1. **Base phase.** k samples per original. Eligible if every sample is correct.
2. **Variants phase.** k samples per `distinct` variant of each eligible problem.
3. **Labelling.** Robust iff accuracy decay ≤ 0; rows go out in the `ingest_public.py` schema.

**e7b result (Finding 19).** With a DeepSeek 8k–16k cap, about a quarter of problems are
eligible, at 300–1,250 s per problem on T4s.

**Now running for the three new models** (E15–E17): base phase over the whole pool, in hash
order so a 10 h budget still samples every tier, with k=4 and a 16k cap.

| Base-phase outcome, per model | Next step |
| --- | --- |
| ≥ 60 eligible problems | Variants phase on `answer_modulus` (k=4), then `label.py` |
| 20–59 eligible | Continue the base phase next session (resumable), then variants |
| < 20 eligible | Label the AMC tier only, or ask the organizers for labels |

---

## 4. Experiment registry

Authoritative run definitions live in `datagen/kaggle_launch.py`.

| ID | Question | Account | Status | Outcome / decision it feeds |
| --- | --- | --- | --- | --- |
| E0 | How far does problem text go? | local | **Done** | 0.6732 / 0.6686: the floor |
| E1 | Prompt hidden states | ISHMAM1 | **Superseded** | Ran on the lossy tokenizer (Finding 14) |
| E1b | Prompt hidden states, correct tokenizer | ISHMAM1 | **Done** | Selection overfits; ensembles AUC 0.749; stagea block kept (Finding 11) |
| E2 / e2-bench-dtype | Trace features; bf16 vs fp16 | ISHRAQ, DIDHITI, TASHIN | **Done** | bf16 emulated and slow; fp16 3.3×; traces diverge (Finding 13) |
| E2f | Trace features in fp16 | DESIKALAKAR, DIDHITI | **Done** | Near chance alone; length leans robust (Finding 15) |
| E3 | Olmo plumbing | TASHIN | **Done** | Loads, finite, 33 hidden states (Finding 12) |
| E4 | Do hidden states over the model's own trace out-rank prompt states? | SALAM | **Done** | **No**: within E4 the best pooling is `prompt_last` (AUC 0.739) and trace-position poolings are worst (0.60–0.69). Fusing it in lifts AUC to 0.739 (+0.091 over text, established on both metrics) but not balanced accuracy, and it costs a full generation at inference (Finding 18) |
| E5 / E6 | Sampled-trace agreement; inference-time consistency | — | **Merged into E14** | |
| E7 smoke | vLLM on T4; eligibility at 4k | ISHMAM1 | **Done** | vLLM works; 4k cap useless; lossy tokenizer |
| e7b | Eligibility at 8k (AMC) and 16k (AIME), tokenizer fixed | TASHIN | **Done** | About 25% eligible; 311 s / 1,258 s per problem (Finding 19) |
| E7 | Olympiad labels for DeepSeek | fleet | Deferred | Only once the leaderboard shows DeepSeek is scored |
| E8 | Olmo-3-7B-Instruct labels | — | **Dropped** | The Instruct model was warm-up only |
| E9 | SAE features | — | Deprioritized | Report only |
| E10 | Fusion v1 | local | **Done** | Text + stagea candidate (Finding 16) |
| E11 v1 / v2 | Codabench bundle: text + stagea for DeepSeek, per-model constants | local + Codabench | **Done** | v1 0.47; **v2 (Qwen `True`) 0.64**; GPU check passes (Findings 20, 21) |
| Probes | Per-model base rates from constant submissions | Codabench | **Done; re-measure** | Qwen `True` +0.06 on 16 Sep (Finding 21) |
| E12 | Do forward-pass label-free signals predict robustness? | ISHMAM1 | **Done** | Only as problem-text proxies (Finding 22) |
| E13 | Do Qwen3.5-4B, Skywork and Olmo-3-Think run under the pins? | TASHIN, ISHRAQ, DESIKALAKAR | **Done** | Yes, on T4 in bf16 (Finding 22) |
| **E14** | **Does the model's answer survive rewrites at test time?** DeepSeek, k=2, 8k cap, original + up to 4 rewrites | ISHRAQ (s0), DIDHITI (s1) | **Done** | **No.** Agreement AUC 0.46 (0.54 within length quintiles); no fusion gain; 28k tokens per problem (Finding 24). Lane D closed |
| **E15** | Qwen3.5-4B base phase over the pool (two single-GPU vLLM engines per account) | TASHIN, SALAM | **Running** | Eligible Qwen problems, then the variants phase and Qwen labels (Lane E) |
| **E16** | Skywork-OR1-Math-7B base phase (TP=2) | TASHIN | **Running** | Skywork labels (Lane E) |
| **E17** | Olmo-3-7B-Think base phase (TP=2) | ISHRAQ | **Running** | Olmo-Think labels (Lane E) |

**Pipeline for every GPU result:**

1. `kaggle_launch.py output`.
2. Check the provenance: dtype, `tokenizer_round_trip`, and non-finite counts.
3. Run the `train/` script.
4. Record a PROGRESS Finding.
5. Update this registry.

---

## 5. Method roadmap

### Lane A: a safe leaderboard entry (done)

E11 v2 scores 0.64. It stays the fallback for every later submission.

### Lane B: text + hidden-state fusion (DeepSeek; frozen)

Text + stagea is packaged and verified (E11). No further DeepSeek modelling until the
leaderboard or the organizers show DeepSeek cases are scored.

### Lane D: test-time answer agreement (closed 17 Sep, Finding 24)

The label is literally "the answer survives perturbation", so measure that at inference
time.

1. **E14 validates it on DeepSeek's labels.** The bar is AUC above chance with a CI, and a
   nested fusion with text that beats text alone.
2. **If it passes,** it becomes the one block that works for every model without labels:
   - score the fraction of rewrite samples agreeing with the original's modal answer;
   - set the threshold from the leaderboard's robust rate.
3. **The runtime is the constraint.** E14 records tokens per problem. On the evaluation GPU
   (96 GB, HF `generate` only, no vLLM), budget about 3000 s for the whole run. If needed,
   cap tokens, use k=1, or use one rewrite.

### Lane E: self-generated labels for the three new models

E15–E17 (base phase), then the variants phase (`answer_modulus` only, compliant), then
`label.py`. Then a per-model text + stagea model trained exactly like E11. Qwen3.5-4B first.

### Lane C: SAE / transcoder (deprioritized)

Report value only; revisit after 10 Oct if time allows.

### Decision gates

| Date | Gate |
| --- | --- |
| **16 Sep, G1: decided** | Trace-uncertainty block out; **text + prompt layer ensemble** is the DeepSeek model; single-layer probes out. |
| **17 Sep, G0: passed** | A small-track entry is live (E11 v2, 0.64). |
| **17 Sep, G-D: decided (no-go)** | E14: answer agreement under rewrites is at chance once length is controlled, and adds nothing to text + stagea (Finding 24). Lane D closed. |
| **18 Sep, G-E: decided** | Qwen 90 and Skywork 157 eligible (≥ 60): variants phase launched (E18, E19). Olmo-3-Think 17 eligible at 30 problems in 9.6 h: too slow for T4s, deferred to the Fields grant (Finding 26). |
| **26 Sep** | E11 v3 submitted: the Qwen text-transfer test (Lane D failed). |
| **3 Oct, G1b** | ≥ 40 Qwen3.5-4B labels; a Qwen model scored under grouped CV. |
| **10 Oct, G2** | For each model, a trained or agreement-based model replaces the constant only where CV (and the leaderboard) beat it. |
| **27 Oct** | Freeze the best ZIP; only repeatability and timing runs after this. |

---

## 6. Compute plan

### 6.1 Fleet

Six accounts, about 30 GPU-h a week each, and at most two concurrent GPU sessions per
account. On 17 Sep:

| Account | Runs |
| --- | --- |
| ISHRAQ | E17 Olmo-Think base phase; E14 shard 0 |
| DIDHITI | E14 shard 1 (its second slot was already occupied) |
| TASHIN | E15 Qwen shards 0–1; E16 Skywork base phase |
| SALAM | E15 Qwen shards 2–3 (its second slot was already occupied) |
| ISHMAM1 | **Token rejected**: needs a new API token in `Secrets.env` |
| DESIKALAKAR | **Token rejected**: needs a new API token in `Secrets.env` |

Every run has a 10 h budget and is resumable; a second session picks up where the first
stopped.

### 6.2 Rules of engagement

- **Tokenizer:** load through `datagen/tokenizer_check.py`. It refuses any tokenizer that fails
  a round trip.
- **Dtype:** forward passes may use (emulated) bf16. **Long generation uses fp16**, never mixed
  within one dataset. vLLM for sampling: fp16, token-id prompts. Use tensor parallel 2 for 7–8B
  models; a model that fits on one T4 (Qwen3.5-4B) gets one engine per GPU via
  `kaggle_launch.per_gpu`.
- **Memory:** batch size 4 for HF generation with a static cache; expandable CUDA segments are
  set by the kernel template.
- **Outputs:** model weights in `/tmp/hf`; outputs merged on Kaggle.
- **Kaggle produces features only.** Fit every shipped artifact in the pinned venv, then
  load-test it in `Dockerfile.competition`.

### 6.3 Fields Model compute grant

Submit the form this week: https://forms.gle/oYkYDN1EtSVeSfMd9. Long-cap olympiad labelling
(16–32k tokens) and native-bf16 refits are exactly what it is for.

---

## 7. Schedule

| Week (2026) | Deliverables |
| --- | --- |
| **W1: Sep 12–19** | Done: env, data, fleet, tokenizer fix, E1b, E2f, E3, E4, E10, E7 runner, E11 v1/v2 (0.64), probes, E12, E13. **Running:** E14–E17. **Remaining:** re-measure probes, organizer message, grant form, new tokens for two accounts. |
| W2: Sep 20–26 | G-D and G-E. Variants phase for eligible Qwen problems. E11 v3 (agreement block or text transfer for Qwen). |
| W3: Sep 27 – Oct 3 | Qwen labels (G1b); Qwen text + stagea model; Skywork/Olmo variants phase. |
| W4: Oct 4–10 | G2: per-model models vs constants. Runtime profiling against 3600 s. |
| W5: Oct 11–17 | Skywork and Olmo-Think models; threshold calibration. |
| W6: Oct 18–24 | Runtime optimization; ablations for the report. |
| W7: Oct 25–31 | Freeze 27 Oct. Repeatability and timing runs. Last upload by 30 Oct. |
| Nov 1–15 | Technical report + open-source repo. |

---

## 8. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| MATH-trained signal does not transfer to olympiad | Leaderboard far below CV | E7 labels before G2; score the frozen candidate out-of-sample |
| Selection effect: adopting the best ablation (text + stagea) | Optimistic CV | Freeze the candidate now; confirm on E7 labels and the leaderboard, not by re-picking |
| E7 too expensive at long caps on T4 | No olympiad labels | e7b decides; Fields grant; label the easiest tier first |
| Tokenizer regressions (`AutoTokenizer`) | Silently corrupted features at eval | `tokenizer_check` in every loader; the E11 bundle must use it too |
| Emulated bf16 (T4) vs native bf16 (Blackwell) hidden states | Feature shift at scoring | Layer ensembles average out per-layer noise; check leaderboard vs CV after E11 |
| Three of four models unlabelled | Constants on most of the test set | E15–E17 self-labels; Lane D needs no labels; ask organizers |
| Self-labels (`answer_modulus`) measure a different robustness than the official types | Models trained on them mispredict | Validate the same pipeline on DeepSeek, where official labels exist |
| vLLM cannot run Qwen3.5 on T4 (hybrid attention kernels) | E15 fails | HF backend fallback (slower); Fields grant GPUs |
| Hidden validation set changes under us | Misread leaderboard | Re-measure probes after any unexplained jump (0.53 predicted, 0.64 observed) |
| Two Kaggle tokens rejected | A third less GPU capacity | Owners issue new tokens; runs moved to other accounts |
| 100-submission budget spent on probing | Leaderboard overfitting | Probes only for per-model base rates, never per case; about 10 used so far |
| Secrets leaking into the public repo | Compromised Kaggle accounts | `Secrets.env` ignored; `git grep --cached "KGAT_"` before every commit |

---

## 9. Immediate actions

**Needs you:**

1. **Re-measure the leaderboard** (2 submissions, `dist/probes/`):
   `solution-probe-all-false-small.zip` and `solution-probe-true-qwen35-4b-small.zip`. Upload
   both to the **Small Model Track**. With v2's 0.64 they show whether the jump came from Qwen,
   from DeepSeek, or from a changed set.
2. **Send the organizer message** (§10).
3. **Replace the ISHMAM1 and DESIKALAKAR Kaggle API tokens** in `Secrets.env`; the old ones are
   rejected.
4. Submit the Fields Model compute-grant form.

**Proceeding here:**

1. Watch E14–E17 for early failures, especially vLLM + Qwen3.5 on T4 (switch to the HF backend if
   needed).
2. When E14 finishes: `train/` evaluation of agreement vs DeepSeek labels, plus fusion with text
   (gate G-D).
3. When E15–E17 finish: eligibility counts, then launch the variants phase (gate G-E).

## 10. Open questions for the organizers

Drafted 17 Sep as one message (not yet sent):

1. Will labelled training data be released for **Qwen3.5-4B, Skywork-OR1-Math-7B and
   Olmo-3-7B-Think**? If not, may we label them ourselves with non-official perturbation types
   (answer-modulus, notation change, premise reorder)?
2. How many cases does the Competition Phase **validation set** hold, for which models, and
   did it change on 16–17 Sep?
3. Does the final **test set** cover all four small-track models, and in what proportions?
   Does `reasoning_effort` apply to the small track?
4. Can we get access to **`aimo-interp/problems-public`**?
5. May several **team members submit from their own Codabench accounts**?
6. **Bug report.** Under the pinned runtime (transformers 5.13.0), `AutoTokenizer` loads
   DeepSeek-R1-0528-Qwen3-8B as a `LlamaTokenizer` that drops all whitespace. Both
   `solutions/trained-probe/probe_inference.py` and `aimo-interp/baselines/scripts/encode.py`
   use it; `PreTrainedTokenizerFast` round-trips correctly.
