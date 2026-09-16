# Small Models Track: Execution Plan

**Scope:** AIMO Interpretability Challenge, Small Models Track (open-weight, <10B parameters).

**Key dates:**

| Date | Event |
| --- | --- |
| 16 Sep 2026 | Plan last updated |
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

## 1. Where we stand (16 Sep)

| Area | State |
| --- | --- |
| Rules, contract, leaderboard mechanics | Done: `COMPETITION.md`, verified from the Codabench API |
| Repo guide, git, GitHub | Done: `CLAUDE.md`; repository live |
| Training set from six public datasets | Done: 187 instances, **138 DeepSeek-8B (36/102)** |
| Text-only floor + confound check | Done: 0.6732 balanced accuracy, not a dataset artifact |
| Kaggle fleet + launcher | Working: `datagen/kaggle_launch.py` |
| **Tokenizer bug** (`AutoTokenizer` strips whitespace) | **Found and fixed** (Finding 14); E1 re-extracted as E1b |
| E1b: prompt hidden states | **Done + analysed** (Finding 11) |
| E2: generation traces | **Done in fp16** (Findings 13, 15); the published features are weak here |
| E3: Olmo-3-7B plumbing | Done (Finding 12) |
| E10: fusion v1 | **Done** (Finding 16); text + stagea is the Lane B candidate |
| E4: trace-state probes | **Running** (SALAM) |
| Perturbation generator (olympiad pool) | Done: 505 problems, 1,249 variants (649 compliance-distinct) |
| E7: labelling runner | Built; **e7b smoke running** (tokenizer fix, longer caps) |
| Codabench entry, organizer questions, compute grant | **Not done: need your accounts** |

### Evidence that shapes the plan

Details are in `PROGRESS.md`:

1. **Almost no public small-model signal in `val-sample`**: one DeepSeek case, zero Olmo.
2. **The shipped probe is at chance** (0.5005 mean per layer). It also tokenizes prompts with
   the lossy `AutoTokenizer` (Finding 14).
3. **Six public datasets.** The recovered label rule unlocked 138 DeepSeek instances.
4. **Text alone scores 0.6732 balanced accuracy.** The signal is real, not dataset detection.
5. **Near-total difficulty shift.** One olympiad instance in training; the test set is 100%
   olympiad. **This is still the largest risk.**
6. **Olmo-3-7B has zero labels.** Its plumbing is verified and its tokenizer is fine.
7. **Kaggle T4s: bf16 is emulated.** It is fine for short forward passes, but long generation
   must run in fp16 (3.3× faster). fp16 and bf16 greedy traces diverge (Finding 13).
8. **Prompt hidden states** (E1b):
   - Single-layer selection overfits (nested AUC 0.640).
   - All-layer ensembles reach AUC 0.749, +0.080 over text [−0.009, +0.168].
   - **Text + ensemble** under nested fusion: balanced accuracy 0.7173, AUC 0.7230. Both gains
     over text alone exclude zero.
9. **Trace-uncertainty features are near chance on our instances** (AUC 0.54). Longer traces
   lean robust, but that may be a cap artifact (Finding 15).
10. **E10 v1, all blocks.** AUC +0.069 [+0.019, +0.121], but no established balanced-accuracy
    gain. The trace block dilutes the fusion (Finding 16).
11. **E7 at a 4,096-token cap is useless for olympiad labels.** 80% (AMC) and 97% (AIME) of
    samples were truncated. That smoke run also used the lossy tokenizer; e7b re-measures.

---

## 2. Models

### 2.1 Evaluated (fixed by organizers)

| Priority | HF ID | Codabench alias | Params | Labels | Plumbing |
| --- | --- | --- | --- | --- | --- |
| P0 | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | `qwen3-8b:low` | 8B | 138 train-ready | 37 hidden states; **needs `PreTrainedTokenizerFast`** |
| P0 | `allenai/Olmo-3-7B-Instruct` | unknown (ask) | 7B | **none** | 33 hidden states; tokenizer round-trips |

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
3. **Olmo-3-7B labels.** Blocked on organizers only.
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

**Feasibility hinges on e7b.** At a 4,096-token cap almost nothing was eligible, so the
decision rule is:

| e7b outcome | Plan |
| --- | --- |
| AMC/AIME eligibility at 8k–16k caps yields ≥ 30 eligible problems per ~100 sampled | Scale the base phase over the pool, sharded across the fleet at the measured tokens/s |
| Yield too low | Label the easiest olympiad tier only, or use Fields grant GPUs for 32k caps |

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
| E5 | Sampled traces (k=4): answer agreement, length variance | fleet | Queued after e7b | Lane B; reuses the E7 sampler |
| E6 | Inference-time consistency under string transforms | fleet | Design | Lane B |
| E7 smoke | vLLM on T4; eligibility at 4k | ISHMAM1 | **Done** | vLLM works (~150 tok/s); 4k cap useless; lossy tokenizer |
| e7b | Eligibility at 8k (AMC) and 16k (AIME), tokenizer fixed | TASHIN | **Running** | Scale decision for E7 (§3.4) |
| E7 | Olympiad labels for DeepSeek | fleet | After e7b | Out-of-sample check of the E10 candidate; training data |
| E8 | Olmo labels | fleet | Blocked on organizers | Olmo artifact |
| E9 | SAE features | ISHMAM1 | Lane C, after G2 | Report |
| E10 | Fusion v1 | local | **Done** | Text + stagea candidate (Finding 16) |
| E11 | Package text + stagea as a Codabench bundle; measure runtime | local + Codabench | **Next** | First Lane B leaderboard entry |

**Pipeline for every GPU result:**

1. `kaggle_launch.py output`.
2. Check the provenance: dtype, `tokenizer_round_trip`, and non-finite counts.
3. Run the `train/` script.
4. Record a PROGRESS Finding.
5. Update this registry.

---

## 5. Method roadmap

### Lane A: a safe leaderboard entry (this week, must have)

- **Upload `solution-uncertainty-profiling-small.zip` first.** It uses the correct tokenizer
  and gives a real small-track score and runtime. The trained-probe bundle is affected by the
  tokenizer bug.
- Refitting the uncertainty model on our data is **deprioritized**: its features are near
  chance here (Finding 15).

### Lane B: text + hidden-state fusion (main bet)

1. **E11: package the candidate.** It combines three pieces:
   - the char tf-idf model;
   - per-layer probes over all 37 layers × 3 poolings of the prompt states, averaged;
   - a mean fusion with a learned threshold.

   Fit it in the pinned venv on E1b features, load the model with `PreTrainedTokenizerFast`,
   and time it. The runtime is about one forward pass per problem.
2. **Out-of-sample check (E7).** Score the frozen candidate on olympiad labels before calling
   it the default.
3. **E4 trace states.** Add them as a block only if they lift the fusion with CI lower bounds
   above zero.
4. **E6 consistency features.** Only if runtime allows.

### Lane C: SAE / transcoder (stretch, report value)

- Classify on SAE activation statistics, averaged over layers per Finding 11.
- A clean negative result is still report material.

### Decision gates

| Date | Gate |
| --- | --- |
| **16 Sep, G1: decided** | Trace-uncertainty block out; **text + prompt layer ensemble** is the Lane B candidate; single-layer probes out. |
| **26 Sep, G0** | A Lane A submission is live on the Codabench small task, and the E11 bundle is uploaded once. |
| **3 Oct, G1b** | E7 labels exist for ≥ 30 olympiad instances, and the frozen E11 candidate is scored on them. |
| **10 Oct, G2** | Lane B beats Lane A on grouped CV (with CI) and on the olympiad labels; it becomes the default submission. |
| **20 Oct, G3** | Lane C in or out. |
| **27 Oct** | Freeze the best ZIP; only repeatability and timing runs after this. |

---

## 6. Compute plan

### 6.1 Fleet

Six accounts, about 30 GPU-h a week each. This week (15–16 Sep):

| Account | Runs |
| --- | --- |
| ISHRAQ | bf16 E2 shard 0 (reference) |
| DIDHITI | bf16 E2 shard 1 (OOM), fp16 E2 shard 1 (4.1 h) |
| TASHIN | E3, dtype benchmark, e7b |
| SALAM | E4 |
| ISHMAM1 | E1, E1b, E7 smoke |
| DESIKALAKAR | fp16 E2 shard 0 (3.7 h) |

### 6.2 Rules of engagement

- **Tokenizer:** load through `datagen/tokenizer_check.py`. It refuses any tokenizer that fails
  a round trip.
- **Dtype:** forward passes may use (emulated) bf16. **Long generation uses fp16**, never mixed
  within one dataset. vLLM for sampling: fp16, tensor parallel 2, token-id prompts.
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
| **W1: Sep 12–19** | Done: env, data, fleet, launcher, tokenizer fix, E1b, E2f, E3, E10, perturbation generator, E7 runner. **Remaining:** e7b and E4 analysis, E11 bundle, Codabench uploads, organizer questions, grant form. |
| W2: Sep 20–26 | E7 base phase at scale (if e7b allows); E11 uploaded (G0); E5 launched. |
| W3: Sep 27 – Oct 3 | E7 variants phase + labels (G1b); E11 scored on olympiad labels; E8 if unblocked. |
| W4: Oct 4–10 | G2. Ablations. SAE activation dumps start. |
| W5: Oct 11–17 | E6 consistency features, timed against 3600 s. |
| W6: Oct 18–24 | G3. Threshold calibration; runtime optimization. |
| W7: Oct 25–31 | Freeze 27 Oct. Repeatability and timing runs. Final ZIPs to both tracks. Last upload by 30 Oct. |
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
| Olmo in the test set but unlabelled | Guessing on a share of cases | Ask organizers; model-agnostic fallback |
| 100-submission budget spent on probing | Leaderboard overfitting | Use submissions for pre-declared candidates only |
| Secrets leaking into the public repo | Compromised Kaggle accounts | `Secrets.env` ignored; `git grep --cached "KGAT_"` before every commit |

---

## 9. Immediate actions

**Needs you:**

1. Codabench account; upload `getting-started/dist/solution-uncertainty-profiling-small.zip`
   (COMPETITION.md §5 has the steps).
2. Send the organizer questions (§10), including the tokenizer bug report if you approve.
3. Submit the Fields Model compute-grant form.

**Can proceed here:**

1. Analyse e7b; decide E7 scale (§3.4).
2. Analyse E4 (`train_stagea_probes.py --section "E4: trace-state probes"`, then fusion with
   `--blocks text stagea tracestate`).
3. Build E11: the text + stagea submission bundle, fit in the pinned venv.
4. Update `CLAUDE.md` with the tokenizer, dtype, perturbation, and E7 workflows.

---

## 10. Open questions for the organizers

Send as one batch; the reply target is 48 h.

1. Is **Olmo-3-7B-Instruct in the small-track test set**, and under which alias?
2. `val-sample` has no Olmo rows. Is the hidden validation set richer?
3. Can we get access to **`aimo-interp/problems-public`**?
4. Do labels for **non-official perturbation categories** (answer-modulus, notation change,
   premise reorder) satisfy the extra-data rule?
5. When does the promised **train split** land, and does it include small-model olympiad
   labels?
6. May several **team members submit from their own Codabench accounts**, or should a team
   use one account?
7. **Bug report, send only if approved.** Under the pinned runtime (transformers 5.13.0),
   `AutoTokenizer` loads DeepSeek-R1-0528-Qwen3-8B as a `LlamaTokenizer` that drops all
   whitespace. `solutions/trained-probe/probe_inference.py` uses it;
   `PreTrainedTokenizerFast` round-trips correctly.
