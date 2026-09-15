# Small Models Track: Execution Plan

**Scope:** AIMO Interpretability Challenge, Small Models Track (open-weight, <10B parameters).

**Key dates:**

| Date | Event |
| --- | --- |
| 15 Sep 2026 | Plan last updated |
| 1 Nov 2026 | Codabench submissions close (6.5 weeks away) |
| 15 Nov 2026 | Technical report and open-source code due |
| 11/12 Dec 2026 | NeurIPS workshop |

**Companion documents:**

- `COMPETITION.md`: rules and contract.
- `PROGRESS.md`: what was built and measured, with numbered Findings.
- `RESULTS.md`: generated metric tables.
- `CLAUDE.md`: working conventions, commands, Kaggle mechanics.

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

Leaderboard submissions are scarce and the public sample holds a single small-model case, so
**development decisions use nested or grouped CV on our own instances**:

- Report balanced accuracy and ROC-AUC.
- Report each difference from text with a paired bootstrap CI. A single AUC at n=138
  carries about ±0.09 of noise.

| Development gate | Grouped CV, DeepSeek-8B |
| --- | --- |
| Floor: char tf-idf over problem text, no model | 0.6732 bal. acc. / 0.6686 AUC |
| A block earns a place in the fused model | Improves fused AUC over text, with the CI lower bound near or above 0 |
| Lane B default | ≥ 0.72 balanced accuracy with an in-fold threshold |

**Secondary goal:** a report for the $5,000 writeup pool, which rewards generalization,
actionable interpretability, negative results, and efficiency. Findings 2 and 11 (layer
selection is what makes probes look good or bad) are already a coherent section.

---

## 1. Where we stand (15 Sep)

| Area | State |
| --- | --- |
| Rules, contract, runtime | Done: `COMPETITION.md` |
| Repo guide | Done: `CLAUDE.md` |
| Local env (pinned venv), starter tests | Done: 25/25 pass |
| Baseline ZIPs (main + small) | Built and scored locally; **not yet on Codabench** |
| Training set from six public datasets | Done: 187 instances, **138 DeepSeek-8B (36/102)** |
| Text-only floor + confound check | Done: 0.6732 balanced acc., not a dataset artifact |
| Kaggle fleet (6 accounts) | **Verified 15 Sep**; experiments via `datagen/kaggle_launch.py` |
| E1 Stage A, DeepSeek prompt hidden states | **Done + analysed** (Finding 11) |
| E2 Stage B, DeepSeek generation traces | **Running** on ISHRAQ + DIDHITI |
| E3 Olmo-3-7B plumbing | **Done**: loads, bf16, finite, 33 hidden states (Finding 12) |
| Codabench entry, organizer questions, compute-grant form | **Not done: need your accounts** |

### Evidence that shapes the plan

Detail lives in `PROGRESS.md`; the numbers that drive decisions:

1. **Almost no public small-model signal in `val-sample`.** It has one `qwen3-8b:low` case
   and zero Olmo cases.
2. **The shipped probe is at chance.** It averages 0.5005 balanced accuracy per layer; its
   58.4% comes from picking the best fold after the fact.
3. **Six datasets exist, not two.** Recovering the label rule exactly (379/379 agreement)
   unlocked **138 DeepSeek instances** with no GPU time.
4. **Problem text alone reaches 0.6732 balanced accuracy.** It holds within single source
   datasets, so it is not dataset detection.
5. **Near-total difficulty shift.** Training has 1 olympiad instance out of 138; the private
   test set is 100% olympiad. **This is still the largest measured risk.**
6. **Olmo-3-7B has zero labels anywhere**, yet it is cached on the evaluation workers.
   Its plumbing is verified (E3).
7. **`problems-public` (the symbolic chains) is gated (401).**
8. **Kaggle T4s run bf16, the evaluation dtype, at 1 s per instance for Stage A.** The
   dtype-drift risk is retired, and forward-pass features are nearly free.
9. **Prompt hidden states: layer selection fails, layer averaging is a lead.**
   - The nested probe reaches only 0.593 AUC.
   - An unselected all-layer ensemble reaches 0.733 AUC against text's 0.669: +0.065, with
     a 95% CI of −0.024 to +0.153.
   - Fusion with text is +0.058 [−0.007, +0.124].
   - None of these is established at n=138.

---

## 2. Models

### 2.1 Evaluated (fixed by organizers)

Only two cached checkpoints are under 10B, so the small-track test set can contain only these:

| Priority | HF ID | Codabench alias | Params | Labels available | Plumbing |
| --- | --- | --- | --- | --- | --- |
| P0 | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | `qwen3-8b:low` | 8B | 138 train-ready | 37 hidden states, bf16 on T4x2 |
| P0 | `allenai/Olmo-3-7B-Instruct` | unknown (ask organizers) | 7B | **none** | 33 hidden states, bf16 on T4x2 |

Every artifact ships per `model_id`, plus a safe fallback of `[False] * len(problems)` for
unknown aliases. `meta-llama/Llama-3.1-8B-Instruct` shows up in public data (49 instances,
all spurious). It is not cached, so it is useful only for cross-model sanity checks.

### 2.2 Development stand-ins

| Model | Use |
| --- | --- |
| `hf-internal-testing/tiny-random-LlamaForCausalLM` (locally cached) | CPU smoke tests of every GPU pipeline |
| `Qwen/Qwen3-1.7B` | Same tokenizer family as the DeepSeek distill, for cheap GPU dry runs |
| `allenai/OLMo-2-0425-1B-Instruct` | Olmo-family chat template and layer naming |

### 2.3 Perturbation generator (offline data creation only)

Any open model on Kaggle GPUs, or a free-tier API locally. The generator never ships.

---

## 3. Data

### 3.1 What we have

Built by `datagen/ingest_public.py` into `data/small_track/`: **574 perturbation rows, 213
(model, problem) instances, and 187 train-ready**.

| Source dataset | Small-model rows | Notes |
| --- | --- | --- |
| `augmented-sample-math-full-filtered` | 259 DeepSeek | per-perturbation; labels derived |
| `augmented-sample-math-agg` | 141 DeepSeek | labelled 36/105; what the uncertainty baseline trains on |
| `aimo-interp-challenge-sample-full` | 64 DeepSeek | per-perturbation; olympiad-level; labels derived |
| `augmented-sample-math-agg-filtered` | 54 DeepSeek | same model under its HF name |
| `augmented-sample-math` | 156 Llama-3.1-8B | all spurious |
| `val-sample` | 1 DeepSeek | contract check only |

The curation rules are fixed; see `CLAUDE.md` → "Data and labels":

- label: `decay <= 0`
- eligibility: `base_accuracy >= 1.0`
- ambiguous band `0 < decay <= 0.3` dropped
- folds grouped by problem

### 3.2 Gaps, ranked

1. **Olympiad difficulty (critical).** 137 of the 138 DeepSeek instances are Hendrycks MATH.
   We cannot yet measure transfer from MATH to olympiad, which is what the leaderboard scores.
2. **Sample size.** At n=138, even a +0.065 AUC gain has a CI spanning zero (Finding 11).
   More labelled instances are what turn leads into results, and olympiad instances do double
   duty.
3. **Olmo-3-7B labels (blocked on organizers only).** Zero rows. Plumbing verified (E3).
4. **No symbolic chains.** Generators must use answer-preserving transforms verified by
   re-solving, not symbolic re-instantiation.
5. **Class imbalance.** 26% robust. Use balanced metrics and learn thresholds in-fold.

### 3.3 Compliance with the extra-data rule

The rule: no extra **labelled** data for the official perturbation types (`rephrase`,
`rename`, `domain`, `distract`, `typos`, `expert_perturbations`, `expert_no_solution`).
Unlabelled data, and labels for other types, are allowed.

- **Freely use** organizer-provided labelled data, including any train split they release later.
- **Self-generated labels use distinct categories only**, such as:
  - premise reordering
  - answer-format change (for example, "give the answer mod 1000")
  - unit or notation change
  - translation round-trip
- Self-generated variants of official types may be used **unlabelled**, for representation
  statistics or inference-time consistency features.
- Get the organizers' reading in writing, and document every generated set in the report.

### 3.4 Olympiad data program (the data build)

Problem pool, already in the local HF cache and all public, so none are the 30 private AIMO 3
test problems:

- `Maxwell-Jia/AIME_2024`
- `MathArena/aime_2026`, `math-ai/aime26`
- `Hothan/OlympiadBench`
- `AI-MO/aimo-validation-amc`

Keep integer-answer problems only.

For each target model M and problem X:

```
1. Base: run M on X with k=8 samples, giving base_accuracy; keep X only if base_accuracy == 1.0
2. Variants: 5 per allowed (non-official) category, answer preserved and verified by re-solving
3. Run M on each variant with k=4 samples, giving permuted_accuracy
4. Label: is_robust = (absolute_accuracy_decay <= 0), dropping 0 < decay <= 0.3
5. Emit rows in the ingest_public.py schema so the same pipeline consumes them
```

**Volume:** about 100 eligible olympiad problems for DeepSeek, enough for a real MATH-to-olympiad
transfer test. Generation throughput will be read from E2's `metadata.json` once E2 finishes,
and replaces any estimate here. Stage A costs 1 s per instance, measured.

---

## 4. Experiment registry

The authoritative run definitions live in `datagen/kaggle_launch.py`; this table records
intent and outcome. Each result feeds a named decision.

| ID | Question | Account | Status | Outcome / decision it feeds |
| --- | --- | --- | --- | --- |
| E0 | How far does problem text alone go? | local | **Done** | 0.6732 bal. acc. / 0.6686 AUC: the floor |
| E1 | Do prompt hidden states, over any layer and pooling, beat text? | ISHMAM1 | **Done** (15 Sep, 311 s) | Selection fails (nested AUC 0.593); all-layer ensemble AUC 0.733, +0.065 over text [−0.024, +0.153]. Layer ensembles become a fusion block; single-layer probes are dropped (G1) |
| E2 | Do the 14 trace-uncertainty features reproduce ~0.63–0.69 on our folds, and does fusing them with text help? | ISHRAQ + DIDHITI | **Running** (2 shards) | G1; Lane A refit |
| E3 | Does Olmo-3-7B load and run finite on T4x2? | TASHIN | **Done** | Yes: bf16, 33 hidden states, 1 s/instance. Olmo labelling is mechanically unblocked |
| E4 | Trace-state probes: do hidden states over prompt + generated trace (teacher-forced, from E2 text) out-rank prompt-only states? | SALAM | Queued after E2 | Lane B item 3 |
| E5 | Sampled traces (k=4): do answer agreement and trace-length variance add signal beyond greedy? | ISHRAQ/DIDHITI | Queued after E2 timing | Lane B inference budget |
| E6 | Inference-time consistency: rename/typo string transforms, 2–3 extra passes, answer stability | fleet | Design | Lane B |
| E7 | Olympiad labels for DeepSeek (§3.4) | ISHRAQ/DIDHITI | Needs perturbation generator | Transfer test; final training set |
| E8 | Olmo labels (§3.4) | TASHIN/SALAM | Blocked on organizers | Olmo artifact |
| E9 | SAE features on residual streams | ISHMAM1 | Lane C, after G2 | Gate G3; report |
| E10 | Fusion v1 (local): text + layer ensemble + trace features, in-fold threshold, bootstrap CI vs text | local | After E2 | G1 → Lane B default candidate |

The former E4 (fp16-vs-bf16 drift) is **retired**: Kaggle already runs bf16 (Finding 10).

**Pipeline for every GPU result:**

1. `kaggle_launch.py status`, then `output <name>`.
2. Check the provenance: `metadata.json` dtype and the non-finite counts.
3. Run the matching `train/` script, which upserts its section of `RESULTS.md`.
4. Add a numbered Finding to `PROGRESS.md`.

---

## 5. Method roadmap

### Lane A: a safe leaderboard entry (this week, must have)

- Upload `dist/solution-trained-probe-small.zip` and `dist/solution-uncertainty-profiling-small.zip`
  to Codabench to prove the path end to end and get real runtime numbers. This needs a
  Codabench account (you).
- After E2, refit the uncertainty model on our 138 instances in the pinned venv. Ship it only
  if its grouped CV beats the shipped artifact's 0.6262.
- Never lose a valid, working submission.

### Lane B: fused behavioural + white-box signal (main bet, weeks 2–5)

Findings 2 and 11 reshape this lane. **Choosing one layer is out**: it fits noise in both the
shipped baseline and our nested test. Averaging over layers is in, as one block among several.
The priorities, in order:

1. **Fusion of what works (E10).** Combine text features, trace uncertainty (E2), trace shape
   (length, token-cap hits, reflection markers), and the prompt layer ensemble (E1).
   - Use a grouped-CV ensemble with thresholds learned in-fold.
   - Add a block only if it lifts fused AUC with a CI lower bound near zero or above.
   - Target: mid-70s balanced accuracy.
2. **Trace-state probes (E4).** Hidden states over the model's own reasoning, with layer
   ensembles. This is cheap (one forward pass over ≤6k tokens) and is the most direct test of
   whether internals track robustness rather than problem surface.
3. **Inference-time consistency (E6).** Answer and logit stability under cheap string
   perturbations generated inside the container. It is compliant because nothing is labelled,
   and it must fit the 3600 s budget alongside the base generation.

### Lane C: SAE / transcoder (weeks 3–6, stretch + report value)

- Train a small SAE on DeepSeek residual streams from math prompts, or use a public one if it
  appears. Classify on activation statistics, averaging over layers per Finding 11.
- A clean negative result is still report material.
- Activation dumping goes to ISHMAM1. Stage A timing (1 s per instance) makes this cheaper
  than planned.

### Decision gates

| Date | Gate |
| --- | --- |
| **19 Sep (G1)** | *E1 half decided:* single-layer probes out, layer ensemble kept as a fusion block. *Pending E2:* trace features below 0.65 balanced accuracy mean the olympiad program (E7) and consistency (E6) get the fleet ahead of more feature work. |
| **26 Sep (G0)** | A Lane A submission is live on the Codabench small task. If not, all effort goes to packaging. |
| **10 Oct (G2)** | Lane B beats Lane A on grouped CV (with CI) **and** on the olympiad holdout from E7, and becomes the default submission. |
| **20 Oct (G3)** | Lane C in or out. If out, it becomes report material only. |
| **27 Oct** | Freeze the best ZIP; only repeatability and timing runs after this. |

---

## 6. Compute plan

### 6.1 Fleet

Six accounts, each with about 30 GPU-h a week and 12 h sessions, verified 15 Sep. That is
about 180 GPU-h a week in total, and all runs go through `datagen/kaggle_launch.py`
(mechanics and gotchas in `CLAUDE.md`).

| Alias | Username | Standing role | This week |
| --- | --- | --- | --- |
| ISHRAQ | `farhanishraqq` | coordinator; DeepSeek generation | E2 shard 0 (running) |
| DIDHITI | `didhitinahid` | DeepSeek generation | E2 shard 1 (running) |
| TASHIN | `tashintahir` | Olmo generation | E3 done; free |
| SALAM | `salam2026` | Olmo generation | E4 after E2 |
| ISHMAM1 | `ishmamahmid` | activation dumps | E1 done; free |
| DESIKALAKAR | `fatkhato` | sweeps, repeatability | reserve |

This week used about 0.2 GPU-h for E1 + E3; E2 is the only substantial job. The binding
constraint is labelled data, not compute.

### 6.2 Rules of engagement

- Use T4x2 (`NvidiaTeslaT4`) for 7B/8B work. Kaggle's torch reports bf16 on the T4
  (emulated), so runs match the evaluation dtype. Still check `metadata.json` dtype and the
  non-finite guards on every result.
- Keep model weights in `/tmp/hf`; outputs in `/kaggle/working` (20 GB cap). Merge on Kaggle,
  and download with `kaggle_launch.py output`.
- **Kaggle produces features only.** Fit every shipped artifact locally in the pinned venv,
  then load-test it inside `Dockerfile.competition`.
- For every generation job:
  - write it shardable and resumable
  - record provenance in `metadata.json`
  - make its merger refuse inconsistent shards

### 6.3 Fields Model compute grant

Submit https://forms.gle/oYkYDN1EtSVeSfMd9 this week, asking for H200 hours for SAE training
and olympiad-scale labelling. Up to 10,000 H200-h is available. This needs you.

---

## 7. Schedule

| Week (2026) | Deliverables |
| --- | --- |
| **W1: Sep 12–19** | Done: env, training set, text floor, fleet, launcher, CLAUDE.md, E1 + E3 run and analysed. **Remaining:** E2 analysis + E10 fusion v1 (G1), Codabench account + first two uploads, organizer questions, compute-grant form. |
| W2: Sep 20–26 | Perturbation generator for non-official categories; olympiad problem pool filtered; E4 and E5 launched; Lane A refit uploaded (G0). |
| W3: Sep 27 – Oct 3 | E7 olympiad labels for DeepSeek; E8 if unblocked. Lane B v1 fusion trained and submitted. |
| W4: Oct 4–10 | G2: Lane B vs Lane A on grouped CV and the olympiad holdout. Ablations on DESIKALAKAR. SAE activation dumps start. |
| W5: Oct 11–17 | E6 consistency features integrated and timed against the 3600 s budget. Second labelling wave on weak spots. |
| W6: Oct 18–24 | G3. Ensemble and threshold calibration. Runtime optimization. |
| W7: Oct 25–31 | Freeze 27 Oct. Three repeated Codabench runs for determinism and timing. Final ZIPs to **both** tracks (with and without `small.txt`). Last upload by 30 Oct. |
| Nov 1–15 | Technical report + open-source repo, drafted from W4 onward. |

---

## 8. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| MATH-trained signal does not transfer to olympiad | Leaderboard far below CV | E7 olympiad holdout before G2; prefer model-behaviour features over text features |
| n=138 too small to separate methods | Chasing noise; wrong block choices | Paired bootstrap CIs on every comparison; grow data via E7/E8 before fine-grained tuning |
| Olmo in the test set but unlabelled | Guessing on a large share of test cases | Ask organizers now; E3 plumbing done; fallback is a model-agnostic trace model |
| Emulated bf16 (T4) vs native bf16 (Blackwell) numeric differences | Small feature shift at scoring | Prefer layer-averaged and rank-based features; confirm predictions on a Codabench run |
| Self-generated labels ruled non-compliant | Retraining | Non-official categories only, written confirmation, and keep an official-data-only variant |
| 3600 s blown by generation + consistency passes | Invalid submission | Time every candidate on Codabench early (Lane A gives real timings); cap tokens |
| Codabench congestion near the deadline | Missed upload | Freeze 27 Oct; final upload by 30 Oct |
| Library version drift in artifacts | Silent load failure | Pinned venv for all final fits; ZIP load test in Docker |
| Workspace not under version control | Lost work; report reproducibility | `git init` with `Secrets.env` excluded before the next code change |

---

## 9. Immediate actions (this week)

**Needs you (account-bound):**

1. Create a Codabench account, join competition 16180, and upload
   `getting-started/dist/solution-trained-probe-small.zip` and
   `solution-uncertainty-profiling-small.zip` (G0, plus real runtime numbers).
2. Send the organizer questions in §10 on Discord or aimo.interp@gmail.com.
3. Submit the Fields Model compute-grant form.
4. Decide on `git init` and a private GitHub remote. The report requires open-sourcing, and
   `Secrets.env` must stay excluded.

**Can proceed here:**

1. When E2 finishes: `kaggle_launch.py output e2-stageb-deepseek-s0 e2-stageb-deepseek-s1`,
   then `train/train_stageb_features.py`, and record the Finding.
2. Build E10 fusion v1 on the saved out-of-fold probabilities (`stagea_oof.npz` + Stage B) to
   close G1.
3. Register and push E4 (trace-state probes) on SALAM.
4. Start the non-official-category perturbation generator and filter the olympiad pool (§3.4).

---

## 10. Open questions for the organizers

Send as one batch; the reply target is 48 h.

1. Is **Olmo-3-7B-Instruct in the small-track test set**, and what alias does the harness pass?
2. The website says the validation set covers every model type in the test set, but
   `val-sample` has no Olmo rows. Is the full validation set richer?
3. Can we get access to **`aimo-interp/problems-public`** (symbolic chains)?
4. Does generating labels for **non-official perturbation categories** satisfy the extra-data rule?
5. Is **olympiad-difficulty labelled data** for the small models planned beyond the one
   instance currently public, and when does the promised train split land?
