# Small Models Track: Execution Plan

**Scope:** AIMO Interpretability Challenge, Small Models Track (open-weight, <10B parameters).

**Key dates:**

| Date | Event |
| --- | --- |
| 29 Sep 2026 | Plan last updated. **Start at §12: the handover note and current decisions.** |
| 1 Nov 2026, 12:55 UTC | Codabench submissions close (about 6.5 weeks away) |
| 15 Nov 2026 | Technical report and open-source code due |
| 11/12 Dec 2026 | NeurIPS workshop |

**Companion documents:**

- `COMPETITION.md`: rules, contract, leaderboard mechanics.
- `PROGRESS.md`: what was built and measured, with numbered Findings.
- `RESULTS.md`: generated metric tables.
- `CLAUDE.md`: working conventions and commands.

**Repository:** https://github.com/farhanishraq17/AIMO-Interpretability-Challenge-SMALL---Team-BakeNekos

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

## 1. Where we stand (26 Sep)

| Area | State |
| --- | --- |
| Rules, contract, leaderboard mechanics | Done: `COMPETITION.md` |
| Training set | 226 instances: **153 DeepSeek-8B (46/107)**, 49 Llama, and the first official rows for the other three models via `train-main-v2` (Qwen 12, Skywork 3, Olmo-Think 9) — enough to check a constant, not to fit a model (Findings 30, 44) |
| Text floor; E1b; E2f; E4; E10 | Done (Findings 9–18). Text + stagea is the DeepSeek model |
| Submission | **`e27k` scores 0.86** (12/14), above the 0.79 that led on 21 Sep. `e27e` 0.71 prior-free, E11 v2 0.64 zero-GPU fallback (Findings 49, 60, 61) |
| Leaderboard probes | Done (Finding 21) |
| E12 label-free signals; E13 plumbing for the three new models | Done (Finding 22) |
| E14 | Done: answer agreement is at chance (Finding 24). Lane D closed |
| E15–E19 self-labelling | **Closed** by E20: `answer_modulus` does not measure official robustness (Finding 31) |
| E21 | Done: the model does not transfer to olympiad problems and there is no cross-model signal (Finding 32) |
| `nakib_v2` port | Findings 33–43 merged by hand; models agree 20/20 on official rows (Finding 38) |
| E27 series | **Done.** Familiarity ranking + per-model robust counts; every count neighbour is worse (Findings 49, 58, 60, 61) |
| E28–E37 | Done by the parallel session: rank successors, activation geometry, counterfactuals, external holdout (Findings 50–57) |
| Main Track | Its own repository, `https://github.com/farhanishraq17/AIMO-Interpretability-Challenge-LARGE---Team-BakeNekos`; L1 panel complete and ungraded (LF-1 to LF-10) |
| Organizers | Answered 26 Sep: `problems-public` is private, labels are final, generating data from `train-main-v2` is allowed (Finding 59) |
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
4. **Near-total difficulty shift, now measured rather than feared.** Training is almost all
   Hendrycks MATH; the test set is olympiad. Against the official olympiad rows the fused
   model scores AUC 0.429, the text block 0.617 with a CI spanning 0.5, and a refit layer
   ensemble 0.476 (Findings 32, 39).
5. **Robustness is a property of the problem, not the model:** the four models agree on 20 of
   20 shared official rows, p = 0.0005 (Finding 38). Labels are therefore poolable across
   models even though our *features* do not transfer.
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

- **Superseded 19 Sep:** the current competition website explicitly retires the prohibition
  on extra labeled data. Labels for official perturbation families may now be generated and
  used, provided no evaluation data leak is involved.
- Existing E15–E24 results are still scientifically valid: they tested whether the cheap
  `answer_modulus` proxy transferred, and it did not. The rule change does not rescue those
  labels; it permits a new dataset built from the actual official-family symbolic programs.
- `label.py` retains its compliance tags for provenance and ablations, not as a current legal
  restriction.

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
| E15 | Qwen3.5-4B base phase over the pool (two single-GPU vLLM engines per account) | TASHIN, SALAM, RSD | **Stopped** | 84 of 270 eligible; labels set aside after E20 (Finding 31) |
| E16 | Skywork-OR1-Math-7B base phase (TP=2) | ISHRAQ | **Stopped** | 143 of 232 eligible; labels set aside (Finding 31) |
| E17 | Olmo-3-7B-Think base phase (TP=2) | Labib | **Stopped** | 18 of 32 eligible once its output was recovered; too slow on T4s (Findings 26, 42) |
| E18 / E19 | Variants phase for Qwen and Skywork | RSD, Labib | **Stopped** | 47 Qwen and 21 Skywork self-labels, all set aside (Finding 31) |
| **E20** | Do our `answer_modulus` self-labels agree with official ones? | ISHRAQ | **Done** | **No.** kappa -0.135, drops correlate -0.25; self-labelling stops (Finding 31) |
| **E21** | Does the DeepSeek model transfer to olympiad problems, and is there a cross-model signal? | DIDHITI, SALAM, TASHIN, ISHRAQ | **Done** | **No to both.** AUC 0.429 on the 10 unseen olympiad rows; every leave-one-model-out fold at chance (Finding 32) |
| Gapfill (`nakib_v2`) | Base samples for problems whose variants were already paid for | NAKIB | **Done** | 26 Qwen and 23 Skywork self-labels (Finding 37), invalidated by E20 |
| Stage A olympiad (`nakib_v2`) | Do prompt states carry olympiad signal where text does not? | MR | **Done** | AUC 0.844 vs text's 0.688 on 12 official Qwen rows, but the strict control gives p = 0.073 and the training labels were self-labels (Finding 41) |
| **E22** | Problem-level acceptance harness on `train-main-v2`, template-grouped | local | **Done** | Nothing clears chance; problem length is best at AUC 0.635, CI [0.42, 0.84] (Finding 46) |
| **E23** | Do familiarity/contamination scores (Min-K%, zlib, suffix NLL) predict robustness? | RSD (Qwen, DeepSeek), Labib (Skywork, Olmo-Think) | **Done** | **G-M fails**, but on olympiad problems familiar problems lean spurious (mean NLL 0.69 unoriented, CI to 0.49); tested on the leaderboard as E27d (Finding 47) |
| **E24** | Do `answer_modulus` self-labels agree with official labels *on olympiad problems*? | ISHRAQ (E24a); TASHIN, ISHRAQ, RSD, Labib (E24b) | **Done** | **G-V2 fails.** kappa +0.286 [-0.19, 1.00] on 15 pairs against 0.72 chance agreement; the direction flips from E20 but the yield is 15 pairs per 15.8 GPU-h (Finding 48) |
| **E27** | Can per-problem predictions from official labels beat the 0.64 constant ceiling on the leaderboard? | local + Codabench | **Built**, awaiting upload | Three ZIPs (§11.6); the constants cap is 9/14 (Finding 45) |

**Pipeline for every GPU result:**

1. `kaggle_launch.py output`.
2. Check the provenance: dtype, `tokenizer_round_trip`, and non-finite counts.
3. Run the `train/` script.
4. Record a PROGRESS Finding.
5. Update this registry.

---

## 5. Method roadmap

### Lane A: a safe leaderboard entry (done)

E11 v2 scores 0.64. Superseded on the leaderboard by `e27e` (0.71) and `e27k` (0.86), but it
remains the zero-GPU fallback any later submission can fall back to (Findings 49, 60).

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

### Lane E: answer-modulus self-labels — CLOSED (20 Sep); official-family lane conditionally reopened 25 Sep

E15–E19 and the two `nakib_v2` gapfill runs built the pipeline and produced 26 Qwen and 23
Skywork `answer_modulus` labels. E20 then ran the same pipeline on 66 DeepSeek problems that
carry official labels: agreement 0.29, Cohen's kappa −0.135, drops correlating −0.25
(Finding 31). Under the rule then in force, `answer_modulus` was the only labelable category,
so that lane closed. The 19 Sep website clarification now permits labels from the official
families, but a replacement lane remains blocked on access to their symbolic programs in
`problems-public`. The old records stay as a documented negative result.

One ported result rests on those labels and is retained only as a measurement: the Stage A
olympiad probe of Finding 41 (AUC 0.844 on 12 official Qwen rows, strict control p = 0.073).
Finding 32 has since asked the same question with official labels on both sides and found
chance.

**What replaces it:** `aimo-interp/train-main-v2`. 153 DeepSeek instances plus 34 official
small-track rows are the whole training set, and those 34 rows are the acceptance test.

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
| **20 Sep, G-V: decided (fail)** | E20: self-labels agree with official ones at kappa -0.14, with drops correlating at -0.25. `answer_modulus` does not measure official robustness, so E15–E19 labels are set aside and the labelling program stops (Finding 31). |
| **20 Sep, G-T: decided (fail)** | E21: the shipped model scores AUC 0.429 on the 10 official olympiad DeepSeek rows and pooled uncertainty scalars score at chance across models (Finding 32). No v3 follows from a model; E11 v2's constants stand. |
| **19 Sep, G-T (`nakib_v2`): decided (no-go)** | E11 v3 as "the text block for every model" is rejected: the MATH-trained block reaches AUC 0.617 on the official rows with CI [0.391, 0.833], and 0.476 on DeepSeek itself (Finding 39). Reached independently of our own G-T. |
| **~~3 Oct, G1b~~ withdrawn** | Was "≥ 40 Qwen3.5-4B labels". No longer meaningful after G-V: the labels it counted do not measure the target. |
| **10 Oct, G2 (revised)** | A trained model replaces a constant only where it beats it **on the 34 official rows** as well as under CV. A model that cannot clear always-robust (0.706 on those rows) does not ship. |
| **29 Sep, G-V2** | E24: `answer_modulus` self-labels vs official labels on olympiad problems. Kappa ≥ 0.3 with CI above 0 reopens Lane E (E25); otherwise it closes for good (§11). |
| **29 Sep, G-M** | E23: familiarity scores become a block only with a problem-level CI lower bound above 0.5 and within-length AUC > 0.6 on MATH (§11). |
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
| MATH-trained signal does not transfer to olympiad | Leaderboard far below CV | **Measured, and it does not** (Findings 32, 39): AUC 0.429 on the official olympiad rows. The constants carry the submission; do not ship a model that cannot clear always-robust there |
| Selection effect: adopting the best ablation (text + stagea) | Optimistic CV | Freeze the candidate now; confirm on E7 labels and the leaderboard, not by re-picking |
| E7 too expensive at long caps on T4 | No olympiad labels | e7b decides; Fields grant; label the easiest tier first |
| Tokenizer regressions (`AutoTokenizer`) | Silently corrupted features at eval | `tokenizer_check` in every loader; the E11 bundle must use it too |
| Emulated bf16 (T4) vs native bf16 (Blackwell) hidden states | Feature shift at scoring | Layer ensembles average out per-layer noise; check leaderboard vs CV after E11 |
| Three of four models near-unlabelled | Constants on most of the test set | `train-main-v2` gives 34 official rows to check constants against (Finding 30); self-labelling is closed (Finding 31); ask the organizers whether more is coming |
| Our constants contradict the official labels | `False` for Skywork and Olmo may be backwards | Those rows are labelled 2/3 and 7/9 robust (Finding 33), but 12 rows cannot outvote a leaderboard probe; re-test if more labels arrive |
| Self-labels (`answer_modulus`) measure a different robustness than the official types | Models trained on them mispredict | Validate the same pipeline on DeepSeek, where official labels exist |
| vLLM cannot run Qwen3.5 on T4 (hybrid attention kernels) | E15 fails | HF backend fallback (slower); Fields grant GPUs |
| Hidden validation set changes under us | Misread leaderboard | Re-measure probes after any unexplained jump (0.53 predicted, 0.64 observed) |
| Two Kaggle tokens rejected | A third less GPU capacity | Owners issue new tokens; runs moved to other accounts |
| Submission budget spent on probing | Leaderboard overfitting | The cap is **300** per person in the Competition phase, not 100 (Finding 35), so the budget is not the binding constraint; the ~14-case validation set is. Probes only for per-model base rates, never per case; about 10 used so far |
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

1. **Nothing is running.** Every lane that needed GPU is closed: D (E14), E (E20), and the
   transfer/cross-model questions (E21).
2. **Group by template, not by text,** wherever `train-main-v2` is used (Finding 34). Our
   folds group on `problem_key`, which splits a template across folds in that table.
3. **Keep the report current as results land** — the four negative results (E12, E14, E20, E21)
   plus the tokenizer bug and the probe-baseline selection artifact are the substance of it.

## 10. Open questions for the organizers

Drafted 17 Sep as one message (not yet sent):

1. **Answered, 26 Sep (Finding 59):** the split will **not** grow beyond "a couple samples"
   for Skywork, and generating additional training data from `train-main-v2` is explicitly
   allowed. The acceptance set is final at 34 small-track rows.
2. **Partly self-answered (Finding 35):** the small-track task was recreated 16 Sep 22:35 UTC
   as v8, and the `train-main-v2` commit log shows 7 rows withheld "used by the small-track
   validation set" that day. Still ask: how many cases does the validation set hold, and for
   which models?
3. Does the final **test set** cover all four small-track models, and in what proportions?
   Does `reasoning_effort` apply to the small track?
4. **Answered, 26 Sep:** `aimo-interp/problems-public` is **private and cannot be shared**;
   the 401 is intentional. E33, which depended on its symbolic programs, is dead. As of 25 Sep, search indexing shows
   a 22-row symbolic validation split, but the live Hub API returns 401 and the dataset is not
   in the public `aimo-interp` listing. The getting-started repository restricts new issue
   creation, and issue #6 already asks whether self-generated counterfactual evaluation is
   permitted but has no organizer response.
5. May several **team members submit from their own Codabench accounts**?
6. **Bug report.** Under the pinned runtime (transformers 5.13.0), `AutoTokenizer` loads
   DeepSeek-R1-0528-Qwen3-8B as a `LlamaTokenizer` that drops all whitespace. Both
   `solutions/trained-probe/probe_inference.py` and `aimo-interp/baselines/scripts/encode.py`
   use it; `PreTrainedTokenizerFast` round-trips correctly.


---

## 11. Refined plan (22 Sep): the next experiments

**This section supersedes §7 (schedule) and §9 (immediate actions).** Earlier sections are kept
as the record of how the plan got here.

### 11.0 Why this plan

As of 22 Sep every modelling lane is closed:

- self-labelling failed E20 (kappa −0.135 on DeepSeek MATH problems, Finding 31);
- the DeepSeek model does not transfer to olympiad problems (AUC 0.429, Finding 32);
- there is no cross-model feature signal (Finding 32);
- answer agreement at test time is at chance (Finding 24).

E11 v2 (0.64) is the submission. Five things shape what is worth doing in the six weeks left:

1. **Labels are the only bottleneck, and the test that closed labelling ran on the wrong
   population.** E20 compared `answer_modulus` self-labels with official labels on *Hendrycks
   MATH* problems. Every other signal we measured broke on the MATH→olympiad shift, so G-V's
   failure may not hold on olympiad problems either. The ported Finding 41 is weak evidence that
   it does not (AUC 0.844 on 12 official Qwen rows, strict control p = 0.073).
   `train-main-v2` lets us run the comparison directly, on its 20 labelled olympiad problems
   (34 rows) and under the current official perturbation families. If self-labels agree there,
   labelling reopens with records we already hold: 47 Qwen and 21 Skywork self-labels plus the
   E16/E17 base phases. **This is the highest-upside experiment left, at about 12–15 GPU-h.**
2. **Robustness is a property of the problem** (Finding 38: 20/20 agreement, 83/85 including
   gpt-oss). Evaluation should therefore be problem-level: labels pooled across every
   configuration, grouped by *template* rather than text (Finding 34).
3. **One cheap, hypothesis-driven signal is untested: familiarity and contamination.**
   "Spurious" means the model solves the original but not its perturbations, which is what
   memorization looks like. The organizers' final report splits results by published vs
   unpublished problems. E12 measured only mean problem NLL on MATH: 0.68 AUC (sign-flipped),
   0.58 within length quintiles (Finding 22). Membership-inference scores designed to remove
   text predictability (Min-K%, zlib ratio, suffix-completion NLL) were never tried, and never
   on olympiad problems. It costs one forward pass per problem and is report material even if
   null.
4. **With 20–33 labelled olympiad problems, nothing ships without clearing G2.** A candidate
   must beat always-robust on the official rows with prevalence-independent metrics, then hold
   on the leaderboard (≤ 3 submissions against ~14 cases).
5. **The report is worth protecting.** The writeup pool rewards negative results and
   generalization, and we have many. It gets its own track from now, not the final week.

### 11.1 Experiments, in priority order

| ID | Question | Compute | Reuses | Gate |
| --- | --- | --- | --- | --- |
| **E22** | **Problem-level acceptance harness.** Pool `train-main-v2` labels across *all* configurations, group by template, and score any per-problem signal with a bootstrap CI and a permutation test. Also: does problem-level agreement hold on MATH (DeepSeek vs Llama on their 34 shared train problems)? | CPU, ~1 day | `scripts/inspect_train_main_v2.py` (`cluster_templates`), `train/eval_train_main_v2.py` (agreement, bootstrap), `train/results_io.py` | Infrastructure: every later gate is scored here |
| **E24a** | **Answer key** for the 20 labelled official problems. 8 overlap our olympiad pool, so their answers are known. The other 12 get Skywork base sampling (k=8) with a placeholder answer, and the reference is the unanimous value in the records' `answers` field. Problems without unanimity are dropped. | ~1 GPU-h | `datagen/perturb/sample.py` (`--pool`, `--problem-keys`), the `deepseek_check_pool.py` pattern | Needs ≥ 12 problems with an integer answer ≥ 13 |
| **E24b** | **Gate G-V2: do `answer_modulus` self-labels agree with official labels on olympiad problems?** The full E7 pipeline (base k=4, then `answer_modulus` variants k=4) runs for each model on its own official problems. Qwen gets one engine per T4, DeepSeek and Skywork TP=2, and Olmo-3-Think goes last because it is slow. | ~12 GPU-h over 4 accounts | `base_sampling`, `variant_sampling`, `per_gpu`, `datagen/perturb/label.py`, E20's comparison | **Pass:** Cohen's kappa ≥ 0.3 with a bootstrap CI above 0 → Lane E reopens (E25). **Fail:** Lane E closes for good, and the report carries E20 and E24 together |
| **E23** | **Familiarity and contamination signals** for all 36 official problems plus the 153 DeepSeek MATH problems, under all four models. Signals: mean NLL, Min-K% (k = 10, 20), zlib ratio, suffix NLL given the first half, and each model's NLL relative to the other three. | ~2 GPU-h (4 × ~20 min) | `datagen/extract_label_free.py` (`mean_nll`, `score_problem`), extended in place | **Gate G-M:** problem-level AUC with CI lower bound > 0.5 on E22 **and** within-length AUC > 0.6 on MATH → candidate block |
| **E30** | Architecture-invariant norm/cosine trajectories from the existing E21 all-layer archives. | CPU only | E21 archives | **Closed:** 12-way max-T p=0.2049; selected signal reverses on MATH (AUC 0.348) |
| **E31** | Does the labelled model's prompt state remain stable under deterministic answer-preserving rewrites? Five signals fixed before extraction. | 0.19 GPU-h on TASHIN/RSD | E14 rewrites, E21 prompt path | **Closed:** best AUC 0.735, template CI [0.380, 0.938], max-T p=0.0667, Qwen AUC 0.531 |
| **E32** | One independent confirmation of E31's selected all-layer cosine sensitivity and fixed direction on 142 DeepSeek MATH rows. | 0.21 h on TASHIN | E31 extractor | **Closed:** AUC 0.545, CI [0.428, 0.658], within-length 0.510, split BA 0.527 |
| **E33** *(blocked on access)* | Generate official-family labels from the symbolic programs in `problems-public`, fit one frozen cross-model stability model, and evaluate once on `train-main-v2` without tuning on it. | Estimate only after inspecting the 22 programs | `problems-public`, E7 sampler, E22 harness | **Start only after access. Pass:** problem AUC >= 0.75, 19-template CI lower > 0.5, worst non-degenerate per-model AUC >= 0.60, and balanced accuracy beats the constants. One Codabench submission maximum. |
| **E34a** *(closed 25 Sep)* | Can the two already-built official-like surface families produce enough genuinely new AIME labels to justify scaling? | TASHIN, 4.65 h | AIME 2024/2026 pool, E7 sampler | **Failed:** 1/52 eligible, one robust label, no spurious label |
| ~~E34b~~ *(dropped)* | Fit one frozen character model on new OlympiadBench+AIME-2024 labels and score once on untouched AMC+AIME-2026 labels | — | Exact E34a generator and label bands | Not launched because E34a failed its prerequisite gate |
| **E35** *(closed 25 Sep)* | Train on published MATH-Perturb robustness outcomes and score once on source-disjoint Putnam-AXIOM | CPU | `ssubhnil/Qwen3-Math-Eval` pinned at `f64c82f`; frozen char+word model | **Failed:** holdout AUC 0.609, CI [0.442, 0.769]; no official transfer or submission |
| **E36** *(pre-registered 25 Sep)* | Do original-answer stability and reasoning-path diagnostics across four token budgets predict independently measured perturbation robustness? | CPU first; GPU only after external gate | Same pinned archive, but model-specific labels and original traces only | Train on MATH-Perturb; score Putnam once; all gates frozen in `protocols/E36_PROTOCOL.md` |
| **E37** *(pre-registered 25 Sep)* | Does E36's unchanged ranking signal confirm on untouched GSM-Plus clear-cut perturbation outcomes? | CPU | Frozen E36 model; within-model half-rank fixed before GSM labels | Pass all gates in `protocols/E37_PROTOCOL.md` before any AIMO GPU extraction |
| ~~E25~~ *(dropped 23 Sep)* | Olympiad labelling at scale. G-V2 failed, and E24 showed the yield would be about one label per GPU-hour on problems selected for difficulty (Finding 48). | — | — | — |
| E26 *(only if G-M or G-V2 passes)* | Single encoder: one small-track model's Stage A features used as a *problem* representation for every `model_id`, trained on pooled problem-level labels | ~1 GPU-h | `extract_features.py --instances`, `train/eval_official_holdout.py` | G2, plus an organizer ruling (§11.4) |

**Deliberately not planned:**

- **More leaderboard probes.** The constants are settled (Finding 27).
- **Flipping the Skywork/Olmo constants.** That would be 12 official rows against a direct
  leaderboard measurement (Finding 33).
- **Anything that generates at inference.** E14 needed 28,000 tokens per problem.
- **SAE work.** Lane C stays report-only.

### 11.2 Gates

These are also added to the §5 table.

| Date | Gate |
| --- | --- |
| **23 Sep, G-V2: decided (fail)** | E24: on olympiad problems self-labels agree at kappa +0.286, CI [-0.19, 1.00], against chance agreement of 0.72 - the sign flips from E20's -0.135 but the gate is not cleared, and only 15 of 27 pairs could be labelled at all for 15.8 GPU-hours. **Lane E stays closed; E25 is dropped** (Finding 48). |
| **24 Sep, G-M: decided (fail)** | E23's seven pre-registered familiarity signals all fail: no candidate has both an olympiad CI lower bound above 0.5 and MATH within-length AUC above 0.6. E27/E28's later rank policies are reported as controlled leaderboard follow-ups, not as a reopened candidate block. |
| **10 Oct, G2** (as revised in §5, made concrete) | A candidate ships as E11 v3 only if all three hold: its problem-level AUC on E22 has a CI lower bound above 0.5; it beats always-robust in balanced accuracy on the official rows; and one leaderboard submission scores ≥ 0.64. Otherwise v2 is frozen. |
| **27 Oct** | Freeze. After that, only repeatability and timing runs. |

### 11.3 Schedule (supersedes §7)

| Week (2026) | Experiments | Report track |
| --- | --- | --- |
| **Sep 25–28** | E34a failed; request `problems-public` access; preserve E27e/E27h (0.71) and E11 v2 (0.64). | Add E30–E34a and the rule correction to the record. |
| Sep 29 – Oct 5 | Only if `problems-public` access arrives: inspect it and execute frozen E33. Otherwise no new experiment. | Figures: layer curves (Finding 11), transfer failure (Findings 32, 39, 53), 20/20 agreement (Finding 38) |
| Oct 6–12 | G2 (10 Oct). If it passes: E11 v3 build, Docker load test, GPU check. | Method and data-provenance sections: label-rule archaeology, the `pool_id` bug, Finding 44 |
| Oct 13–26 | Runtime and repeatability; the ablations the report needs | Full draft |
| Oct 27 – Nov 1 | Freeze 27 Oct; last upload by 30 Oct | Polish |
| Nov 1–15 | — | Final report and cleaned open-source repo |

### 11.4 Needs you (supersedes §9)

1. **Sent, and answered on 26 Sep** (Finding 59): `aimo-interp/problems-public` is a private
   dataset that will not be shared, the official labels are final bar a couple of Skywork rows,
   and generating additional training data from `train-main-v2` is explicitly allowed.
   **E33 is dead** - it needed those symbolic programs.
2. **Still unasked:** may a solution load a cached small-track model other than the `model_id`
   it was called with? E26 depends on the answer, and nothing else does.
3. **Best submission is `e27k` at 0.86** (12/14, Findings 60-61), with `e27e` at 0.71 as the
   prior-free version and E11 v2 at 0.64 as the zero-GPU fallback.

### 11.5 Housekeeping, bundled with E22

- `.gitignore`: un-ignore `kernels/runs/`. The `runs/` pattern hides the per-run manifests
  that `CLAUDE.md` says `kernels/` holds.
- Commit the RESULTS regeneration sequence used on 20 Sep as `scripts/regenerate_results.sh`,
  so every number in the report rebuilds with one command.

### 11.6 Leaderboard push (added 22 Sep)

Another team reached **0.79 (11/14)** on 21 Sep. The probes show v2's constants are the ceiling
for constants (9/14, Finding 45), so the only way up is per-problem prediction. **E27** fits
three estimators on the 33 official problems, labels pooled across configurations. The
estimators ship, not the labels; their training provenance is documented for reproducibility.

| ZIP (`dist/`) | Predictions | What its score tells us |
| --- | --- | --- |
| `solution-e27a-official-text-small.zip` | char tf-idf on official labels, every model | **0.57** (8/14): official-label text carries nothing |
| `solution-e27b-official-text-e11-small.zip` | the same, E11 fusion kept for DeepSeek | **0.57** |
| `solution-e27c-official-length-small.zip` | problem length only, every model | **0.57** |
| `solution-e27d-familiarity-small.zip` | each model's own NLL; the less familiar half robust | **0.29** (4/14): inverted, so its complement scores 10/14 (Finding 49) |

**Read on 22 Sep (Finding 49).** The three official-label models all scored 8/14, one below
v2, so that route is closed. E27d's 4/14 is below every constant, which can only happen if its
split is systematically inverted: **familiar problems are robust**, the direction MATH showed
all along (Findings 22, 47), not the olympiad reading E23 suggested. The overlap caveat is
retired: had the validation cases reused `train-main-v2` problems, `e27a` would have scored far
above 0.57.

### 11.7 Decomposing the familiarity split (23 Sep)

`e27e` turns `e27d`'s split the other way up. These six ZIPs then say where the gain comes
from, per model, never per case. **Scored 26 Sep** (Finding 58):

| Upload | Predicts | Score |
| --- | --- | --- |
| `solution-e27e-familiar-robust-small.zip` | familiar half robust, every model | **0.71** (10/14) |
| `solution-e27f-constants-small.zip` | Qwen and DeepSeek `True`, Skywork and Olmo `False` | 0.64 (9/14), confirming Finding 45 |
| `solution-e27g-split-qwen35-small.zip` | split for Qwen only, constants elsewhere | 0.64 (delta 0) |
| `solution-e27h-split-deepseek-small.zip` | split for DeepSeek only | **0.71** (delta +1) |
| `solution-e27i-split-skywork-small.zip` | split for Skywork only | 0.64 (delta 0) |
| `solution-e27j-split-olmo-think-small.zip` | split for Olmo-3-Think only | 0.64 (delta 0) |

The deltas sum to +1, matching `e27e` minus `e27f`, so the check passes and the Olmo cell that
Finding 50 left queued is filled. Per-model contributions are additive, so **10/14 is the
ceiling of this family**: no recombination reaches 11. The same split scores **0.353 on the 34
official rows** against 0.706 for always-robust, so G2's official-row leg would block the
leaderboard winner - a freeze decision (§11.10 in spirit, resolved at 27 Oct), not a modelling
one. `dist/solution-e27k-tilted-small.zip` (Skywork and Olmo call one fewer problem robust) was
submitted on 26 Sep and scored **12/14 = 0.86**, the best result so far and one case above the
previous leaderboard best. Searching one step past it costs a case in both directions tried
(`e27m` Qwen and `e27n` DeepSeek, 11/14 each), so the counts the probes imply are optimal
wherever they have been tested (Finding 61). Both tilts gained, so the familiarity ranking holds for every model
and only the counts were wrong (Finding 60). The count search stops there: the counts are
calibration to this validation set, and `e27e` is the prior-free fallback.

### 11.8 Rank-based familiarity successors (24 Sep)

E27's independent leaderboard result fixes the useful direction: within a model batch, more
familiar problems lean robust. E28 keeps the same batch-rank policy and asks whether a more
targeted one-pass statistic is stronger than whole-statement mean NLL.

| ZIP | Statistic | Pre-submission evidence |
| --- | --- | --- |
| `solution-e28a-suffix-nll-small.zip` | NLL on the second half, conditioned on the first | MATH AUC 0.675 and balanced accuracy 0.686, against mean NLL's 0.662 / 0.633 |
| `solution-e28d-min-k-gap-small.zip` | worst-token 10% NLL minus worst-token 20% NLL | official-row AUC 0.646; pooled problem AUC 0.867, template CI [0.728, 0.973], maxT-adjusted p = 0.001 across 12 candidates |

Both compute mean, suffix, and tail NLLs in the same forward pass and fall back to E11 v2's
per-model constants on any failure. Packaged checks on all four checkpoints returned native
booleans without fallback; the two-pass T4 checks took 45–147 seconds per model, leaving wide
headroom under the 3600-second evaluation limit. Checks ran on TASHIN and RSD only, not the
excluded `farhanishraqq` account.

**Submission read (24 Sep):** E28a is submission 942047 and scored **0.71**, exactly tying
E27e. E28d first failed in Codabench ingestion as 942257; an exact retry, submission 942361,
finished and scored **4/14 = 0.29** despite coverage 1.0 locally. Reject Min-K gap as a hidden-set
transfer signal. The other E28 variants remain local; no leaderboard grid search.

### 11.9 Cross-model stability and a DeepSeek-only successor (24 Sep)

`train/eval_e28_stability.py` stress-tests only the three pre-registered statistics. Min-K gap
keeps problem-level AUC above 0.73 when any one scoring model is removed; mean and suffix NLL
fall below 0.37. On evaluator-shaped official batches, Min-K gap is strongest for DeepSeek
(AUC 0.762, familiar-half accuracy 0.600). This agrees with the hidden-set decomposition:
DeepSeek's E27 split supplies the whole +1-case gain, while Qwen and Skywork each contribute 0
(Olmo remains queued).

`solution-e29a-deepseek-min-k-gap-small.zip` therefore applies Min-K gap only to DeepSeek and
uses the proven E11 constants elsewhere. It passes the official local scorer (coverage 1.0,
zero invalid predictions), but submission 942281 scores **8/14 = 0.57**, below E27e/E27h's
10/14 = 0.71. Together with all-model E28d's 4/14, **both controlled transfer tests fail:**
retain E27 and close the tail-gap branch. Do not tune the fraction or add weighted ensembles
on the 14 cases.

---

## 12. Handover note (29 September 2026)

Read this first, then `PROGRESS.md` for why each decision was taken. Earlier sections retain
historical plans; this section records the current decision.
**Small-track update, 29 Sep:** E37's GSM-Plus confirmation passed, but the official transfer
fails its frozen runtime gate. The repaired Qwen run on MUBAMAN and Skywork run on Arifin
completed; their combined runtime already exceeds the whole-run limit. DeepSeek on Nipun
hit its process time limit with partial output. Olmo on Tamim3 is currently inaccessible
through the API despite a verified account identity. No incomplete-data accuracy is reported.
All four old jobs with CUDA linker failures are confirmed cancelled. The driver-linker repair
and process guard are preserved along with the exact launch sources and runtime evidence.
Tamim1's latest token is verified as `fehazulislamsawmiq`,
distinct from Tamim2 (`duneak47`); its latest completed check returned no GPU or internet.
Tamim3 (`abrarrafi1552`) passed its version 2 capability retry with two T4 GPUs and internet.
MUBAMAN's replacement token maps to `mksadman`, now verified
with two T4 GPUs and internet. See `experiments.md` for execution status.
This update makes no claim about the Large-track fleet.

E37 is closed under its protocol; no further GPU retry or submission follows. E38's CPU
gate also failed. E27k remains the current leaderboard candidate. Reproduce the tables with
`python train/report_e37.py`, `python train/report_e37_runtime.py` and `python train/report_e38.py`.
The current task is to preserve and push the results; another prediction method requires a
new preregistered plan before execution.

### 12.1 What to submit, and how to rebuild it

| ZIP | Leaderboard | What it is |
| --- | --- | --- |
| **`dist/solution-e27k-tilted-small.zip`** | **0.86 (12/14)** | The policy that stands |
| `dist/solution-e27e-familiar-robust-small.zip` | 0.71 | The same ranking without per-model counts |
| `dist/solution-e11-v2-qwen-true-small.zip` | 0.64 | Zero-GPU fallback |

**The final solution goes in through the organizers' Google Form**; the leaderboard is validation
only (13 Oct, Finding 75). E50 solved the 14-case set: `e27k` is perfect on DeepSeek, Skywork and
Olmo, and its two misses are one Qwen pair. Do not build a Qwen swap to chase 14/14.

`python submission/make_e27.py` rebuilds every E27 ZIP deterministically. **The policy in one
sentence:** load the model being scored, take one forward pass over each bare problem, and call
the more familiar problems (lower mean token NLL) robust - `ceil(n/2)` of each batch, minus one
for Skywork and Olmo-3-Think, which is the count their class balance implies.

Everything else about the submission contract is in `COMPETITION.md`: native `bool`s,
`small.txt` at the root for the small track and **never** for the Main Track, offline loading,
3600 s for the whole run.

### 12.2 What is settled, so nobody re-runs it

| Question | Answer | Where |
| --- | --- | --- |
| Do per-model constants suffice? | No; they cap at 9/14 | Finding 45 |
| Does the shipped probe baseline work? | No, it is at chance; its headline is a selection artifact | Findings 2, 11 |
| Do MATH-trained models transfer to olympiad problems? | No: AUC 0.429 to 0.617, CIs spanning 0.5 | Findings 32, 39, 40 |
| Can we self-label with `answer_modulus`? | No - and on olympiad problems the yield is ~1 label per GPU-hour | Findings 31, 48 |
| Does test-time answer agreement predict robustness? | No, chance once length is controlled | Finding 24 |
| Do familiarity/contamination scores pass an offline gate? | No (G-M), though the leaderboard likes the MATH direction | Findings 47, 49 |
| Is there a cross-model feature signal? | No | Finding 32 |
| More official labels? | No: the split is final bar a couple of Skywork rows | Finding 59 |
| `problems-public`? | Private, will never be shared. E33 is dead | Finding 59 |

### 12.3 The one unresolved contradiction

The shipped policy's direction - **familiar problems are robust** - holds on DeepSeek's 142 MATH
instances and on the hidden validation set twice over. It **reverses** on the 34 official
`train-main-v2` rows, where the same split scores 0.353 against 0.706 for always-robust
(Finding 58). So **gate G2's official-row leg would block the policy that wins the leaderboard**,
and no more official labels are coming to break the tie.

The hypothesis worth testing, and worth writing up either way: `train-main-v2` holds *published*
problems (2025 AIME I appears verbatim), where familiarity means memorisation and predicts
spurious; the validation cases are mostly *not* those problems (Finding 49), where familiarity is
ordinary fluency and predicts robust. The organizers' own report splits results by published
versus unpublished, so the distinction is theirs, not ours.

**This no longer decides the freeze (30 Sep).** It was the open question while the choice was
`e27k` against `e27e`, and that choice is now settled on arithmetic instead: with the balances the
probes measure, `e27k` calls exactly the right *number* of robust cases for every model and `e27e`
provably does not, in all 56 consistent batch compositions (Finding 63). The re-probe on 30 Sep
reproduced all four balances exactly, twelve days after the first measurement, so the assumption
that argument rested on is discharged (Finding 64). **Ship `e27k`.**

The contradiction itself stands and is unresolved: it is about whether the familiarity *ranking*
means anything, which the null test leaves at p = 0.055. It is report material, not a submission
decision. The hypothesis worth writing up either way is the published-versus-unpublished one
above.

### 12.4 Experiments to improve the Small-track submission

| Priority | Experiment | Action and acceptance decision |
| --- | --- | --- |
| 1 | E37 official transfer | Closed: completed models already exceed the frozen whole-run runtime gate. Preserve partial outputs and runtime evidence; do not evaluate accuracy on incomplete coverage. |
| 2 | E38a exact-prefix audit | Completed on Tamim3 and failed: every stored shorter trace differed from the longest trace's prefix. See the generated RESULTS section. |
| 3 | E38b target-engine equivalence | Closed by the failed CPU gate; no GPU pilot. |
| 4 | Candidate packaging | No eligible new candidate. Keep E27k and preserve the negative results. |
| 5 (6 Oct) | **E49**: E37's frozen budget-path score, batched `transformers` extraction, one scoring of all 34 official rows | Pre-registered in `protocols/E49_PROTOCOL.md` (review: `REVIEW_GENERALIZATION.md`). **Closed 8 Oct: failed** (problem AUC 0.659 passed; half-rank balanced accuracy 0.521 against > 0.55). `e27k` stays (Finding 74). |

E27k remains the incumbent. No count/threshold search on the 14-case leaderboard. E37 and E38
are closed under their protocols. Record their negative results before proposing another
prediction method; do not reinterpret either failed gate.

No more compute is needed for these closed experiments. Arifin, Nipun, Tamim2, MUBAMAN and
Tamim3 previously passed GPU checks; Tamim3's current notebook access must be resolved before
future use. Tamim1 has no verified GPU access. Check quotas and identity before any new plan.
No extra paid compute is authorized.

Reproducibility: E37's table is rebuilt by `train/report_e37.py`. The older omnibus
`scripts/regenerate_results.py` does not yet rebuild every later section; do not claim it does.

### 12.5 Traps that cost us time

- **`AutoTokenizer` silently strips whitespace** for DeepSeek under transformers 5.13. Always
  load through `datagen/tokenizer_check.py`, which refuses a tokenizer that fails a round trip.
- **Kaggle `RUNNING` does not prove a healthy model process.** Inspect bounded live logs with
  `scripts/kaggle_live_logs.py`; worker failures can leave the parent alive. The retry bundle
  checks CUDA linking and uses `scripts/kaggle_process_guard.py` to stop failed or overlong jobs.
- **A Main-Track ZIP containing `small.txt` scores 0.00** - it cost four uploads (Finding 21).
- **Kaggle accounts are shared.** Historical Eshad and MUBAMAN tokens lacked GPU access;
  the current MUBAMAN token is `mksadman` and passed its capability test. SALAM and
  DIDHITI often have both slots taken by another project; RSD/Labib are the same accounts as
  MEDIUM/CHOTON elsewhere (Finding 42, `CLAUDE.md`).
- **Never mix the two label rules** for one pair: the older `absolute_accuracy_decay` rule and
  `train-main-v2`'s three-valued `max_drop` rule. Mixing them corrupted two labels (Finding 44).
- **Group by template, not by text**, whenever `train-main-v2` is used (Finding 34).

### 12.6 Next steps toward generalization (13 October)

**Why the direction changes.** Finding 76: on olympiad problems a model has solved, surface
rewrites almost never flip the label. The family that does is `expert_no_solution`, an added
premise that contradicts the rest. Our only large label source, DeepSeek's MATH tables, measures
domain-rewording brittleness instead, which is why nothing trained on it transferred. The next
labels must target the family that decides olympiad labels.

**Not worth doing, and why:**

- **A RAG pipeline.** Nothing at decision time is generated, so there is nothing to augment. Used
  as a nearest-neighbour lookup it is a classifier whose training set ships inside the ZIP, which
  the rules forbid ("never bundle datasets or labels"). Used as a contamination check it finds
  nothing on the test problems, which are new and half unpublished, and the model's own NLL
  already measures memorisation without an index.
- **More paraphrase-style datasets** from old math questions. Surface families drop 0-3% of
  eligible olympiad rows (Finding 76); labels built from them would teach the MATH mechanism again.

**E52, pilot: do our models notice an impossible premise, and can one forward pass predict it?**
Pre-register in `protocols/E52_PROTOCOL.md` before any output is read.

| Item | Choice |
| --- | --- |
| Problems | `BeyondHsueh/ReliableMath` at `c059d6e6`: 79 AMC 2023 originals, one `contradict` rewrite each, fixed by its id before sampling |
| Models | Qwen3.5-4B and Skywork-OR1-Math-7B first, the two fastest under vLLM (75 and 117 tok/s per engine, Findings 26, 37). DeepSeek and Olmo run at about 30 tok/s and join only if the pilot passes |
| Decoding | The organizers' settings (FAQ): temperature 1.0, top-k 40, top-p 0.95. `datagen/perturb/sample.py` needs a `--top-k` flag |
| Samples and cap | k = 2 on the original as the eligibility screen, k = 4 on the rewrite, 12k-token cap. The cap undercounts late abstentions, which reasoning models produce at length on ill-posed problems (MiP-Overthinking); record the cap-hit rate |
| Grading | A rewrite sample counts as correct when it abstains: no numeric final answer, or a no-solution statement in the final answer. Audit the grader by hand on 30 traces before reading any result |
| Cost | About 4M generated tokens per model: 8-10 session-hours at the measured rates, split over three accounts |

Three questions, each with a gate fixed now:

1. **Per-model detection rate** on eligible problems. This is the first independent evidence on
   the prior bet behind `e27k`'s tilts (Finding 67): the validation balances predict that Skywork
   detects contradictions less often than Qwen. Reported, not gated.
2. **Is detection a property of the problem?** Cohen's kappa between the two models on shared
   eligible problems. Reported.
3. **Does a one-pass signal on the original predict detection?** Mean NLL, length, and a
   prompt-state probe, as in the pre-generation probes of Lugoloobi et al. (arXiv 2602.09924).
   **Gate:** AUC with a bootstrap CI lower bound above 0.5 on at least 60 eligible pairs.

**E53, only if gate 3 passes:** fit that signal on all E52 pairs and ship it as a one-pass
submission with `e27k`'s ranking as the per-batch fallback. Then run the Docker test, read the
leaderboard once as a sanity check, and decide by 27 October which ZIP goes in the organizers'
Google Form. If the gate fails, `e27k` is the submission and Finding 76 plus E52 go in the report
as the mechanism and its negative result.

| Dates | Step |
| --- | --- |
| 14 - 15 Oct | `--top-k`, the abstention grader, ReliableMath pool, protocol; push the Qwen and Skywork pilots |
| 16 - 18 Oct | Grader audit, then the three questions; gate 3 decides E53 |
| 19 - 24 Oct | E53 build, GPU check, Docker test, one leaderboard read |
| 27 Oct | Freeze; the chosen ZIP goes in the Google Form before 1 Nov |

**E52 outcome (13 Oct): gate 3 failed with no positive class** (Finding 77). Within 16k tokens Skywork
never flagged a contradiction and answered through 79% of rewrite samples, often with the original
answer; Qwen ran out of budget on 79%. No one-pass signal can be trained from labels with one class,
so E53 is not planned and **`e27k` is the submission**. What E52 does add is mechanistic support for
the direction of the Skywork tilt (Finding 67): on the deciding family, Skywork is the model that
answers through the contradiction. The Olmo tilt has no such support: E52b could not sample Olmo at these settings on T4s and was
stopped on 14 Oct (Finding 78). **Next: the technical report**, with Findings 75-78 as its account of
what the labels measure and how the small models behave on the family that decides them.

### 12.7 Plan to the end (14 October): the leaderboard, then the prize

Two different targets, and they must not be confused. **The leaderboard** is 14 validation cases
that decide nothing; another team leads at 13/14. **The prizes** are decided on a new private set of
problems and variants, from the ZIP each team sends through the organizers' Google Form.

**A. The leaderboard: no fitting to the 14 cases (owner's rule, 14 Oct).** E50 solved the set
(Finding 75), and a per-case fix of Qwen's swapped pair would show 14/14. It was built and then
withdrawn: it fits five known cases and says nothing about new problems. **Only methods that
generalize are built from now on**; the board is read once per candidate, as a sanity check, never to
choose between variants.

- Suffix NLL with `e27k`'s counts is not worth an upload either: given `e28a`'s recorded 10/14 it can
  score at most 12 (E50 section of RESULTS).
- The route with a real chance of beating `e27k` on new problems is labelling the deciding family at
  the organizers' 100k budget on better compute, gated on reproducing the 34 official labels first
  (planned with the owner, 14 Oct).

**B. The prize: send `e27k` through the Google Form.** `dist/solution-e27k-tilted-small.zip`,
SHA-256 `b79addef…`: Docker-tested in the competition image (Findings 65-66), GPU-audited and
repeatable (Finding 62), one forward pass per problem.

- **Nothing measured beats it on unseen problems.** E49 failed its official-row gate (Finding 74), E52's
  gate had no positive class (Finding 77), and suffix NLL cannot exceed it (above). Every other
  candidate failed earlier (PAPER §6).
- **The one open bet stays as decided:** the Skywork and Olmo tilts (Finding 67). E52 gave Skywork's a
  mechanism; Olmo's rests on its balance alone (Finding 78).
- **Runtime, the tie-break, is already near its floor:** loading each model dominates, and the method
  generates no tokens.
- **Freeze 27 Oct**; send before 1 Nov, 12:55 UTC; keep the confirmation with the ZIP's SHA-256.

**C. The report:** fill in the author list, and decide whether to mention the public validation-label revision of
Finding 75 (after telling the organizers).
