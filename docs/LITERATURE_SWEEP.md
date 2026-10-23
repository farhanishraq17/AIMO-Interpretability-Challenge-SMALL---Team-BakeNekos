# Literature sweep (E61 checkpoint C0)

Five narrow research prompts, run **before** any E61 detector hypothesis is written. C0 is the entry
condition for checkpoint C2: the mechanism the detector targets is chosen by what comes back here, and
that choice is recorded as amendment 1 of `../protocols/E61_PROTOCOL.md` before any detector code
exists. See `../PLAN.md` §13.3 for where C0 sits in the plan.

**How to run them.** Paste one prompt verbatim into a research-capable assistant (Claude or Gemini
deep research). Run them in any order; they are independent. Paste the answer under the matching
**Answer** heading below, with the date and which assistant produced it. Do not summarise the answer
away — the raw output is the record, and §13.5's rule against relaxing gates applies to evidence too.

**Prompt structure.** Each follows the shape the published prompting guidance converges on: role,
precise task with scope, context already held, source preferences and exclusions, an explicit output
contract, and an instruction to flag absent evidence rather than assemble a case from adjacent
results. Keep that last clause if you edit a prompt — it is what makes a null answer usable.

**What a result means for C2.** A mechanism qualifies only with **at least two independent papers
reporting a usable effect size**. One paper, or a set of papers that only establish that a behaviour
exists without relating it to robustness, is not enough: that is the position E52 and E60 already
reached by measurement (Findings 77, 85). If no prompt clears that bar, C2 does not run and `e61a`
ships — which is a legitimate outcome, not a failure of the sweep.

| Prompt | Question | Status | Qualifies C2? |
| --- | --- | --- | --- |
| R1 | Does premise-ablation predict brittleness? | not run (**downgraded**: R5 found a competing entrant scored 0.38 on the official rows with it) | — |
| R2 | Cheap faithfulness tests inside a small token budget | not run (**next**: confirms the truncation length for the selected mechanism) | — |
| R3 | High-precision detection at tiny n | not run (**promoted**: one pooled threshold is now the binding question) | — |
| R4 | Memorization against robustness | not run | — |
| R5 | Prior art on the task itself | **done 23 Oct** | **yes — selected C2's mechanism** |

---

## R1 — does premise-ablation predict brittleness?

**Why it is asked.** This is the mechanism the official olympiad labels appear to turn on:
`expert_no_solution` decides 0.43 of spurious rows there, against ≤ 0.03 for every surface family
(Finding 76, RESULTS "E51"). E52 and E60 established that our models answer through contradictions;
neither established that this *predicts* the label.

### Prompt

> You are a research assistant surveying LLM mathematical-reasoning evaluation.
>
> Task: find empirical evidence on whether a model's answer stability under premise deletion or premise
> contradiction predicts that model's brittleness on answer-preserving perturbations of the same
> problem. Scope: 2023–2026, math word problems and olympiad-level math, open-weight models 4B–70B.
>
> Context I already have: ReliableMath (Xue et al. 2025) pairs AMC problems with contradictory-condition
> rewrites. I have measured that Skywork-OR1-Math-7B answers through such contradictions in 274 of 274
> samples, returning the original problem's answer 44% of the time. I need to know whether anyone has
> shown this behaviour correlates with measured robustness, not just that it occurs.
>
> Sources: prefer arXiv, ACL/EMNLP/NeurIPS/ICLR proceedings, and papers with released code or data.
> Exclude blog posts and leaderboard pages.
>
> Output: a table with columns — paper, venue and year, models tested, perturbation type, the
> correlation or AUC reported between premise-sensitivity and robustness, and sample size. Then three
> sentences on whether the effect is large enough to be useful at n of about 30. Then a list of anything
> that failed to find the effect.
>
> If the evidence is thin or absent, say so explicitly rather than assembling a plausible case from
> adjacent results.

### Answer

_Not run._

---

## R2 — cheap faithfulness tests inside a small token budget

**Why it is asked.** The budget is the binding constraint on every generative method we tried. The
organizers' own baseline spends 4,096 generated tokens per problem, so that cost is affordable by
their construction; our rejected methods wanted 28k (PAPER §10). This asks what fits in 4k.

### Prompt

> You are a research assistant surveying chain-of-thought faithfulness measurement.
>
> Task: identify test-time measurements that detect unfaithful or shortcut reasoning and cost under
> about 4,000 generated tokens per problem. Rank them by reported effect size per token spent.
>
> Context: I need a per-problem signal computable in one evaluation pass on a single 96 GB GPU, no
> network, no training labels. Candidates I know of: early-answering and corrupted-CoT (Lanham et al.
> 2023), biasing-feature tests (Turpin et al. 2023), GSM-Symbolic irrelevant-clause insertion (Mirzadeh
> et al. 2024). I do not know which are cheapest or which have the strongest reported effects.
>
> Sources: arXiv and major ML/NLP venues, 2023–2026. Prefer papers reporting per-example cost or token
> counts.
>
> Output: a ranked table — method, paper, tokens or forward passes per example, effect size reported,
> and whether it needs labels to calibrate. Then name the single cheapest method with a reported effect
> above chance on math reasoning.
>
> Flag any method whose cost is not reported rather than estimating it.

### Answer

_Not run._

---

## R3 — high-precision detection at tiny n

**Why it is asked.** E61 changes the target from a classifier to a flip-only detector, whose value is
`flips × (2·precision − 1)`. Gate B1 asks for precision ≥ 0.60 over ≥ 3 flips, and at that sample size
the interval is very wide. This asks whether any framework gives a usable guarantee at n = 34, or
whether B1 can only ever be a screen.

### Prompt

> You are a statistics-literate research assistant.
>
> Task: what methods reliably produce a high-precision minority-class detector when only about 30
> labelled examples exist, the minority class is about 30% of the data, and the metric is plain accuracy
> against a majority-class baseline?
>
> Context: I must decide whether to flip individual predictions away from a majority default. I need
> precision above 0.5 on the flipped subset, and I cannot afford to fit a threshold on the same 30 rows
> I evaluate on. I am looking for selective-prediction, conformal, or abstention frameworks with
> finite-sample guarantees at n of about 30, and for honest statements of where they break down.
>
> Sources: statistics and ML methodology literature; conformal prediction, selective classification,
> Learn-then-Test. Prefer sources that state small-sample behaviour explicitly.
>
> Output: three to five named methods, each with its assumptions, the minimum n where its guarantee is
> non-vacuous, and what it gives up. Then one recommendation for n of 34 with a 0.7 majority prior.
>
> If no method gives a usable guarantee at n of 34, say that plainly — that is a useful answer.

### Answer

_Not run._

---

## R4 — memorization against robustness

**Why it is asked.** The project's one unexplained result (PAPER §11.1): familiarity predicts robust on
MATH at AUC 0.662 and reverses to 0.312 on the official olympiad rows. The publication-status
explanation was tested at n = 8 and found no support (Finding 69). If the reversal is documented
elsewhere, it stops being our anomaly.

### Prompt

> You are a research assistant surveying data contamination in LLM math evaluation.
>
> Task: find evidence on whether memorization or contamination indicators (Min-K%, Min-K%++,
> perplexity-based membership inference, n-gram overlap, DE-COP, CDD) predict a model's robustness to
> answer-preserving perturbations — and critically, in which direction.
>
> Context: I have a documented sign reversal. On Hendrycks MATH, low perplexity ("familiar") predicts
> robust at AUC 0.662. On 34 official olympiad rows the same statistic reverses to AUC 0.312. My
> hypothesis was that published problems invert the direction through memorization, but testing it at
> n of 8 found no support. I want to know whether this reversal is documented elsewhere.
>
> Sources: arXiv 2023–2026, membership-inference and contamination literature. Include negative results
> and replication failures.
>
> Output: a table — paper, indicator, dataset, direction of the relationship with robustness or
> generalization, and sample size. Then a direct answer: is a population-dependent sign flip in this
> relationship documented in the literature? Yes, no, or unclear.
>
> Do not reconcile conflicting papers into a single story. Report the conflict.

### Answer

_Not run._

---

## R5 — prior art on the task itself

**Why it is asked.** Fifteen months of work in this repo assumed the task is novel. If the prediction
problem exists under another name, someone has already measured which signals work — and R5 is the
cheapest of the five to run.

### Prompt

> You are a research assistant doing a prior-art sweep.
>
> Task: find any published method, competition write-up, or paper that predicts whether a given model
> answers a given problem robustly, as a binary per-pair prediction, without re-running the full
> perturbation suite.
>
> Context: this is the AIMO Interpretability Challenge task (NeurIPS 2026, arXiv 2607.13899). I want to
> know whether this prediction problem has been posed before under another name — robustness
> forecasting, failure prediction, performance prediction, or predicting perturbation sensitivity — in
> math reasoning or elsewhere.
>
> Sources: arXiv, competition proceedings, NeurIPS/ICML competition tracks, and any public solution
> write-ups for this specific challenge.
>
> Output: a list of the closest prior formulations, what each predicts, the best accuracy reported, and
> the cheapest method that achieved it. Then one paragraph on whether any of it transfers to a 4-model,
> 30-label, 70-minute-budget setting.
>
> If this exact task appears novel, say so — that is itself the answer.

### Answer

# Prior-Art Sweep: Binary Per-(Model, Problem) Robustness Prediction Without Re-Running Perturbations

The AIMO Interpretability Challenge is real: arXiv 2607.13899 resolves to it, and it is an accepted NeurIPS 2026 competition. Its exact task is to predict, from the original problem alone, whether a model's correct answer survives a majority of perturbations, as a binary label per (model, problem) pair. I found no earlier math-reasoning paper that poses this task. One June 2026 clinical-VQA paper poses a structurally near-identical target, so "novel" holds only with qualifications. The challenge's own baselines and the public entries suggest the signal is weak (roughly 58–69% accuracy) and often amounts to difficulty or solve rate rather than robustness.

## TL;DR
- **The challenge is verified.** arXiv 2607.13899 ("AIMO Interpretability Challenge", Štefánik, Mondorf, et al., submitted 15 Jul 2026) is marked "Accepted Competition at NeurIPS 2026". It runs on Codabench (competition 16180), submissions close 1 Nov 2026, and the competition workshop is 12 Dec 2026 in Paris.
- **The exact formulation has no prior paper in math reasoning.** The closest structural match is Fazla, Testoni, Abu-Hanna, Plank and Calixto (arXiv 2606.16583, posted June 2026, v2 dated 3 Sep 2026, clinical VQA). It benchmarks "8 methods across 12 VLMs" and tests whether uncertainty on the unperturbed input predicts correct→wrong flips under one perturbation type, finding that it "reliably anticipates which predictions will collapse under NOTA." Other adjacent work predicts correctness rather than robustness: correctness probes, assessors, routing and semantic entropy. One further paper needs training dynamics rather than inference signals (Kang et al., arXiv 2411.07681).
- **For a 30-label, 4-model, tight-budget setting, prefer zero- or few-parameter signals.** Examples are answer-token log-probability, uncertainty-trace shape and a difficulty prior, with thresholds tuned on the 30 labels. High-dimensional per-model probes, assessors, IRT and routers need hundreds or more labels. Multi-sample consistency methods are probably too expensive for long olympiad reasoning traces.

## Key Findings

### 1. Verification verdict: arXiv 2607.13899 / AIMO Interpretability Challenge

**Verdict: REAL and verifiable.** I checked the arXiv abstract and HTML directly.

- **Paper.** "AIMO Interpretability Challenge" by Michal Štefánik, Philipp Mondorf, Andreas Waldis, Qianying Liu, Chuan Yang, Michal Spiegel, Josef Kuchař, Marek Kadlčík, Adam Vawda-Oomerjee, Chaoran Liu, Simon Frieder, Barbara Plank, Fazl Barez and Pontus Stenetorp. Listed as cs.AI, v1 dated 15 Jul 2026, with the comment "Accepted Competition at NeurIPS 2026".
- **Formal task.** The input is a triple (Problem X, Model M, Distribution shift Δ), where M(X) is already correct. The label is 0 if, for a majority of perturbed variants ΔX, M(X) is correct and M(ΔX) is wrong; otherwise it is 1.
  - The task is therefore conditional on initial correctness, and the target is binary at the per-pair level.
  - The proposal states: "the task is to decide if the model responds to the given problem robustly w.r.t. to a pre-defined set of distribution shifts."
  - Participants may use "weights, activations, token probabilities, or any other black-box statistics."
- **Data plan in the proposal.**
  - 180 problems from AIMO 2/3, JMO and the last two years of AIME, split into 150 train and 30 test.
  - At least 8 models, "resulting in roughly 1200 training and 240 test instances".
  - A fixed compute budget of "1 hour per problem on 8-GPU nodes".
  - Perturbations come from newly annotated symbolic templates modelled on GSM-Symbolic. A subagent read of Appendix B lists the types as rephrase, rename, domain, distract, typos, expert perturbations and expert no-solution.
  - There are two tracks: Main, and Small Models (under 10B parameters).
- **Organizers' own baselines.** These were run on DeepSeek-R1-0528-Qwen3-8B with stratified 10-fold cross-validation on their sample validation set.
  - Last-token linear probe: up to 58.37% (±7.5).
  - Uncertainty-based classifier built on Grünefeld et al. 2026: up to 69.23% (±10.13).
  - Random guessing: 50%.
- **Organizers' novelty claim.** The proposal calls the challenge "an entirely new competition" relative to prior ML-conference competitions. It distinguishes the challenge from the interpretability benchmarks RAVEL, InterpBench and MIB. It also flags as an untested assumption that black-box behaviour on the original problem will "in most cases likely not be sufficient".
- **Logistics.**
  - The website describes the system as one that "decides, for a single problem–model pair, whether that model answers that problem robustly". It lists the Aug 15 – Nov 1, 2026 competition window and the Dec 12, 2026 workshop in Paris.
  - The organizers' Bluesky account reports 120 participating teams in the main phase.
  - Prize figures conflict. The website and Bluesky say $12,500 / $12.5K, while the MLContests listing shows $10,000. I treat the organizer figure as authoritative.
- **Relationship to AIMO Prize and Kaggle.** The website states the challenge "is a separate competition" that gives access to AIMO 3 model submissions. I found no interpretability track inside the Kaggle AIMO Progress Prize itself. AIMO 3 is cited only as the source of models and problems ("more than 4,000 teams"). I did not independently fetch aimoprize.com or the XTX Markets announcements.
- **Discrepancy with your stated setting.** Your 4-model / 30-label / 70-minute setting does not match the proposal's design numbers (at least 8 models, ~1,200 training instances, 1 h per problem). Public entrant repositories describe different working conditions:
  - byeolki: "Official 34 rows" of labels; a Codabench validation set of 14 cases (small track); 3,600 s for the whole prediction run on one RTX PRO 6000.
  - EconLearn: 21 main-track and 14 small-track validation cases, and a possible "170 seconds per sample" limit in the final.
  - Treat your parameters as the operative constraints; I could not verify them from the primary source.

### 2. Public solution write-ups for this challenge (Step 3)

All three public entries are pre-deadline, self-reported and scored on very small validation sets, so treat them as anecdotes rather than benchmarks.

- **byeolki/aimo-interp (Small Models track).** Report: "Prompt Probes Read Solve Rate, Not Robustness: Negative Results from the AIMO Interpretability Challenge".
  - Shared prompt probe: 0.643 on Codabench validation (14 cases), and cross-validation up to 0.78 on the official 34 rows.
  - The always-true baseline already scores 0.68 on those 34 rows, because the labels are imbalanced.
  - Self-probe blend: 0.42–0.49 within-model.
  - Test-time counterfactual check (self-generated perturbations): 0.50 on validation and 0.38 on the official rows.
  - Total rented compute was about USD 10.
- **EconLearn/aimo-interp-difficulty-vs-fragility.** Report: "Difficulty is not fragility: an audited robustness predictor and 70 negative results".
  - Main track: 17/21 = 0.81 (2,634 s).
  - Small track: 9/14 = 0.64 (1,913 s).
  - Earlier versions scored 16/21 (0.76) and 10/14 (0.71).
  - With n=21, a 95% binomial interval around 0.81 spans roughly 0.58–0.95, so the gaps between these versions are noise-level.
- **Other repositories** (dawoodwasif starter, BakeNekos) contain no reported results.
- **Shared lesson from the substantive entries.** Signals readable from the prompt mostly track difficulty or solve rate, not fragility. In-run self-perturbation did not help. This fits Lugoloobi et al.'s E2H-AMC result: pre-generation probes decode human difficulty at Spearman ρ=0.83–0.87 but model-specific difficulty at only 0.40–0.64. In their words, models encode "a model-specific notion of difficulty that is distinct from human difficulty."

### 3. Closest prior formulations, ordered from closest to furthest

| # | Work | Predicts (granularity) | Needs perturbations at inference? | Best reported metric | Cheapest near-best method |
|---|---|---|---|---|---|
| 1 | AIMO Interp. Challenge baselines (arXiv 2607.13899, 2026) | Binary robust/spurious per (model, problem), given correct | No | 69.23% acc (±10.13), uncertainty classifier, DeepSeek-R1-0528-Qwen3-8B | Last-token linear probe, 58.37% |
| 2 | Fazla et al., "Uncertainty Is Not a Safety Net for Clinical VQA, but Can It Anticipate Model Failure?" (arXiv 2606.16583, 2026) | Binary correct→wrong flip under NOTA perturbation, per (model, item), from unperturbed-input uncertainty | No (original input only) | Flipped items' baseline self-consistency uncertainty 4.41× (semantic entropy 3.90×) the average, Qwen2.5-VL-72B; AUROC tables in appendix | Label NLL (single pass) leads correctness discrimination on 11/12 models |
| 3 | Kang et al., "What Do Learning Dynamics Reveal About Generalization in LLM Reasoning?" (arXiv 2411.07681) | Per-example robustness to perturbation (GSM8K, MATH) via "pre-memorization train accuracy" | No, but needs fine-tuning trajectories | Binned degradation; no AUROC found | Not applicable to frozen third-party models |
| 4 | Han et al., "Fragile Reasoning" (arXiv 2604.01639, 2026) | Answer flips under name/number paraphrase, GSM8K (677 pairs) | Yes (compares original and perturbed runs) | Cascading Amplification Index (CAI) AUC up to 0.679 | Logit-lens first-divergence layer |
| 5 | Grünefeld et al., "Tracing Uncertainty in Language Model 'Reasoning'" (arXiv 2605.07776, 2026) | Binary CoT correctness from uncertainty-trace profile | No | AUROC 0.807 (5 LMs, GSM8K/ProntoQA) | First few hundred tokens: AUROC 0.801 |
| 6 | Lugoloobi et al., "LLMs Encode Their Failures" (arXiv 2602.09924, accepted at COLM 2026) | Binary success per (model policy, question), math/code | No (pre-generation activations) | AUROC >0.7 for "several models"; probe AUROC falls 0.78→0.64 as reasoning budget grows, even while accuracy rises 86.6%→92.0% | One prompt forward pass + linear probe |
| 7 | Cencerrado et al., "No Answer Needed" (arXiv 2509.10625; ICLR 2026 workshop) | Binary correctness from question-only activations | No | Strong on trivia; Lugoloobi et al. describe "substantially weaker results on mathematical reasoning (GSM8K AUROC ≈ 0.6–0.7)" | Difference-of-means direction |
| 8 | Kadavath et al., "Language Models (Mostly) Know What They Know" (arXiv 2207.05221, Anthropic 2022) | P(IK) per question, no proposed answer | No | AUROC rises with model size; OOD calibration weak | Fine-tuned value head |
| 9 | Assessors: Hernández-Orallo et al. 2022; Pacchiardi, Cheke & Hernández-Orallo, "100 instances is all you need" (arXiv 2409.03563, Proc. KDD 2024 Workshop on GenAI Evaluation); PredictaBoard (ACL Findings 2025); ADeLe "General scales" | Per (LLM, instance) success | No | ADeLe demand-based assessor AUROC 0.88 (GPT-4o), OOD weighted AUROC 0.81 | Random-forest on annotated demands; ~100 reference instances per new LLM |
| 10 | Routing: RouteLLM (Ong et al., arXiv 2406.18665) | Win probability per (query, strong/weak model) | No | Matrix factorization reaches 95% of GPT-4 quality with 26% GPT-4 calls (14% with augmentation) | Matrix factorization router |
| 11 | Semantic entropy (Farquhar et al., Nature 630, 2024) | Per-question confabulation/error | No, but needs k samples | Mean AUROC 0.790 vs naive entropy 0.691, P(True) 0.698 | Discrete semantic entropy over sampled answers |
| 12 | Math perturbation suites: GSM-Symbolic/NoOp (ICLR 2025), MATH-Perturb (ICML 2025), Functional MATH() (arXiv 2402.19450) | Measure breakage; do not predict it | Yes (that is the point) | NoOp drops up to 65%; MATH-P-Hard o1-mini −16.49%; reasoning gaps 58.35–80.31% | n/a |

**What each row means for novelty:**

- **Row 2 (Fazla et al.) is the only published target I found that is structurally identical: predict, from the unperturbed input, whether a currently correct per-(model, item) answer flips under perturbation.**
  - Its hypothesis H2 asks whether "uncertainty on the original input predict[s] whether a correct prediction will remain stable or flip under perturbation". The authors frame a positive result as "a novel use of UE as a response-fragility signal".
  - Flipped samples had higher baseline uncertainty in every model/method combination under +nota, and in 85 of 96 combinations under -correct+nota.
  - The differences from the AIMO task: one perturbation type rather than a majority-of-suite label, multiple-choice clinical VQA rather than open-answer olympiad math, and no interpretability framing.
  - It was posted on 15 Jun 2026, one month before the challenge paper. One author, Barbara Plank, is also an AIMO challenge organizer.
- **Rows 3–4 are math-specific but not equivalent.**
  - Kang et al. (Llama3 8B and Gemma2 9B on GSM8K and MATH) report that pre-memorization train accuracy "is also indicative of whether individual model predictions are robust to perturbations in the training query." That signal comes from fine-tuning dynamics on training queries, not held-out test problems, and is unavailable for frozen AIMO models.
  - Han et al. need the perturbed forward pass; they explain flips rather than predict them.
- **Rows 5–11 predict correctness, not robustness conditional on correctness.**
  - In the AIMO task every instance is already correct, so a pure correctness predictor can only work through a correlation between fragility and residual uncertainty or difficulty.
  - byeolki's "solve rate, not robustness" finding and EconLearn's "difficulty is not fragility" finding suggest that correlation exists but is weak.
- **Adjacent but further away.** Related ideas that are not the same problem:
  - CCPS (EMNLP 2025) perturbs hidden states, not inputs, to estimate confidence.
  - Huang et al. (ICML 2025, arXiv 2505.11770) use internal causal mechanisms to predict out-of-distribution correctness on non-math tasks; the AIMO proposal cites this line as aligned.
  - "Adaptive Distraction" (arXiv 2502.01609) trains a classifier on problem text to flag items that are easy to perturb. It is largely model-agnostic and used to filter attack targets.

### 4. Novelty statement

- **Within mathematical reasoning, the exact task appears unposed before the AIMO Interpretability Challenge.** I mean binary per-(model, problem) prediction of robustness under a perturbation suite, conditional on initial correctness, using only the original problem. The challenge itself is the first statement of it I found.
- **The formulation is not unprecedented in general.** Fazla et al. (June 2026) pose the same conditional flip-prediction target in clinical VQA under a single perturbation.
- **Older literature studies the inputs to the task, not the task itself.** It covers correctness prediction (P(IK), probes, assessors, routing, semantic entropy) and breakage measurement (GSM-Symbolic, MATH-Perturb, Functional MATH).
- **How I would phrase the claim:** "first benchmark/competition for per-pair robustness prediction in olympiad math, with labels from symbolic-template perturbation suites; the general idea of anticipating perturbation-induced flips from unperturbed-input uncertainty has been explored in clinical VQA (Fazla et al., 2026)." A blanket "never posed before" would be wrong.

## Details: transfer to a 4-model, 30-label, 70-minute setting

**Methods that survive.** With 30 labels across four models (about 7–8 per model), any method whose decision boundary has more than a handful of free parameters will overfit.
- High-dimensional supervised probes are the clearest failure. The organizers' probe reached only 58% with roughly ten times more data, and byeolki's per-model self-probes fell to 0.42–0.49.
- Assessor, IRT and router families also fail here. Pacchiardi et al.'s generic assessor saturates around 100 reference instances per new LLM, and RouteLLM-style routers need large preference datasets. They are only viable if you can pool public correctness data, and even then they predict correctness rather than robustness.
- Zero-training scalar signals survive: answer-token log-probability (Label NLL, the best single-pass discriminator in Fazla et al.), uncertainty-trace slope and linearity (Grünefeld et al.), reasoning-trace length, and a difficulty prior. Each needs only one or two thresholds, which 30 labels can tune.
- Per-model calibration is the sharpest constraint. Absolute thresholds differ by model (Fazla et al. show UE quality and calibration vary by model and family), and 7–8 labels per model cannot set them reliably. Use rank- or z-normalization within each model on unlabeled problems, then a single pooled threshold.
- A difference-of-means direction (Cencerrado-style) is the only activation method plausible at this n, and should be treated as one feature, not a probe.

**Compute.** The budget matters more than the method.
- If 70 minutes covers roughly 30 problems × 4 models (~120 pairs), that is about 35 s per pair. That is enough for one prompt forward pass, or one truncated reasoning trace, but not for self-consistency or semantic entropy with k long olympiad traces.
- Pre-generation probes and log-probabilities from a single (possibly truncated) trace fit the budget. Grünefeld et al. report AUROC 0.801 using only the first few hundred tokens, which suits truncation.
- Re-running self-generated perturbations is both expensive and, in byeolki's test, uninformative (0.50 on validation).

## Recommendations
1. **State novelty precisely.** Claim novelty for olympiad-math, suite-majority, interpretability-based robustness prediction. Cite Fazla et al. (2606.16583) as the closest prior formulation and Kang et al. (2411.07681) as the closest math-specific per-example robustness predictor.
2. **Build on cheap signals first.** Start from answer-token log-probability plus uncertainty-trace features from a truncated single trace, normalized per model, with one pooled threshold. Report balanced accuracy, because always-true already scores 0.68 on the public rows.
3. **Control for the difficulty confound.** Two independent entrants found that their signals track solve rate rather than fragility. Show your predictor beats a difficulty-only baseline at matched solve rate; that is the experiment a reviewer will ask for.
4. **Report uncertainty honestly.** With 14–30 evaluation cases, differences under about 15 points are not distinguishable. Use the bootstrap and McNemar tests the organizers propose.

## Caveats
- The Grünefeld et al. figures are now checked against the arXiv 2605.07776 abstract (Grünefeld, Højer, Mondorf, Plank et al., 8 May 2026): "AUROC up to 0.807" across "five LMs evaluated on GSM8K and ProntoQA", and "AUROC 0.801 using only the first few hundred tokens". Two of its authors, Mondorf and Plank, are also AIMO organizers. The Kang et al. characterization is checked against the arXiv 2411.07681 abstract.
- I did not retrieve the AUROC values in the Fazla et al. appendix. The 4.41× / 3.90× ratios and the 85/96 count are from the main text.
- The public challenge entries are self-reported, pre-deadline, and evaluated on 14–34 cases.
- I did not independently verify aimoprize.com, the Kaggle pages or the XTX Markets announcements. The link to AIMO 3 comes from the challenge's own materials.
- A very recent 2026 preprint on GSM-Symbolic flip prediction could exist that my searches missed. The organizers' novelty claim and an independent automated review (Pith) both say they found no such task, but neither is an exhaustive search.
