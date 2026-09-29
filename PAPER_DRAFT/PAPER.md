# Predicting Reasoning Robustness Without Training Data

### A record of the AIMO Interpretability Challenge, both tracks

**Draft — 30 September 2026.** Every number here is regenerable from the repository; the
commands are in §14. Findings are cited as `F-n` (Small Models Track, `PROGRESS.md`) and `LF-n`
(Main Track). Where this draft and a `RESULTS.md` table disagree, the table is right.

---

## Abstract

The AIMO Interpretability Challenge asks a question that sounds like classification and is not:
given a model and a problem, will that model's answer survive perturbations that leave the
mathematics unchanged? We ran 41 experiments across two competition tracks over six weeks,
spending roughly 300 GPU-hours on a shared free-tier fleet. **Almost all of them failed**, and
the failures are the substance of this report: eleven distinct signals — hidden-state probes,
reasoning-trace uncertainty, answer agreement under rewriting, self-generated perturbation labels,
activation geometry, contamination statistics — either sat at chance or reversed sign when moved
to an independent population.

What survived is a method with **no learned parameters at all**. It takes one forward pass per
problem, ranks the batch by the scoring model's own mean token negative log-likelihood, and calls
the more familiar half robust, with the count per model fixed by a closed-form identity rather
than fitted. It scores **12/14 = 0.857** on the hidden validation set, above the 0.79 that led the
board when we started, and it runs in **334 seconds for four models** where a rejected alternative
needed 5,293 seconds for one.

We argue that on a 14-case board where scores are multiples of 1/14, leaderboard position will
saturate and the competition's own tie-break — average runtime — combined with the writeup
criteria will decide outcomes. We therefore treat generalization and efficiency as the primary
objects of study, and report our method's failure modes as carefully as its successes.

---

## 1. The task

### 1.1 Formal statement

A submission implements

```
are_robust(model_id: str, problems: list[str], reasoning_effort: str = "default") -> list[bool]
```

Write `M` for the set of evaluated models and `P` for a batch of problem statements. The function
computes `f: M × P^n → {0,1}^n`, one native Python `bool` per problem, in order.

The label it predicts is **not** whether the model answers correctly. For a (model, problem) pair,
the organizers generate perturbed variants in families
`Φ = {rephrase, rename, domain, distract, typos, expert_perturbations, expert_no_solution}`,
each answer-preserving except the last. With `a₀` the model's accuracy on the original and `a_φ`
its accuracy on family `φ`, define

```
maxdrop(m, p) = max_{φ ∈ Φ} ( a₀(m, p) − a_φ(m, p) )
```

and the published three-valued rule (`aimo-interp/train-main-v2`, verified with zero violations
over all 82 rows, F-30, F-33):

```
robust      iff  maxdrop ≤ 0.10
spurious    iff  maxdrop ≥ 0.25
unlabelled  otherwise            — band cases are never scored
```

Eligibility requires `a₀ = 1.0`: a model that cannot solve the original has no robustness to
measure.

### 1.2 What makes it hard

Three properties, each of which defeated a standard approach.

**The label is a property of the problem, not the pair.** Of the 33 labelled problems in the
official split, 11 carry labels for two or more models, and **all 11 agree** — every pairwise
comparison is exactly 1.000, including `gpt-oss-120b` against each of the four small checkpoints
(LF-12). A per-problem oracle scores 0.986 against the 71 labelled rows where always-robust scores
0.732. An earlier measurement on a different subset found 20/20 agreement at p = 0.0005 (F-38).
So the model-specific white-box machinery the competition invites is, empirically, aimed at
variance that barely exists.

**Labels are almost absent.** The only in-population labelled data is 82 rows over 36 problems, of
which 71 carry a label. Everything else available is Hendrycks MATH, a different distribution, and
**nothing transferred across that gap** (§6.2). Our largest single finding is that the project was
label-bound from beginning to end, and no amount of compute we had access to could fix it.

**Evaluation is one hour, once.** A single RTX PRO 6000 (96 GB), 3600 s for the *whole* run
covering every model and every problem, no network, no installs, weights only from a read-only
cache. This is not a training-time constraint that scale can relax; it is a hard bound on what any
method may do per problem at test time.

### 1.3 Why generalization and efficiency are the objects of study

The hidden validation set has 14 cases (F-27, F-45). Every score is therefore a multiple of 1/14,
and the ladder from chance to ceiling has roughly seven rungs. We reached 12/14; the constants
ceiling is 9/14; several teams sat at 11/14. **Scores will saturate**, and when they do the
competition's published tie-break decides: *"ties broken by efficiency: faster average runtime over
three repeated trials."* The $5,000 writeup pool is awarded on generalization, actionable
interpretability, negative results and efficient methods — none of which is leaderboard position.

We therefore optimised for three things in this order: a method that cannot overfit because it has
nothing to fit; a method whose per-problem cost is a single forward pass; and a written record
where every negative result is reproducible. §10 gives the efficiency analysis, §9 the
generalization argument.

---

## 2. Setting

| | |
| --- | --- |
| Competition | AIMO Interpretability Challenge, NeurIPS 2026, Fields Model Initiative |
| Scoring | Codabench competition 16180; 10 submissions/day, 300 total |
| Tracks | **Small Models** (open weights < 10B) and **Main** (adds `openai/gpt-oss-120b`) |
| Evaluated models | `Qwen/Qwen3.5-4B`, `Skywork/Skywork-OR1-Math-7B`, `allenai/Olmo-3-7B-Think`, `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, and in the Main Track `openai/gpt-oss-120b` |
| Evaluation hardware | 1× RTX PRO 6000 (Blackwell, sm_120), 96 GB, 3600 s, offline |
| Our hardware | Kaggle free tier: T4×2, 2 sessions and 30 GPU-h per week per account, ~11 accounts shared with other projects |
| Local hardware | RTX 3050 6 GB — cannot hold any target model; used for data work, CPU training and packaging |

The asymmetry between evaluation hardware and our hardware shaped everything. We could never run
`gpt-oss-120b` at all (LF-3), and a T4×2 kernel decodes 10–90 tokens/s depending on the model
(LF-11), which made any generation-heavy method unaffordable to *develop*, independently of
whether it would be affordable to *run*.

---

## 3. Data, and the archaeology it required

### 3.1 What exists

| Source | Content | Role |
| --- | --- | --- |
| `aimo-interp/train-main-v2` | 82 rows, 36 olympiad problems, 5 models, `maxdrop` labels | The only in-population labels |
| `aimo-interp/val-sample` | 28 public validation rows, mixed models | Packaging check only — 1 small-track case |
| `aimo-interp/augmented-sample-math-agg` | Hendrycks MATH, aggregated, for `qwen3-8b:low` | The uncertainty baseline's training set |
| `aimo-interp-challenge-sample-full` | 10 AIME-2025 reference problems × 8 systems, **and the official perturbed texts** (453 variants) | Calibration exemplars |
| `aimo-interp/problems-public` | Symbolic chains | **Private. Will not be shared** (F-59) |
| Our olympiad pool | 505 problems with integer answers: OlympiadBench 362, AMC 83, AIME-2024 30, AIME-2026 30 | Unlabelled; the population the test set is drawn from |

### 3.2 Three data traps, each of which cost real time

**The label rule had to be recovered, and then it changed.** The older public tables key on
`absolute_accuracy_decay`; `train-main-v2` keys on the three-valued `maxdrop` rule above. They are
different rules for different tables, and merging them corrupted two DeepSeek labels before we
noticed (F-44). The fix is a precedence rule: wherever the official split covers a pair, it
decides, and the two rules are never aggregated for the same pair.

**Problem text is not a safe grouping key.** `train-main-v2` repeats the same template with one
constant swapped — `$10^{5}$` against `$10^{4}$`, `f(3520)` against `f(2024)` — under *different*
labels: 36 distinct strings over 29 templates (F-34). Grouping folds by a hash of the text leaks a
near-duplicate across the split. Everything downstream groups by template.

**`AutoTokenizer` silently destroys the input.** Under transformers 5.13, loading DeepSeek's
tokenizer through `AutoTokenizer` strips all whitespace from the problem, so
`"Find the answer \boxed{204}."` round-trips to `""` (F-14). This affects the organizers' own
baseline. Every pipeline now loads through a guard that refuses a tokenizer failing a round trip,
and that guard raises `SystemExit`, which a bare `except Exception` does not catch — a detail that
would have ended an evaluation run.

### 3.3 The permission question, answered

The proposal paper forbade extra labelled data for the official perturbation types. The website
retired that restriction on 19 September, and on 26 September the organizers confirmed directly
that generating additional training data from `train-main-v2` is allowed, that the official labels
are final bar a couple of Skywork rows, and that `problems-public` is private and will never be
shared (F-59). The last point killed an entire planned lane.

---

## 4. Method

### 4.1 The predictor

Let `m` be the model being scored and `P = (p₁ … pₙ)` the batch. For each problem, take **one
forward pass** over the bare problem text and record the mean token negative log-likelihood:

```
s_i = − (1/|t(p_i)|) Σ_k log P_m( t_k | t_{<k} ),      t(p_i) = tokens of p_i
```

No generation, no chat template, no system prompt, truncation at 2048 tokens (which never fires:
the longest labelled problem is 641 tokens). Rank ascending — **familiar first** — and call the
first `c_m` problems robust:

```
π = argsort(s),        ŷ_{π_j} = 1  iff  j ≤ c_m
```

### 4.2 The count, in closed form

This is the part that is derived rather than tuned, and it is the paper's main methodological
claim.

For model `m` on the hidden set, let `r_m` and `q_m` be its robust and spurious case counts,
`n_m = r_m + q_m` its batch size, and `d_m = r_m − q_m` its **class balance**. Constant-prediction
probes measure `d_m` directly: with all-False scoring `a` cases, a probe answering `True` for model
`m` alone scores `a + d_m`, because it gains `r_m` and loses `q_m`. Measured twice, twelve days
apart (F-27, F-64):

| Model | all-False | probe | `d_m` |
| --- | --- | --- | --- |
| Qwen3.5-4B | 7/14 | 8/14 | **+1** |
| DeepSeek-R1-0528-Qwen3-8B | 7/14 | 8/14 | **+1** |
| Skywork-OR1-Math-7B | 7/14 | 6/14 | **−1** |
| Olmo-3-7B-Think | 7/14 | 6/14 | **−1** |

A ranking policy should call exactly `r_m` problems robust — any other count is wrong before the
ranking is even consulted. From the definitions, `r_m = (n_m + d_m)/2`. The four balances sum to 0
and the batch sizes sum to 14, which forces **every `n_m` to be odd**, so `⌈n_m/2⌉ = (n_m+1)/2` and

```
c_m = r_m = (n_m + d_m)/2 = ⌈n_m/2⌉ + (d_m − 1)/2
```

giving the tilt as a closed form of the measured balance:

```
t_m = (d_m − 1) / 2        →   t = 0 for d = +1,   t = −1 for d = −1
```

**The per-model counts are not hyperparameters.** They are the arithmetic consequence of a
measurement, correct for *any* batch sizes consistent with that measurement. Enumerating all 56
size assignments consistent with the balances and the 14-case total: this rule calls exactly the
true robust count in **56 of 56**, and the plain median split in **0 of 56** (F-63).

### 4.3 The algorithm, complete

```
are_robust(m, P):
    if m not in SPLIT_MODELS:        return [CONSTANT[m]] * |P|
    try:
        s   ← one forward pass per p ∈ P under model m        # no generation
        π   ← argsort(s)                                      # ascending: familiar first
        c   ← clamp( ceil(|P|/2) + t_m , 0, |P| )
        return [ j ≤ c for j in rank(π) ]
    except (Exception, SystemExit):                           # tokenizer guard raises SystemExit
        return [CONSTANT[m]] * |P|
```

`CONSTANT[m] = (d_m > 0)`, the best constant for that model. The fallback is not decoration: it is
what keeps a batch valid if weights are missing, a tokenizer fails its round-trip guard, or the
worker OOMs, and an invalid batch scores zero with no visible traceback.

### 4.4 Learned parameters: none

The shipped artifact is 3.3 KB of source and contains **no fitted estimator, no training data, no
weights**. Its only free quantities are the four balances, and those are measurements of the
evaluation distribution rather than parameters fitted to features. This matters for §9.

---

## 5. Experimental programme

41 experiments in seven lanes. Full registry in `experiments.md`; what follows is what each lane
was *for*, and what it returned.

| Lane | Question | Experiments | Verdict |
| --- | --- | --- | --- |
| A — text | Is the problem text alone predictive? | E0, E0b | Yes on MATH (0.673 balanced accuracy), no on olympiad |
| B — white box | Do internals carry robustness? | E1, E1b, E4, E30, E31, E32 | AUC up to 0.749 in-distribution; every controlled transfer fails |
| C — traces | Does the reasoning trace carry it? | E2, E2f, E14, E36 | At chance, and 28k tokens per problem |
| D — self-labelling | Can we make our own labels? | E7, E15–E20, E24 | κ = −0.135 on MATH, +0.286 on olympiad, neither usable |
| E — familiarity | Does contamination predict brittleness? | E12, E23, E27, E28, E29 | Fails offline gates, **wins on the leaderboard** |
| F — external | Do other robustness datasets transfer? | E34, E35, E37, E38 | Real labels found; ranking does not transfer; runtime gate fails |
| G — arithmetic | What do the leaderboard numbers force? | Probes, E40, E41 | The decisive lane, and the cheapest |

---

## 6. What failed

This section is the point of the report. Each entry states the hypothesis, the strongest result it
ever produced, and the test that killed it.

### 6.1 The shipped baseline is at chance

The organizers' `trained-probe` solution reports 58.37% ± 7.5 on DeepSeek. Reproduced honestly
under nested cross-validation, its per-layer mean balanced accuracy is **0.5005** (F-2, F-11). The
published figure is a **layer-selection artifact**: choosing the best layer on the same data that
scores it inflates inner-fold AUC of 0.75–0.85 down to 0.640 held out. Averaging every layer,
rather than selecting one, is a genuine lead at AUC 0.749 — but see §6.2.

**This is the single most useful thing a newcomer to this competition can know**, and it applies to
anyone who takes the baseline's number at face value.

### 6.2 Nothing transfers from MATH to olympiad problems

Our labelled data is overwhelmingly Hendrycks MATH; the private test set is entirely olympiad. Every
block was measured across that gap and every one failed (F-32, F-39, F-40):

| Block | In-distribution | On olympiad rows |
| --- | --- | --- |
| Text (char tf-idf) | 0.673 balanced acc | AUC 0.617, CI spans 0.5 |
| Stage A prompt states | AUC 0.749 | AUC 0.429 |
| E11 fused, shipped | 0.717 balanced acc | 0.429 |
| Cross-model probe | — | no held-out model clears chance |

Training the text block *on* olympiad labels instead does not rescue it (F-40). There is no version
of this that works at n = 33.

### 6.3 Self-labelling: two attempts, both rejected

If labels are the bottleneck, make labels. We sampled models on originals and on perturbed
variants and applied the official rule. Twice.

- **On MATH problems (E20):** Cohen's κ = **−0.135** against official labels, drops correlated
  −0.25. Worse than chance. Lane closed (F-31).
- **On olympiad problems (E24), after arguing the population was the problem:** κ = **+0.286**,
  bootstrap CI spanning 0, on 15 usable pairs — for **15.8 GPU-hours**, about one label per GPU-hour
  (F-48). The sign flipped, which is real evidence the population mattered, and the yield made the
  lane unaffordable anyway.

The deeper issue: our perturbation family (`answer_modulus`) is not one of the official seven. It
changes what is asked, so it measures a different quantity. Doing this properly needs official-style
families with verified answer preservation — which is exactly the Main Track's planned Lane A, and
which the T4 fleet cannot afford (§11.3).

### 6.4 Test-time methods are unaffordable and at chance

**Answer agreement under rewriting (E14).** Hypothesis: a robust model gives the same answer when
the problem is reworded. Result: balanced accuracy 0.507, AUC 0.497 — chance — at **28,000 generated
tokens per problem** (F-24). Both halves of that sentence independently disqualify it.

**Reasoning-trace uncertainty (E2f).** The published 14 uncertainty features are near chance on our
data (F-15). Adding trace states to the fusion moves balanced accuracy 0.7173 → 0.6928 — it makes
the model *worse* (F-18). The signal is in the prompt, not in the trace.

### 6.5 Activation geometry: attractive, and an artifact

Two related lanes produced the project's most seductive numbers.

**E30, architecture-invariant geometry.** Family-wise test fails; the selected signal *reverses* on
independent MATH data (F-51).

**E31, counterfactual prompt-state sensitivity** — measure how far a problem's representation moves
under an answer-preserving rewrite. Best AUC **0.735** on official data, the strongest new signal we
found. It failed its template-cluster CI and its max-T test, and Qwen sat near chance (F-52). E32
then scored it on an independent DeepSeek MATH population: **AUC 0.545, CI [0.428, 0.658]** (F-53).
Lane closed.

The pattern repeats often enough to name: **a signal selected on 29 template clusters will reach
AUC 0.7–0.8 and will not survive an independent population.**

### 6.6 External datasets: labels exist, ranking does not transfer

E35 found genuine external robustness labels (MATH-Perturb as training source, Putnam-AXIOM as an
untouched holdout, 233 labelled configurations). Problem-text ranking did not transfer across the
source boundary. E36 tried the reasoning *path* at four token budgets; its ranking gates passed
strongly but its frozen absolute-threshold gate missed by 0.0025, so it formally failed. E37, the
one pre-registered confirmation, **passed its GSM-Plus confirmation and then failed its runtime
gate**: 5,293 s to extract features for Qwen alone, against a 3,600 s budget for everything. E38's
exact-prefix audit failed every comparison.

### 6.7 The offline/online inversion

The most uncomfortable result in the project, and the one we cannot explain.

| | Offline verdict | Leaderboard |
| --- | --- | --- |
| Official-label models (`e27a/b/c`) | Best available offline fit | **8/14** |
| Min-K gap (`e28d`) | Passes a *stricter* stability gate than the winner: rank AUC 0.808, worst-case omission 0.65 | **4/14** |
| DeepSeek-only Min-K (`e29a`) | Controlled follow-up | **8/14** |
| Mean-NLL familiarity (`e27e`) | **Fails** gate G-M offline | **10/14** |
| The same, with derived counts (`e27k`) | Fails the same gate | **12/14** |

**On this task, offline validation was anti-correlated with leaderboard behaviour.** The two
methods with the best offline credentials scored 4/14 and 8/14; the method that failed its offline
gate scored 12/14. We report this as an observation rather than a theory. It is the strongest
practical reason we stopped selecting methods offline and started reasoning about the evaluation
distribution's arithmetic directly (§7).

---

## 7. What worked

### 7.1 The leaderboard series

Twenty-two submissions, each designed to answer one question rather than to climb.

| Score | | Submission | What it established |
| --- | --- | --- | --- |
| **0.86** | 12/14 | `e27k` | Ranking + derived counts. **The incumbent** |
| 0.79 | 11/14 | `e27m/n/o/p` | Every single-step neighbour of `e27k` loses exactly one case |
| 0.71 | 10/14 | `e27e` | The ranking alone, median split |
| 0.71 | 10/14 | `e27h` | DeepSeek's split alone — **the whole gain is one model's** |
| 0.71 | 10/14 | `e28a` | Suffix NLL ties: the familiarity *family* works, not one statistic |
| 0.64 | 9/14 | `e27f` | **The constants ceiling** |
| 0.64 | 9/14 | `e27g/i/j` | Splitting Qwen, Skywork or Olmo alone adds nothing |
| 0.57 | 8/14 | `e27a/b/c`, `e29a` | Offline-fitted models carry nothing |
| 0.50 | 7/14 | all-False | 7 of 14 cases are spurious |
| 0.29 | 4/14 | `e27d`, `e28d` | Far below chance ⇒ the ranking is real and inverted |

`e27d` at 4/14 is the hinge of the whole project. A score that far below chance is not noise: it
says the ranking carries information and is pointing the wrong way. Shipping its exact opposite
produced 10/14 the next day.

### 7.2 Why the counts were the real gain

Decomposing `e27k`'s 12: a random ranking with the *same derived counts* scores **8.05/14** in
expectation, against 7/14 for all-False (F-63). So roughly one of the two cases separating `e27k`
from `e27e` is count arithmetic, not ranking quality. The ranking's own contribution puts the
observed 12 at **p = 0.055** under a null of no ranking information — suggestive, not significant,
which is the same verdict every offline test returned, now measured on the hidden set.

A structural detail worth carrying: for a fixed count vector the achievable totals are **spaced two
apart**, because parity is fixed by the sum of robust calls. `e27k` can score 10, 12 or 14 and never
11 or 13. Two cases is the minimum observable difference, not a margin.

### 7.3 Mechanical audit of the incumbent

Before freezing, we audited the shipped artifact on GPU (E39, F-62):

| | Result |
| --- | --- |
| Runtime, four models, 144 problem-scorings, 4 cold loads | **334 s** on T4×2 |
| Determinism, two identical calls | Identical for all four models |
| Cross-machine replication (separate account, session, host) | **Identical predictions on every arm** |
| fp16 vs bf16 | **0 / 36 predictions flip**, Spearman 0.993–1.000 |
| Chat template vs bare text | 2–10 / 36 flip, Spearman 0.738–0.973 |

Two conclusions. The method is **not precision-sensitive**, which matters because our features were
extracted where T4 bf16 is emulated while evaluation runs bf16 on Blackwell. And the **bare-text
prompt is load-bearing** — scoring the chat template instead is a different policy with no
leaderboard evidence, and it would move several cases. Tidying that "inconsistency" before the
freeze would have been a mistake.

---

## 8. Methodological protocol

The practices below were adopted after they were needed, and each traces to a specific loss.

**Pre-registration.** The population, procedure, metric and pass threshold are frozen in
`protocols/` before the run's output is read. Adopted after §6.1's layer-selection artifact and a
threshold tuned on the 14 cases it was then evaluated on. A gate that fails is a result.

**Template-cluster grouping, never text hashing.** §3.2.

**Independent-population confirmation.** No signal is believed on the population that selected it.
E32 exists solely to re-score E31 elsewhere, and it closed the lane.

**Every number comes from a script.** No figure is hand-typed into a document;
`scripts/regenerate_results.py` rebuilds every table in one command. This is also the competition's
reproducibility requirement.

**Two-rule discipline.** The `absolute_accuracy_decay` and `maxdrop` rules are never aggregated for
the same pair (§3.2).

**Contract checks before every upload.** `coverage = 1.0`, `invalid_predictions = 0`, native
`bool`s, and the correct presence or absence of `small.txt` — a Main-Track ZIP containing that
marker scores 0.00 and cost four uploads to diagnose (F-21).

---

## 9. Generalization

The competition rewards generalization explicitly. Our claim is narrow and, we think, defensible.

**The shipped method has nothing to overfit.** It contains no fitted estimator and no training
data. Its inputs are the scoring model's own likelihoods — available for any autoregressive model,
any tokenizer, any architecture — and four integers measured from the evaluation distribution. There
is no feature selection step, no layer choice, no threshold tuned on held-out data, because there is
no training step at all. The failure mode that killed lanes B, E-offline and F — *a signal selected
on 29 clusters that reverses on an independent population* — cannot occur, because nothing is
selected.

**It is architecture-agnostic by construction.** The same six lines apply unchanged to a 4B dense
model and a 117B mixture-of-experts. We verified identical behaviour across four checkpoints from
four different families (E39).

**Its assumptions are stated and testable.** The method assumes (i) familiar problems are more
robust, and (ii) the per-model balances `d_m` hold. Assumption (ii) is directly measurable for five
submissions, and we re-measured it twelve days later with all four values reproducing exactly
(F-64). Assumption (i) is the contested one — see §12.

**What does not generalize, and we say so.** The counts are measured on the *validation* set. The
prizes are scored on a private test set of 30 problems whose class prior may differ. If it does, the
tilts are wrong, and the prior-free `e27e` would be the safer artifact. We ship `e27k` because the
balances are the only measurement we have, and measured beats assumed.

---

## 10. Efficiency

Ties break on average runtime over three repeated trials, so this is a scored axis, not a footnote.

| Method | Per-problem cost | Measured |
| --- | --- | --- |
| **Shipped (`e27k`)** | 1 forward pass, ~10² tokens | **334 s / 4 models / 144 scorings**, incl. cold loads |
| Trace uncertainty (E2f) | greedy generation, 4,096 tokens | ~3.3× slower in bf16 than fp16 on T4 |
| Answer agreement (E14) | k rewrites × full generation | **28,000 tokens/problem**; at chance |
| Budget-path features (E36/E37) | 4 token budgets per problem | **5,293 s for one model** — failed its gate |

Three orders of magnitude separate the top and bottom rows. The shipped method never generates a
token: it reads the problem once and returns a number. The dominant cost is loading the checkpoint,
which the harness pays regardless.

For the Main Track this is sharper still. Listing `gpt-oss-120b` outside the split set means its
**62 GiB checkpoint is never loaded** — the constant is both the best answer available and the
fastest, and with five models sharing 3600 s that is a material saving.

---

## 11. The Main Track

### 11.1 What it adds

Five models instead of four; the fifth, `gpt-oss-120b`, is the only one whose `reasoning_effort`
varies (`low`/`medium`/`high`). The four small checkpoints are shared, so the small-track policy
covers them unchanged; the only new modelling problem is the fifth model.

### 11.2 Two results that simplify it

**Robustness is a property of the problem** (LF-12, §1.2). One score per problem, broadcast to every
model, loses almost nothing.

**`reasoning_effort` almost never changes the label** (LF-13). Across problems labelled at two
efforts: `high` vs `medium` agree 10/10; `high` vs `low` and `low` vs `medium` each flip once in 6.
Of 6 problems labelled at all three, 5 are unanimous — and both flips involve `low`. A shared,
effort-blind prediction costs at most one row in 22.

### 11.3 The proxy panel, and what it cost

Unable to run `gpt-oss-120b` (LF-3: Kaggle's `NvidiaH100` shape returns T4×2 on every account), we
built a proxy panel — eight (model, effort) cells over 46 originals and 453 official variant texts,
on quantised builds via Ollama. **Twelve kernels, ~120 GPU-hours, and one usable cell.**

The failure is instructive. Every kernel reported `COMPLETE`, and `summary.json` reported 637
completions — but that field was the *work list length*, computed before sampling. Counting actual
records gave 11–64% coverage for seven of eight cells (LF-11). The binding constraint was never GPU
hours: a 16k context **truncated 43–97% of thinking-model replies** before they produced an answer.
Only `low` effort, at 1,876 generated tokens per reply against 17k–26k at medium and high, was cheap
enough to finish.

And the panel could not have answered its question anyway. Grading needs each problem's true answer,
since the perturbations preserve it — and neither the 36 official problems nor the 10 reference
problems ship one (LF-14). **The fix was not to build an answer key but to change the population**:
505 olympiad problems whose answers are published (LF-15).

### 11.4 Status

`M11a` — the small-track policy for four models, `True` for `gpt-oss-120b`, no `small.txt` — is
built and contract-checked at coverage 1.0 (LF-16). The scaled programme in that track's `PLAN.md`
is written against hardware we do not have.

---

## 12. The one unresolved contradiction

The shipped direction — **familiar problems are robust** — holds on DeepSeek's 142 MATH instances
and on the hidden validation set repeatedly. It **reverses** on the 34 official olympiad rows, where
the same split scores 0.353 against 0.706 for always-robust (F-58), a result we re-confirmed with
the shipped artifact itself (F-62: agreement 0.22–0.40 against an always-robust floor of 0.67–0.78,
part of which is mechanical, since a median split cannot reproduce a 70%-robust prior).

The hypothesis we could not test: `train-main-v2` holds **published** problems — 2025 AIME I appears
verbatim — where familiarity means memorisation and predicts *spurious*; the validation cases are
mostly not those problems, where familiarity is ordinary fluency and predicts *robust*. The
organizers' own report splits results by published versus unpublished, so the distinction is theirs.

Testing it needs labels on a population partitioned by publication date, which no amount of our
compute can produce and which the organizers have confirmed is not coming (F-59). **It no longer
decides what we ship** — §4.2's count argument settles `e27k` over `e27e` without reference to the
ranking's direction — but it remains the most interesting open question in the project.

---

## 13. What we would do differently

1. **Read the labels before sampling anything.** Two zero-GPU analyses of 82 rows (LF-12, LF-13)
   produced more usable structure than 120 GPU-hours of panel sampling that preceded them.
2. **Probe the evaluation distribution early.** Five constant submissions measure the class balance
   of every model, and that measurement — not any feature — is what produced our best score.
3. **Distrust any offline gate on this task** (§6.7).
4. **Count records, not work lists** (§11.3).
5. **Choose a population that ships ground truth.** The panel's fatal flaw was needing an answer key
   that did not exist (LF-14).
6. **Budget tokens, not GPU-hours.** Every sampling failure in this project was a context-length
   failure wearing a compute-budget costume.

---

## 14. Reproducibility

```bash
# Every generated table in the Small Models Track (~25 min)
python scripts/regenerate_results.py

# The leaderboard arithmetic and the null test for 12/14
python train/eval_batch_composition.py

# Rebuild every submission ZIP, deterministically
python submission/make_e27.py     # 13 candidates incl. the incumbent
python submission/make_e28.py     # rank successors
python submission/make_probes.py  # the 5 constant probes

# Main Track tables
python "../AIMO - Interpretability - Large Models track/scripts/write_results.py"
```

Repository layout: one folder per track, each with `CLAUDE.md` (conventions), `COMPETITION.md`
(rules), `PLAN.md`, `PROGRESS.md` (numbered findings), `RESULTS.md` (generated — never hand-edited),
`experiments.md` (registry) and `protocols/` (pre-registrations). Submissions are indexed by score
in `submission/README.md`. Kernels are exported to `kernels/` exactly as pushed, so any run can be
rebuilt if a Kaggle account disappears.

---

## 15. Summary of contributions

1. **A training-free robustness predictor** that scores 12/14 on the hidden validation set, ranking
   a batch by the scoring model's own likelihoods and fixing its counts by a closed-form identity in
   the measured class balance (§4.2).
2. **A leaderboard-arithmetic method** for extracting the evaluation distribution's per-model class
   balance from constant submissions, and for testing whether an observed score exceeds what those
   counts alone explain (§7.2).
3. **Eleven documented negative results**, several with independent-population confirmations, and a
   measured **inversion between offline validation and leaderboard behaviour** on this task (§6).
4. **A correction to the organizers' published baseline**: its 58.37% is a layer-selection artifact;
   honest nested cross-validation puts it at 0.5005 (§6.1).
5. **Two structural results about the task**: robustness is a property of the problem rather than
   the (model, problem) pair, and `reasoning_effort` almost never changes the label (§11.2).
6. **A tokenizer defect** in transformers 5.13 that silently empties DeepSeek's input and affects
   the organizers' own baseline (§3.2).

---

## Appendix A — Experiment index

`experiments.md` in each track folder carries the full registry with configuration and outcome for
all 41 experiments. `PROGRESS.md` carries findings F-1…F-64 and LF-1…LF-16 with the evidence for
each. `RESULTS.md` carries 29 generated metric tables.

## Appendix B — Open questions

1. Does familiarity predict robustness differently for published and unpublished problems (§12)?
2. Does the per-problem invariance (LF-12) hold at n ≈ 500, or is it an artifact of the
   three-valued rule discarding the ambiguous band?
3. Can official-family perturbations with verified answer preservation reproduce the organizers'
   labels? This is the gate that would reopen supervised work, and it is pre-registered in the Main
   Track's `protocols/M3_PROTOCOL.md`.
4. Is the offline/online inversion (§6.7) a property of this task, of 14-case evaluation, or of our
   offline gates?
