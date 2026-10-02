# Exemplar competition reports, and what makes them win

Compiled 2 Oct 2026. Companion to `REPORT_STANDARD.md`.

**Verification status:** seven papers read through their full HTML text, not abstracts. Two could
not be read in full and are marked. Where a claim is quoted, it is quoted from the paper.

The useful finding is not any single template. It is that **the winners do not agree on structure**,
and two of them win with almost no apparatus at all. What they share is narrower and more
transferable than a section order.

---

## A. Papers read in full

### A1. First-Place Solution, NeurIPS 2024 Erasing the Invisible (watermark removal)

[arXiv 2508.21072](https://arxiv.org/abs/2508.21072) — the closest structural model for us.

> Abstract → Intro → Related Work → **Proposed Approach** → Additional Experiments →
> **Exploratory and Ineffective Approaches** → Conclusion → Appendices

- **§5 is a numbered section for what failed**, with figures showing why: ControlNet edge guidance,
  aggressive diffusion rinsing, white-box perturbations, untrained-network reconstruction. Rare in
  competition writeups, and it legitimises the winning method by contrast.
- **Margins in the abstract**: "outperformed the runner-ups by 26% and 31.7%."
- **Reproduction detail**: learning rate, batch size, epochs, gradient clip norm, diffusion steps,
  and "NVIDIA A6000 (48GB), under two GPU hours." Methods given as numbered pseudocode.
- **Publishes its own manual labels** — cluster assignments for every test image.
- **Discloses metric-gaming**: hyperparameters "tuned to balance removal aggressiveness vs. content
  fidelity," i.e. tuned against the evaluation metric. Saying so costs nothing.

**Read-across, checked against our full text:**

| | Watermark paper | `PAPER.md` |
| --- | --- | --- |
| Failures section | §5, four approaches, qualitative | §6, **fourteen**, each with the test that killed it and a "best result it reached" column |
| Statistics on the nulls | none | §6.7 power analysis **and** the converse |
| Margins in abstract | "26% and 31.7% over runner-up" | 12/14 against the 0.79 that led the board; 334 s against 5,293 s |
| Compute | A6000, "under two GPU hours" | §10 table, 334 s / 4 models / 144 scorings |
| Metric-gaming disclosed | §3.3 | §9 prices the exposure at ~0.89 cases |
| Unglamorous artifacts | manual cluster assignments | SHA-256 of tested ZIP and script; ZIPs committed |
| **Pseudocode** | **numbered algorithms per method** | **§4.3 only — the fourteen failures are prose** |

We are ahead on every row but the last. **That last row is a real gap:** if the prize is partly for
negative results, a reader who wants to reproduce a *failure* currently cannot.

### A2. Solving Trojan Detection Competitions with Linear Weight Classification

[arXiv 2411.03445](https://arxiv.org/abs/2411.03445) — **the closest argumentative analogue to us.**

A deliberately simple method (a hyperplane over raw model parameters) beating elaborate tailored
entries. Their headline framing:

> "Competing methods are tailored to each round, while we run the same method on all rounds."

That is our argument almost word for word — six lines applying unchanged to a 4B dense model and a
117B MoE, no fitted estimator, nothing selected. They lead with **generality**, not with the score.

Two further moves worth copying:

- They **do not attack competitors**. The method is positioned as advantageous because it needs "no
  trigger knowledge or model execution" — an argument about requirements, not a put-down.
- They **name where it fails**: Round 13, when training and test distributions diverge. Honest
  failure inside a winning paper, exactly like our §12.

### A3. NeurIPS 2023 Privacy-Preserving Federated Learning DocVQA

[arXiv 2411.03730](https://arxiv.org/abs/2411.03730) — organizer report, but it documents winners in
detail and the winners wrote their own methodology sections.

- The winning number is a **single dramatic ratio**: baseline 44.65 GB of communication, winner
  **0.38 GB — a 99.14% reduction**. One figure carries the whole result.
- **Formal proofs were required**, not empirical claims: Track 2 submissions needed "theoretical
  privacy proof" plus code review.
- §6.1 is candid about failure: "even small mistakes in the implementation of DP methods can
  invalidate or severely weaken the DP guarantees," and manual review "does not scale."
- A runner-up published a **post-competition negative finding** about their own optimizer.

### A4. LPCVC 2025 — Evaluation of Winning Solutions

[arXiv 2604.19054](https://arxiv.org/html/2604.19054v2) — efficiency-first judging, three tracks.

- Structure: a **latency gate first**, then rank by accuracy. `Final Score = Accuracy / max(Execution
  Time / 2, 1)`. A hard feasibility threshold before quality is scored — structurally identical to
  AIMO's 3600 s budget.
- **Every winning team built on established architectures** (MobileNet, CLIP, Depth-Anything-V2)
  and won through pruning, quantization and resolution tuning. Nobody won by inventing a network.
- Track 1 winner: 0.974 accuracy against a 0.692 baseline, within the latency gate.

**Read-across:** our §10 finding — two method families could never have shipped at any accuracy,
missing the budget by 10× and 11× — is the same insight these organizers built into the scoring. We
discovered by failure what they enforced by design, which makes it a transferable lesson rather than
a local mishap.

### A5. NeurIPS 2024 ML4CFD — Results and Retrospective

[arXiv 2506.08516](https://arxiv.org/html/2506.08516v1) — how organizers judge when accuracy alone
will not do.

- Multi-criteria scoring: **40% ML performance, 30% physics compliance, 30% out-of-distribution
  generalization.** Generalization was nearly a third of the score.
- The organizers found a **shared structural trait across all four winners**: every one "included a
  geometry processing step," explicit or implicit. Winners converged on an insight, not a technique.
- They **state their own protocol's limits**: "further refinement is possible," particularly on how
  training cost enters the comparison.

### A6. RMIT–ADM+S at MMU-RAG, NeurIPS 2025 — *the instructive counter-example*

[arXiv 2602.20735](https://arxiv.org/abs/2602.20735) — won Best Dynamic Evaluation, Open Source.

~4,500 words. Abstract, Intro, Qualitative Evaluation, System Architecture, Conclusions, Appendices
of prompts. And then:

- **No quantitative results tables.**
- **No statistical significance testing.**
- **No failure-case analysis.**
- **No systematic ablations** — design rationale emerges through comparative description.

It won anyway, because the evaluation happened *in the competition*, not in the paper. The report
documents; the competition judges.

**Why this matters to us.** It is evidence against assuming more rigor always wins — but note *what
it won*. "Best Dynamic Evaluation" was awarded on how the system performed in live evaluation, not
on the report's scientific content. AIMO's writeup prize is the opposite setup: "judged on their
scientific contribution," explicitly independent of ranking. For that pot, the apparatus **is** the
product.

So RMIT is the right model for the **1–2 page methods summary** every participant files
(`REPORT_STANDARD.md` §0), and the wrong model for the award report. The two are different
artifacts and we likely owe both.

### A7. AIMO-2 Winning Solution — the sibling competition

[arXiv 2504.16891](https://arxiv.org/abs/2504.16891) — same prize family, different task.

Organised around **"three key pillars"** rather than a chronological account: dataset creation,
methodology, model selection. Three claims under one theme, exactly as the craft advice prescribes.

Its reproducibility stance goes beyond compliance — code, models and the complete 540K-problem
OpenMathReasoning dataset released "under a commercially permissive license." Open-sourcing framed
as **a contribution in itself**, not a rule obeyed.

## B. Read partially

- **[The SaTML '24 CNN Interpretability Competition](https://arxiv.org/abs/2404.02949)** — closest in
  subject. Judged by **human crowdworkers** answering 8-option multiple-choice questions about
  trojan triggers; the metric was the proportion of 100 subjects choosing correctly. Two findings
  transfer. First, the organizers reported that among the entries "**none Pareto dominated any
  other**," and treated that diversity as the encouraging result. Second, they stated plainly that
  their "measures were helpful for clear evaluation, but they do not measure all possible desiderata
  for interpretability methods." **Interpretability-competition organizers know their metric is
  partial** — which legitimises a report arguing its contribution lies beyond the metric. That is
  our §6.8 argument, and this is precedent for it.
- **[The Trojan Detection Challenge](https://proceedings.mlr.press/v220/mazeika23a/mazeika23a.pdf)**
  (PMLR v220) — organizer report, skimmed only.

## C. A correction worth keeping

I expected the **NeurIPS Competition Track PMLR volumes** to be a library of winning participant
reports. They are not. [v220](https://proceedings.mlr.press/v220/) holds 21 papers, essentially all
**organizer retrospectives** — "Lessons Learned from Ariel Data Challenge," "The CityLearn Challenge
2022: Overview, Results, and Lessons Learned." The PMLR route is offered to *organizers* for
post-competition analyses, not to participants. Useful for seeing how organizers judge a field;
useless as a writeup template. Same for
[v123](https://proceedings.mlr.press/v123/) · [v133](https://proceedings.mlr.press/v133/) ·
[v176](https://proceedings.mlr.press/v176/).

---

## What actually makes them win

Across all seven, in rough order of how much it would change our draft.

**1. Generality is the headline, not the score.** The strongest framings are about *scope of
application*: "the same method on all rounds" (A2); one method across vision, NLP and RL (A2);
released weights and data so anyone can rerun it (A7). Winners say what their method *is not
tailored to*. Our claim is unusually strong here — no fitted estimator, nothing selected, identical
behaviour across four model families — and it currently reads as a property of the method rather
than as the headline.

**2. One dramatic ratio beats a table.** 44.65 GB → 0.38 GB. 0.974 against 0.692. "26% and 31.7%
over runner-up." Ours exists and is buried: **334 s against 5,293 s**, and two method families
missing the budget by 10× and 11×.

**3. Problem selection is what distinguishes important from competent.** Carlini's argument, and the
exemplars bear it out: ML4CFD's winners converged on geometry awareness; LPCVC's all optimised
rather than invented. Our equivalent choice — stop selecting methods offline and reason about the
evaluation distribution's arithmetic instead (§7.2) — *is* the taste decision that produced the
score, and it is told as narrative rather than claimed as insight.

**4. Honest failure inside a winning paper.** A1 §5, A2's Round 13, A3 §6.1, A3's runner-up
publishing against their own optimizer. Nobody hides it; everybody gets credit. We have fourteen and
a published unresolved contradiction — further than any of them.

**5. Organizers admit their metrics are partial.** A5 states its protocol needs refinement; the
SaTML interpretability organizers say outright their measures "do not measure all possible desiderata."
A report arguing that the leaderboard number is not the whole contribution is pushing on an open
door — *if* it argues it with evidence, as our §6.8 inversion does.

**6. Feasibility gates before quality.** LPCVC enforces latency, then ranks accuracy. AIMO has the
same shape with 3600 s. Our one-multiplication gate (§10) is the same device, discovered the hard way.

**7. Reproducibility as contribution, not compliance.** A7 released a 540K-problem dataset under a
permissive licence. A1 published its manual cluster labels. A3 required formal proofs. Cheap,
unglamorous, disproportionately credible. We have the ZIP hashes and committed artifacts; what we
lack is pseudocode for the failures.

**8. Separable ablations.** A1 tabulates cumulative contributions. We have the decomposition — §7.2
shows a random ranking with the same counts scores 8.05/14, so one of the two cases separating
`e27k` from `e27e` is count arithmetic — but it is prose and a figure, not a table a reader can scan.

## What none of them had that we do

Most of these are leaderboard winners explaining how they won. We have fourteen negative results
with effect sizes and confidence intervals, a power analysis that says our nulls are partly
measuring sample size *and* says the converse, a documented inversion between offline validation and
held-out behaviour, and an unresolved contradiction we chose to publish.

Not one exemplar does the statistical work on its own nulls that §6.7 does. The writeup prize is
independent of ranking and explicitly interested in negative results. Most teams have nothing to say
when the method does not work. **That is the gap to occupy — and the thing to lead with.**
