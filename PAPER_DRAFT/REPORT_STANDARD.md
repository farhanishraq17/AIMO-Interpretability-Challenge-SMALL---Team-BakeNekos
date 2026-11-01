# What a prize-winning report has to do

Research compiled 2 Oct 2026 for `PAPER.md`. Companion file: `EXEMPLAR_REPORTS.md`, which analyses
seven real competition reports in depth. Sources are linked inline. Anything attributed to the
organizers comes from `COMPETITION.md`, the competition site, or their arXiv proposal (2607.13899).
Rules, dates and §0 were corrected on 1 Nov against the live site and the OpenReview form.

**Frame:** this is a *competition technical report* judged by the AIMO organizers and submitted on
OpenReview. There is a $5,000 pot, a panel, four stated interests, and a peer-review round where
rival teams read our report. Workshop craft advice applies to the *writing*. It does not apply to
the *rules*.

---

## 0. Report length: a 5-page body, with the evidence in the appendix

**Corrected 1 Nov.** On 2 Oct this section found no page limit on any source. The live rules now state
one, and the OpenReview form agrees on what falls outside it:

| Part | Limit | Source |
| --- | --- | --- |
| Main body | **5 pages, NeurIPS template** | live site, rules and FAQ |
| AI usage statement | not counted, and mandatory | live site, OpenReview form |
| Limitations, references, appendix | not counted | live site |

**Conclusion: `PAPER.md` cannot be the body.** It is the long form. Its evidence can sit in the
appendix, and the body has to carry the argument alone, because rival teams and the panel may read
nothing else. The four stated interests of §2 have to be answered inside those 5 pages.

Also from the site and the form:

1. **Anonymity.** On 31 Oct the form asked for an anonymized body. On 1 Nov neither the form nor the
   site says so. Keep the body anonymous unless the organizers say otherwise.
2. **Reciprocal reviewing can sink the report.** A listed reviewer who misses a review gets the
   report desk rejected and out of the awards.
3. **One report per first author**, and no two reports on highly similar methods.
4. **The proposal's other artifacts** (a 1-2 page methods summary, a 1-page workshop report) appear
   in the proposal only. The site's rules name this one report.

A second, overlooked incentive from the proposal paper, worth knowing because it is not the $5,000:

> "Submissions with strong empirical results **or broadly impactful methodological lessons
> documented in their report** will be invited to present at the in-person competition venue."

That clause is written for a report like ours.

---

## 1. What we are actually graded on

| | |
| --- | --- |
| Prize | **$5,000**, "for selected technical reports, **regardless of their ranking**" |
| Judged on | "scientific contribution" |
| Also at stake | an invitation to present in person, for "broadly impactful methodological lessons" |
| Due | **5 Nov 2026 (AoE)**, on OpenReview |
| Then | we peer-review up to 3 other teams' reports, 5–12 Nov |
| Hard gate | organizers must reproduce our score **from the report alone** |

The leaderboard prize and the writeup prize are **independent**. A report can take the writeup pot
having placed nowhere. This is the most important strategic fact here: the report should not read as
a victory lap for a board score. It should read as a scientific contribution that a competition happened to
validate.

## 2. The four stated interests, decoded

"Generalization of interpretability methods, actionable interpretability, negative results,
efficient methods." Three of the four have established literatures with concrete standards.

### 2.1 Actionable interpretability

A named agenda with an ICML workshop and a position paper, not a buzzword. The
[Hitchhiker's Guide](https://actionable-interpretability-guide.github.io/) grades it on **two axes**:

- **Concreteness** — how specific the action is, from "could inform safety research" to "exact
  implementation with code."
- **Validation** — whether it was tested, from "this might help" to "quantitative evidence it does."

At the ICML 2025 workshop, **21.8% of submissions were flagged by at least one reviewer as
insufficiently actionable**. The failure mode is clustering in the low-concreteness, low-validation
corner.

Their six criteria, which read as a rubric:

1. **Comparative utility** — beats *non-interpretability* baselines, not just other interpretability
   methods
2. **Mechanistic faithfulness** — intervening on identified components produces the predicted change
3. **Generalization** — holds across seeds, architectures, scales
4. **Specificity** — targeted effects, no entanglement side-effects
5. **Task enhancement** — measurable improvement for a real user
6. **Understandability** — the audience can use it

**Where we stand:** unusually strong on concreteness and validation — a deployed artifact, scored by
a third party, on a held-out set. That is the top-right corner most interpretability work never
reaches. We currently never say so: "actionable" appears once in `PAPER.md`, at line 130, inside a
quotation of the prize criteria. We never claim it of our own work.

### 2.2 Negative results

We have fourteen, and §6 is already the longest section. What separates a strong null from an
uninteresting one:

- **Shift the burden from outcome to question + rigor.** The paper must show the experiment was a
  well-powered, valid test of an important hypothesis. A null from a weak test is worthless.
- **Distinguish absence of evidence from evidence of absence.** "We found no effect" is weak; "we
  rule out an effect larger than X" is strong. Needs effect sizes with intervals, equivalence tests
  or Bayes factors.
- **Report nulls as specifically as positives** — effect size, CI, test, n. Not prose.
- **Explain why.** A mechanism for the null turns a dead end into an agenda.

**Where we stand:** §6.7 already does the hardest version of this — a power analysis *plus* the
converse ("we do not know, not that we were nearly right"). That is better than any exemplar in the
companion file. The remaining gap is reproducibility of the failures: see `EXEMPLAR_REPORTS.md`, the
pseudocode point.

### 2.3 Efficiency

Ties break on runtime, and E39 measured 334 s for four models against a 3600 s budget. Two of the
exemplars won substantially on efficiency, and both led with a **single dramatic ratio** rather than
a table — 44.65 GB → 0.38 GB in one case. Our equivalent is 334 s against 5,293 s, and the fact that
two pursued method families could never have shipped at *any* accuracy. That number should be more
prominent than it is.

### 2.4 Generalization

Our biggest risk is also our most honest generalization story: training rows are mostly MATH, the
private set is olympiad, and every block measured across that gap sits at or below chance. §9 should
not soften this. The strongest exemplar framing (see Trojan, in the companion file) is **"the same
method, unchanged, everywhere"** — which is exactly our claim and we should borrow the phrasing.

## 3. Craft standards

### 3.1 The narrative rule

From [Neel Nanda's advice](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)
(an interpretability researcher, so directly on point):

> "The north star of a paper is ensuring the reader understands and remembers the narrative."

**One to three concrete claims** under one theme. Attention is uneven — many read the title, some
the abstract, fewer the intro and figures, almost nobody everything — so:

> "Spend about the same amount of time on each of: the abstract, the intro, the figures, and
> everything else."

### 3.2 Problem selection beats execution

[Carlini on what actually wins awards](https://nicholas.carlini.com/writing/2026/how-to-win-a-best-paper-award.html):

> "One excellent paper is worth a thousand mediocre ones, and takes less time to write."

His argument is that award-winning work differs from competent work in **which problem was chosen**,
not in technical sophistication. Competent work solves a defined problem correctly; important work
identifies which problem deserves solving. Concrete rules he gives:

- Abstracts need **specific numbers** and a clear claim of importance
- Conclusions must answer **"so what?"** — if the honest answer is "we improved a metric by 2%,"
  reconsider the project
- "Put in an unreasonable amount of effort" on experimental rigor
- Make the paper **maximal** — no obvious unfinished improvement left visible

**Read-across:** our "so what" is not a board score. It is that offline interpretability validation was
anti-correlated with held-out behaviour on this task, and that a method with nothing to fit beat
every fitted one. That is a claim about interpretability methodology, and it survives the "so what"
test far better than the score does.

### 3.3 Specific rules worth adopting

- **"Inform, not persuade."** Overclaiming is detected by exactly the readers who matter.
- **Baselines are the classic interpretability failure.** Competitor methods must get equal
  optimisation effort. We are strong here — the constants ceiling and all-False floor are honest and
  unflattering.
- **p < .05 is too weak for exploratory work**; prefer .001. Our headline 12/14 sits at **p = 0.055**
  and §7.2 already treats it as a null test rather than leaning on it. Keep that.
- **Plain language.** "You get points for quality technical insights, not for sounding fancy."
- **Red-team the draft before a competitor does** — and one literally will, in the review round.

### 3.4 The NeurIPS checklist as a quality bar

Not binding, but it is what our peer reviewers have internalised.
[Checklist](https://neurips.cc/public/guides/PaperChecklist) items that matter here:

| Item | What a good answer looks like |
| --- | --- |
| Claims | Abstract and intro match what was shown; aspiration separated from result |
| **Limitations** | A dedicated section. Reviewers are instructed **not to penalise honesty** |
| Reproducibility | For us a hard competition rule, not a nicety |
| Code and data | Open-sourcing is mandatory for award eligibility |
| Experimental detail | Splits, hyperparameters, and *how they were chosen* |
| **Error bars** | What the interval captures, how computed, SD vs SE |
| **Compute** | Hardware, per-run time, total — including unreported preliminary runs |

## 4. Gaps in our draft

Verified against the full text of `PAPER.md` (990 lines, 9,640 words) on 2 Oct. Line numbers are
where to look. Ranked by risk:

1. **Citations are from memory and unverified.** Twenty entries; R3, R4, R7, R17, R18 and R20 are
   self-marked *medium*. Award-eligible teams peer-review each other, so a competitor who finds a
   citation that does not exist as described will say so. The confidence column and checkboxes are a
   good system; it has to be executed. About a day of work.
2. **Eight contributions in §16 (`PAPER.md:877`).** The craft evidence is unanimous on one-to-three.
   A clean three already sits inside the eight: the training-free predictor with counts derived in
   closed form (§4.2); nothing transfers across the MATH→olympiad gap, with the power analysis
   quantifying how underpowered we were (§6.2, §6.7); and offline validation was anti-correlated
   with held-out behaviour (§6.8).
3. **We never claim actionability in their vocabulary.** §2.1 above. A free win.
4. **The failures are not reproducible to the standard the method is.** §4.3 gives pseudocode for
   the shipped policy; the fourteen rejected methods are prose only. If the prize is partly for
   negative results, the negative results need to be reproducible too.
5. **Scope is unsettled.** Subtitle says "both tracks," §11 is the Main Track. For a Small-track-only
   submission this touches §9, §11, §15 and the title. Decide before reorganising.
6. **§6.7 and §12 overlap heavily** — both cover the reversal, both discuss F-69 and F-70, several
   sentences near-duplicate. One should own it.
7. **Appendix A is stale (`PAPER.md:978`)** — says `F-1…F-64`, actual range is `F-1…F-73`.
8. **Author list incomplete** — one name, no affiliations, no licence.
9. **The title undersells** — names the method, hides the negative results and the inversion.

**One thing to push back on rather than fix.** `PAPER.md:791` says "We contribute no new
interpretability method." True, honest, and stated flatly enough that a skimming judge may take it
as the summary. The actual position — a training-free method beat every fitted one, and offline
interpretability metrics were anti-correlated with held-out performance — is a substantive claim
*about* interpretability methodology. It should read as a finding, not a disclaimer.

## 5. What is verified and what is not

**Verified by reading:** everything in §4, with line numbers. All finding references `F-1…F-73`
resolve — an earlier version of this file claimed otherwise on a faulty grep, and that was wrong.

**Not done:** a claim-by-claim audit that each *number* in `PAPER.md` matches its generating script.
The repo's rule is that where the paper and a `RESULTS.md` table disagree, the table wins. Nobody has
run that audit end to end.

---

## Sources

**Competition**
- [AIMO Interpretability Challenge site](https://aimo-interp.github.io/)
- [Proposal paper, arXiv 2607.13899](https://arxiv.org/abs/2607.13899) — the 1–2 page line
- [getting-started issue #10 — format question, unanswered](https://github.com/aimo-interp/getting-started/issues/10)

**Rubric literatures**
- [The Hitchhiker's Guide to Actionable Interpretability](https://actionable-interpretability-guide.github.io/)
- [Position: Interpretability Can Be Actionable, arXiv 2605.11161](https://arxiv.org/abs/2605.11161)
- [Actionable Interpretability workshop, ICML 2025](https://icml.cc/virtual/2025/workshop/39962)
- [Position: Embracing Negative Results in ML, arXiv 2406.03980](https://arxiv.org/pdf/2406.03980)

**Craft**
- [Highly Opinionated Advice on How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)
- [Carlini, How to win a best paper award](https://nicholas.carlini.com/writing/2026/how-to-win-a-best-paper-award.html)
- [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist)
- [NeurIPS Reviewer Guidelines](https://neurips.cc/Conferences/2019/PaperInformation/ReviewerGuidelines)
- [Heuristics for Scientific Writing](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/)
