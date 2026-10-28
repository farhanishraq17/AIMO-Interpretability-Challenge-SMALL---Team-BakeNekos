Investigation complete. I read PLAN, PROGRESS (all 85 findings), RESULTS, the experiment log, all 13 protocols, the paper draft, and the actual shipped solution code, then re-derived the numbers myself.

Here's what I found, worst first.

---

# Problem 1: our submission is probably worse than a one-line answer

This is the big one, and it's already in our own records.

[PROGRESS.md:1760](PROGRESS.md#L1760) — the shipped constants, scored on the 34 official olympiad rows:

| | Qwen | Skywork | Olmo | DeepSeek | **Pooled** |
|---|---|---|---|---|---|
| Best constants (= `e27f`, what we ship) | 0.667 | 0.333 | 0.222 | 0.700 | **0.529** |
| **Always robust** (`return [True]*len(problems)`) | — | — | — | — | **0.706** |

I re-derived it from the per-model label counts and it checks out: 18/34 vs 24/34.

**And it's structurally worse than that.** `e27f` answers True for two models and False for two. So for any robust prior `p` shared across models:

| prior p | `e27f` | always-True |
|---|---|---|
| 0.30 | 0.500 | 0.300 |
| 0.50 | 0.500 | 0.500 |
| 0.706 | **0.500** | **0.706** |
| 0.80 | 0.500 | 0.800 |

**`e27f` is pinned at exactly 0.500 for every prior.** The two True's and two False's cancel. Its *only* possible edge is if Skywork and Olmo genuinely lean spurious while Qwen and DeepSeek lean robust.

And that is exactly where our two data sources disagree:

| model | validation (14 cases) | official labels (34 rows) | agree? |
|---|---|---|---|
| Qwen | 3/5 = 0.600 | 8/12 = 0.667 | yes |
| DeepSeek | 2/3 = 0.667 | 7/10 = 0.700 | yes |
| **Skywork** | 1/3 = 0.333 | 2/3 = 0.667 | **NO** |
| **Olmo** | 1/3 = 0.333 | 7/9 = 0.778 | **NO** |

## Why I think this happened

The per-model distinction rests on **three validation cases each** for Skywork and Olmo — a one-case difference, read off the leaderboard. The organizers' own labelled olympiad data, 34 rows under the same labelling rule, says **all four models are majority robust** — i.e. no per-model distinction at all. Official labels imply always-True.

The team saw this. [PROGRESS.md:1764](PROGRESS.md#L1764) logs it verbatim: *"Gate G2's official-row leg would therefore block the policy that wins on the leaderboard, and that conflict is a freeze decision, not a modelling one."* It was resolved in favour of the board on 26 Sep, and `e27f` inherited it.

Two things then changed and nobody re-priced it:

1. **The 14 Oct rule** ("no fitting to the 14 cases") arrived *after* these constants were chosen from the 14 cases. They were never re-examined under it.
2. **The per-problem change** removed the excuse. In batch mode, the defence was mechanical — a median split *can't* reproduce a 70%-robust prior. Per-problem, constants can reproduce any prior you like. The excuse evaporated; the constants stayed.

And the gate language locked it in: [E57_PROTOCOL.md:54](protocols/E57_PROTOCOL.md#L54) requires a new model to beat *"the constants' (0.529)"* — the 0.706 sitting two lines above it in our own table was never made the bar.

---

# Problem 2: we have no method, and the one we had was a harness artifact

`e27k`'s 12/14 came from counting how many problems to call robust **inside the batch the evaluator happened to pass**. The paper says this plainly ([§13 item 7](PAPER_DRAFT/PAPER.md#L1000)): *"The interface allowed it, the organizers called it overfitting, and one change to the harness turned 12/14 into 9/14."*

**Why:** the method's load-bearing component was a property of the *calling convention*, not of problems or models. Nothing was held in reserve that made an absolute per-problem call, because every per-problem candidate had already been rejected — and that brings us to the instrument.

---

# Problem 3: our offline instrument is anti-correlated with the real score

[§6.8](PAPER_DRAFT/PAPER.md#L529), the thing the paper calls the most uncomfortable result in the project:

| candidate | offline verdict | leaderboard |
|---|---|---|
| Min-K gap (`e28d`) | **best of twelve**: AUC 0.867, CI [0.728, 0.973], maxT p = 0.0010 | **4/14** |
| official-label text models | best available offline fit | 8/14 |
| mean-NLL familiarity (`e27e`) | **failed** its gate | 10/14 |
| the same + counts (`e27k`) | failed the same gate | **12/14** |

**Every lane closure in the last six weeks — E49, E52, E54, E58, E59, E60 — was decided by this instrument.** If it's inverted, some of those lanes were closed on noise.

## Why I think it's inverted

[§6.2](PAPER_DRAFT/PAPER.md#L394) has the answer and it's the deepest finding in the repo: **our labels and the competition's are decided by different perturbation families.**

- On MATH (where 153 of our 226 rows live): `domain` decides — 0.86 of rows drop ≥ 0.25.
- On olympiad (what we're scored on): `expert_no_solution` decides — 0.43, with surface families at ≤ 0.03.

So for most of the project we were fitting a model of *"does this survive a domain swap"* and being scored on *"does this model check its premises."* Those are different questions about different mechanisms. Transfer failing isn't surprising — it's required.

That table needed no GPU and used a public file available on day one. It was **the last analysis we ran, not the first** ([§13 item 6](PAPER_DRAFT/PAPER.md#L1000)).

---

# Problem 4: labels were the binding constraint, and we spent the project on signals instead

- 34 official small-track olympiad rows. 33 labelled problems. 14 validation cases.
- Statistical power ran **14% to 53%** across the lanes we closed ([§11.2](PAPER_DRAFT/PAPER.md#L905)).
- At that power, **most of what we tested could not have reached significance** even if the effect were real.

**Why:** we gated candidate signals rigorously — pre-registrations, template-grouped bootstraps, max-T corrections — but never gated the prior question *"can this experiment possibly resolve anything at n=33?"* Our own [§13 item 5](PAPER_DRAFT/PAPER.md#L1000) says knowing that in week one *"would have redirected the whole project toward acquiring labels."*

Same shape with cost: [§10](PAPER_DRAFT/PAPER.md#L834) shows two method families (E14, E36/E37) were **10–11× over budget and could never have shipped at any accuracy**. The check is one multiplication. We ran them for weeks.

---

# Problem 5: every sampling experiment was truncated, so those negative results are soft

The experiments that closed self-labelling were budget-limited, not idea-limited:

- **E52:** 79% of Qwen's rewrite samples ran out of the 16k cap, and *"several truncated traces had already noticed the clash and were still weighing it."*
- **E59:** *"Most samples are still deliberating at the cap."*
- **E60:** flagging jumped from ~0 to 19/72 (Qwen) and 20/64 (Olmo) once the variant was right — still at our caps.
- **The organizers sample to 100k tokens.** We ran 16k–64k.

**Why:** 2×T4 at 15 GB. An 8B model in bf16 needs 16.4 GB, so everything ran split across two cards with a cache sized for ~20k tokens. E52b died outright — vLLM couldn't hold two 16k Olmo samples. We never once observed these models in the regime where the labels are actually defined.

To be fair to the record: this damages the **labelling** route, not the submission route. The cost gate independently rules these out as submissions (496 min against a 50-min allowance), and no plausible hardware factor closes 10×. So this is why we couldn't *generate labels*, not why we couldn't *ship a method*.

---

# Problem 6: the one live route is outsourced, with no fallback and 4 days left

E57 is built, pre-registered, plumbing-tested, and its features exist for all 548 problems × 4 models. Its **precondition is a third party's gate passing** ([E57_PROTOCOL.md:17](protocols/E57_PROTOCOL.md#L17)).

- No collaborator status recorded since 20 Oct.
- Freeze is 27 Oct — **4 days**.
- If it doesn't land, `e27f` ships. That's the entire contingency.

**Why:** our own compute can't run the 100k-token regime, so the only path to labels left the building. That was a forced move, but it left the project's single remaining upside dependent on something we can't schedule, verify, or chase.

---

# Problem 7: we're ineligible for the White-Box award

Small Models Track, an *interpretability* challenge, **+$1,000** for using internal signals with an ablation showing they help.

Our shipped method reads a token-level NLL. **No activations, no probes, no interventions.** Our entire internals programme — E1, E1b, E4, E30, E31, E32 — is a documented negative result ([§5 lane B](PAPER_DRAFT/PAPER.md#L330): *"AUC up to 0.749 in-distribution; every controlled transfer fails"*).

**Why:** the honest answer is that internals didn't work here and we proved it carefully. But it means the prize designed for this track is out of reach, and the submission has no interpretability content at all.

---

# Problem 8: the board rewards label recovery, and we correctly refused

14 binary cases, 10 submissions/day, 300 total. Flip one case at a time and read the delta — **all 14 labels are recoverable in about 15 uploads.** We did it partially: `e27m/n/o/p` are exactly that, and E50 solved the hidden set to two states. Plus `train-main-v2` revision `25e45665` publicly exposed **7 of the 14 labels** before the organizers withheld them.

`ishitarane767` has submissions at 0.71 → 0.86 → 0.93 → 1.00 across consecutive days. That's the signature.

**Why this is a problem anyway:** the 14 Oct rule was correct — prizes are scored on a new private set where recovered labels are worthless. But the consequence is that we now have **no public signal of any kind** about our own method, and we gave up the one thing that would have kept us visible, while the top of the board tells us nothing about who we're actually competing with.

---

# What I think the single root cause is

**We optimised against the only feedback that moved, and it was the wrong signal.**

The 14-case board gave fast, crisp numbers. The official 34 rows gave slow, wide, mostly-null answers. So the board became the instrument — it set the constants, it produced the 12/14, it even became the tie-breaker when our own documented gate (G2) disagreed. Every time the board and the labelled data conflicted, the board won.

Then the board stopped being available (per-problem calls, erased leaderboard), and we were left holding four hardcoded booleans that were never valid for the population we're graded on.

---

# The one change I'd put in front of you

**Ship always-True instead of `e27f`.**

- `e27f` is mathematically pinned at 0.500 expected accuracy for any shared prior. Always-True scores the prior itself.
- Our two sources put the prior at 0.50 (validation, n=14) and 0.706 (official labels, n=34). **Always-True weakly dominates across both.**
- It loses only if the private test set is *below* 50% robust — and no labelled source we have suggests that.
- Zero runtime, so the tie-break is unaffected. It's a one-line change to an existing builder.

The counter-argument, stated fairly: the validation set is the organizers' own competition data and may be balanced by design, in which case the test set may be too — and `e27f` beat always-True there by 2 cases. That's a real bet, and it's your call. But note that `e27f`'s edge there came from **3 validation cases per model** on precisely the two models the 34 official rows contradict.

Want me to build the always-True ZIP alongside `e27f` so the decision is a file choice rather than a code change, and write up the comparison as a finding?