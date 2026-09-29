# PAPER_DRAFT

The technical report for the AIMO Interpretability Challenge, due **15 November 2026** for award
consideration, plus a peer review of up to three other reports between 15 and 30 November.

| File | What it is |
| --- | --- |
| `PAPER.md` | The whole draft. Written to be read end to end by someone who has never seen the repository |

## How to read it

Fifteen sections, and they answer different questions:

- **New to the project?** §1 (the task and why it is hard), §4 (what we actually ship), §7 (what
  worked), §13 (what we would do differently). That is the manual.
- **Want the evidence?** §6 is the failures, one subsection per lane, each with the test that
  killed it. It is the longest section on purpose — the writeup prize is awarded partly on negative
  results, and we have eleven.
- **Want the method?** §4.2 is the one piece of real formalism: the per-model robust count in
  closed form, `t = (d − 1)/2`, derived from a measured class balance rather than tuned.
- **Judging it?** §9 generalization, §10 efficiency, §15 contributions.

## What it is not, yet

This is a draft, and three things are missing:

1. **Figures.** Four would carry their weight: the layer curve behind §6.1's selection artifact,
   the leaderboard series of §7.1 as a ladder, the null distribution of §7.2, and the
   offline-versus-leaderboard inversion of §6.7 as a scatter.
2. **A related-work section.** Membership inference (Min-K%, Min-K%++, zlib ratio), GSM-Symbolic and
   the perturbation-robustness literature, and the probing literature the baseline comes from.
3. **Author list, affiliations and licence.**

## Rules it has to satisfy

- Organizers must be able to reproduce the score **from the report alone** — hence §14, and hence
  the rule that no number is hand-typed into a document.
- The implementation must be open-sourced alongside it.
- Award-considered teams peer-review up to three other reports.

## Keeping it honest

Every number in `PAPER.md` comes from a script in one of the two track folders. If a number here
and a number in a `RESULTS.md` table disagree, **the table is right** — it was generated, the paper
was typed. Re-check with `scripts/regenerate_results.py` before submitting.
