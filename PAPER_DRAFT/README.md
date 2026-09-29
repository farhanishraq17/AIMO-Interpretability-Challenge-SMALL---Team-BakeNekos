# PAPER_DRAFT

The technical report for the AIMO Interpretability Challenge, due **15 November 2026** for award
consideration, plus a peer review of up to three other reports between 15 and 30 November.

| File | What it is |
| --- | --- |
| `PAPER.md` | The whole draft. Written to be read end to end by someone who has never seen the repository |

## How to read it

Fifteen sections, and they answer different questions:

- **New to the project?** §1 (the task and why it is hard), §4 (what we actually ship), §7 (what
  worked), §14 (what we would do differently). That is the manual.
- **Want the evidence?** §6 is the failures, one subsection per lane, each with the test that
  killed it. It is the longest section on purpose — the writeup prize is awarded partly on negative
  results, and we have eleven.
- **Want the method?** §4.2 is the one piece of real formalism: the per-model robust count in
  closed form, `t = (d − 1)/2`, derived from a measured class balance rather than tuned.
- **Judging it?** §9 generalization, §10 efficiency, §15 contributions.

## Figures

`figures/` holds four, as `.svg` (use these) and `.png` (previews). Regenerate all four with

```bash
python PAPER_DRAFT/make_figures.py
```

**Run it with system Python, not the pinned competition venv** — that venv exists to serialise
submission artifacts at the competition's library versions, and matplotlib is not one of them.

| Figure | Section | What it shows |
| --- | --- | --- |
| `fig1_layer_curve` | §6.1 | Every layer beats the honestly nested probe. The gap is the selection artifact |
| `fig2_leaderboard_ladder` | §7.1 | All 22 uploads, with the chance and constants-ceiling lines |
| `fig3_null_distribution` | §7.2 | Correct counts alone score 8.05; the observed 12 is at p = 0.055 |
| `fig4_inversion` | §6.7 | Offline AUC against leaderboard score — the relation runs backwards |

Colours come from a palette validated for colour-vision deficiency (adjacent-pair ΔE and
contrast), and every series carries a direct label rather than relying on hue alone.

## What it is not, yet

1. **Citations are from memory and unverified.** §13 says so in its own first line. Every author,
   year and title needs checking against the literature before this is submitted — that is the
   single largest remaining risk in the document.
2. **The author list is incomplete.** Only one name is filled in. Add co-authors, affiliations and
   a licence.
3. **No bibliography.** §13 cites inline; a proper reference list is still to be written.

## Rules it has to satisfy

- Organizers must be able to reproduce the score **from the report alone** — hence §15, and hence
  the rule that no number is hand-typed into a document.
- The implementation must be open-sourced alongside it.
- Award-considered teams peer-review up to three other reports.

## Keeping it honest

Every number in `PAPER.md` comes from a script in one of the two track folders. If a number here
and a number in a `RESULTS.md` table disagree, **the table is right** — it was generated, the paper
was typed. Re-check with `scripts/regenerate_results.py` before submitting.
