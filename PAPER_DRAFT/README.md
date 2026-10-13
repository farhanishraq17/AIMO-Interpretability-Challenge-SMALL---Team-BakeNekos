# PAPER_DRAFT

The technical report for the AIMO Interpretability Challenge, due **15 November 2026** for award
consideration, plus a peer review of up to three other reports between 15 and 30 November.

| File | What it is |
| --- | --- |
| `PAPER.md` | The whole draft. Written to be read end to end by someone who has never seen the repository |

**Small Models Track only.** The Main Track's experiments are not complete, so the paper carries no
Main Track content — no sections, results or `LF-n` findings. See that track's `CLAUDE.md`.

## How to read it

Fifteen sections, a verified reference list and two appendices. They answer different questions:

- **New to the project?** §1 (the task and why it is hard), §4 (what we actually ship), §7 (what
  worked), §13 (what we would do differently). That is the manual.
- **Want the evidence?** §6 is the failures, one subsection per lane, each with the test that
  killed it. It is the longest section on purpose — the writeup prize is awarded partly on negative
  results, and we have sixteen.
- **Want the explanation?** §6.2 and §6.9: our training labels and the competition's are decided
  by different perturbation families, and on the competition's the models answer through an
  impossible premise.
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
| `fig4_inversion` | §6.8 | Offline AUC against leaderboard score — the relation runs backwards |

Colours come from a palette validated for colour-vision deficiency (adjacent-pair ΔE and
contrast), and every series carries a direct label rather than relying on hue alone.

## Before this can be submitted

Two things, and the first is the real one.

1. **Audit every number against its generated table.** Nobody has done this end to end. One known
   mismatch: the generated E10 and E4 fusion tables in `RESULTS.md` were rerun on 138 of 153
   instances and give text + prompt states **0.6916** balanced accuracy (AUC 0.7305), where the
   paper uses `PROGRESS.md`'s **0.7173** (AUC 0.7230). That touches §6.2, §6.4 and rows 2 and 7 of
   the §6 failure table. The citations are done: all 25 references carry identifiers and were
   checked against arXiv or the publisher in October.
2. **The author list is incomplete.** One name is filled in. Add co-authors, affiliations, and a
   licence for the repository.

Optional, in rough order of value: a diagram of the method in §4 (currently prose and pseudocode),
and per-figure alt text if the submission format wants it.

## Rules it has to satisfy

- Organizers must be able to reproduce the score **from the report alone** — hence §14, and hence
  the rule that no number is hand-typed into a document.
- The implementation must be open-sourced alongside it.
- Award-considered teams peer-review up to three other reports.

## Keeping it honest

Every number in `PAPER.md` comes from a script in the Small Models Track folder. If a number here
and a number in a `RESULTS.md` table disagree, **the table is right** — it was generated, the paper
was typed. Re-check with `scripts/regenerate_results.py` before submitting.
