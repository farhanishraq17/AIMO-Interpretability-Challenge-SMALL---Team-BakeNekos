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
| `fig2_leaderboard_ladder` | §7.1 | All 20 uploads, with the chance and constants-ceiling lines |
| `fig3_null_distribution` | §7.2 | Correct counts alone score 8.05; the observed 12 is at p = 0.055 |
| `fig4_inversion` | §6.8 | Offline AUC against leaderboard score — the relation runs backwards |

Colours come from a palette validated for colour-vision deficiency (adjacent-pair ΔE and
contrast), and every series carries a direct label rather than relying on hue alone.

## Before this can be submitted

Item 1 is done; item 2 is still open.

1. **Number audit: done 14 October.** All 395 numeric tokens were checked for a source, every line
   holding two or more for a single source paragraph containing them all, and every single-number line
   for a source sharing its context. Six errors were corrected, none of which changes a conclusion:
   E14's agreement result (it carried E12's numbers), the E10/E4 fusion figures (now the generated
   0.6916 / 0.7305 and 0.7245 / 0.7410 on 138 instances, replacing `PROGRESS.md`'s older values), the
   0.5005 baseline figure (the artifact's own per-layer mean, not nested selection, which gives 0.552),
   the artifact size (a 3.3 KB ZIP of 6.3 KB source), the longest problem (609-667 tokens by tokenizer,
   from E23's records) and the upload count (20). Re-audit after any edit that adds a number.
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
