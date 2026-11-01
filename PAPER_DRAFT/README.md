# PAPER_DRAFT

The technical report for the AIMO Interpretability Challenge, due on OpenReview by **5 November 2026
(AoE)** for award consideration, plus a peer review of up to three other reports between 5 and 12
November. Decisions are announced on 15 November.

| File | What it is |
| --- | --- |
| `PAPER.md` | The whole draft. Written to be read end to end by someone who has never seen the repository |
| `REPORT_STANDARD.md` | What the report is judged on, with the rules as read on 1 November |
| `EXEMPLAR_REPORTS.md` | Seven real competition reports, analysed |
| `PAPER_DRAFT.md` | An alternative draft of 2 October, superseded by `PAPER.md` |
| `Project_Manual.pdf` | The project manual as of 8 October |

**`PAPER.md` is the long form, not the PDF we upload.** The form takes a 5-page body in the NeurIPS
template; the AI use statement, limitations, references and appendix do not count. See "Before this
can be submitted".

**Small Models Track only.** The Main Track's experiments are not complete, so the paper carries no
Main Track content — no sections, results or `LF-n` findings. See that track's `CLAUDE.md`.

## How to read it

Fifteen sections, a verified reference list and two appendices. They answer different questions:

- **New to the project?** §1 (the task and why it is hard), §4 (what we submit: `e64-difficulty`,
  predicted difficulty from one forward pass), §7 (what worked), §13 (what we would do differently).
  That is the manual.
- **Want the evidence?** §6 is the failures, one subsection per lane, each with the test that
  killed it. It is the longest section on purpose — the writeup prize is awarded partly on negative
  results, and we have twenty. §6.10-6.12 are the per-problem ones: familiarity stays at the
  constants (E61, E63), E64 reverses on DeepSeek MATH (E65), and a 64k-cap refit fails its gates (E66).
- **Want what changed on 15 October?** §7.4: per-problem scoring turned our 12/14 into the constants'
  9/14. §7.5: E64 reads 10/14 per problem (submission 1010961, 12th of 93) and is the submission we nominate on OpenReview.
- **Want the open problem?** §9: familiarity and difficulty both flip between DeepSeek's MATH rows and
  the official rows, and the hidden board sides with each depending on how it is read.
- **Want the explanation?** §6.2 and §6.9: our training labels and the competition's are decided
  by different perturbation families, and on the competition's the models answer through an
  impossible premise.
- **Want the two direct tests?** §7.5: steering along E64's direction moves its read of later layers,
  not trace length (E67). §11.1: Qwen flags added contradictions less on familiar problems (E62).
- **Want the method?** §4.1-4.5 is E64: a per-layer ridge from prompt states to Qwen's log trace
  length on 497 unlabelled problems, averaged over layers, cut at each model's share. §4.2 is the one
  piece of real formalism: the share `p_m = (n_m + d_m)/(2 n_m)` and the batch count `t = (d − 1)/2`,
  both derived from a measured class balance rather than tuned. The count acts only inside a batch;
  the share sets E64's cutoffs. §4.6 is the batch method it replaced.
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
| `fig2_leaderboard_ladder` | §7.1 | All 24 scored uploads in two panels (20 batch, then 4 per problem plus `e27k`'s re-score), with the chance and constants lines |
| `fig3_null_distribution` | §7.2 | Correct counts alone score 8.05; the observed 12 is at p = 0.055 |
| `fig4_inversion` | §6.8 | Offline AUC against leaderboard score — the relation runs backwards |

Colours come from a palette validated for colour-vision deficiency (adjacent-pair ΔE and
contrast), and every series carries a direct label rather than relying on hue alone.

## Before this can be submitted

Due on OpenReview by 5 November AoE. Item 1 needs one more pass; the rest are open.

1. **Number audit: done 14 October.** All 395 numeric tokens were checked for a source, every line
   holding two or more for a single source paragraph containing them all, and every single-number line
   for a source sharing its context. Six errors were corrected, none of which changes a conclusion:
   E14's agreement result (it carried E12's numbers), the E10/E4 fusion figures (now the generated
   0.6916 / 0.7305 and 0.7245 / 0.7410 on 138 instances, replacing `PROGRESS.md`'s older values), the
   0.5005 baseline figure (the artifact's own per-layer mean, not nested selection, which gives 0.552),
   the artifact size (a 3.3 KB ZIP of 6.3 KB source), the longest problem (609-667 tokens by tokenizer,
   from E23's records) and the upload count (20). Re-audit after any edit that adds a number.

   **Re-audited 27 October** for the E61-E66 update (§4 rewritten for E64; §6.10-6.12, §7.5 and the
   per-problem parts of the abstract, §1.3, §9, §10, §11, §14, §15 new). Every numeric token on the
   441 added lines was matched to `RESULTS.md`, `PROGRESS.md`, `PLAN.md`, `submission/README.md`, a
   protocol or `kernels/runs/e64-gpu-check-Nipun/result.json`; the 15 lines whose numbers come from
   more than one source (mostly composite tables) were checked cell by cell. No error found. One
   source mismatch was found and not copied: `PROGRESS.md` Finding 89 places E64's D3 correlations
   (−0.35 to −0.50) on the official rows, but `train/eval_e64.py` computes them over E55's 548
   problems, which is what §9 says.

   **Still to audit:** the E67 paragraph (§7.5, F-94) and the E62 paragraph (§11.1, F-95), added since.
2. **The 5-page body does not exist yet.** Cut it from `PAPER.md` in the NeurIPS template and keep the
   rest as the appendix.
3. **The AI use statement is not written.** It is mandatory and goes after the main text, before the
   references: which tasks used which tools, which tasks did not, and that the authors reviewed
   everything and take responsibility. It must match the form's checklist.
4. **The author list is incomplete.** One name is filled in. Every author needs an OpenReview profile.
   Keep names, the email and the repository link out of the PDF body: the form asked for anonymization
   on 31 October, and on 1 November nothing says the request was dropped or kept.
5. **The nomination.** 1010961 on the small track. A second ID is optional, must be one the report
   describes, and the better of the two counts.
6. **The white-box award.** E64 reads hidden states, so ticking it requires an ablation with and
   without that component in the report.
7. **The code is not open yet.** The repository returned 404 to an anonymous request on 1 November and
   has no licence file. Open it with a licence, or upload the code as a supplementary ZIP of 50 MB or less.

Optional, in rough order of value: a diagram of the method in §4 (currently prose and pseudocode),
and per-figure alt text if the submission format wants it.

## Rules it has to satisfy

- Organizers must be able to reproduce the score **from the report alone** — hence §14, and hence
  the rule that no number is hand-typed into a document.
- The implementation must be open-sourced alongside it.
- Award-considered teams peer-review up to three other reports, 5-12 November. A listed reciprocal
  reviewer who misses a review gets the report desk rejected.
- One report per first author, and no two reports on highly similar methods.

## Keeping it honest

Every number in `PAPER.md` comes from a script in the Small Models Track folder. If a number here
and a number in a `RESULTS.md` table disagree, **the table is right** — it was generated, the paper
was typed. Re-check with `scripts/regenerate_results.py` before submitting.
