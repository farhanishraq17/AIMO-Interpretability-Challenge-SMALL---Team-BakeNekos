# CLAUDE.md

This repository holds the **Small Models Track** entry to the AIMO Interpretability Challenge
(NeurIPS 2026). The Main Track entry (adds `gpt-oss-120b`) moved to its own repository on
10 Oct 2026, with its history:
`https://github.com/farhanishraq17/AIMO-Interpretability-Challenge-LARGE---Team-BakeNekos`.
The two share the Kaggle fleet and the competition rules, and each keeps its own plan, progress
log, results, launcher and `Secrets.env`.

| Folder | Track | Start here |
| --- | --- | --- |
| `AIMO - Interpretability - Small Models track/` | Small Models Track (open weights, <10B) | its `CLAUDE.md`, then `PLAN.md` §12, the handover note |

Findings are numbered `1..64` here; the Main Track's `LF-n` findings live in its repository and
are cited here by number only.

`PAPER_DRAFT/` holds the technical report due 15 November. Start at its `README.md`.
**Paper writing is for the Small Models Track only.** The Main Track's experiments are not
complete, so no paper content is written for it (see that repository's `CLAUDE.md`).

**Work inside the track's folder.** Every script resolves its paths from its own location, so
run them from anywhere, but read the track's `CLAUDE.md` before changing anything: the
conventions, the Kaggle fleet notes and the label invariants live there.

No cross-repository imports remain: the Main Track keeps frozen copies of the small-track files it
uses in its `vendor/small_track/`. If you change `datagen/perturb/answers.py`,
`datagen/tokenizer_check.py` or `submission/e27d-familiarity/solution.py`, tell the Main Track.

Never commit `Secrets.env`, and before every commit `git grep --cached -n "KGAT_"` must print
nothing.
