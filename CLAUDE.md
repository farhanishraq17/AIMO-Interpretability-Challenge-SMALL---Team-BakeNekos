# CLAUDE.md

This repository holds **two entries** to the AIMO Interpretability Challenge (NeurIPS 2026),
one folder each. They share the Kaggle fleet and the competition rules, and each keeps its own
plan, progress log, results, launcher and `Secrets.env`.

| Folder | Track | Start here |
| --- | --- | --- |
| `AIMO - Interpretability - Small Models track/` | Small Models Track (open weights, <10B) | its `CLAUDE.md`, then `PLAN.md` §12, the handover note |
| `AIMO - Interpretability - Large Models track/` | Main Track (adds `gpt-oss-120b`) | its `CLAUDE.md`, then `PLAN.md` §0, the state in ten lines |

Both folders now carry the same document set — `CLAUDE.md`, `COMPETITION.md`, `PLAN.md`,
`PROGRESS.md`, `RESULTS.md`, `experiments.md`, `protocols/` — with the same role for each file.
Findings are numbered separately: `1..61` in the small track, `LF-1..LF-15` in the Main Track.

`PAPER_DRAFT/` holds the technical report due 15 November, which covers **both** tracks and is
the only document that does. Start at its `README.md`.

**Work inside one track's folder.** Every script resolves its paths from its own location, so
run them from anywhere, but read that track's `CLAUDE.md` before changing anything: the
conventions, the Kaggle fleet notes and the label invariants live there.

The only cross-track link: the Main Track's `scripts/write_results.py` reads the small track's
E24a answer key and imports its `\boxed{}` grader.

Never commit `Secrets.env`, and before every commit `git grep --cached -n "KGAT_"` must print
nothing.
