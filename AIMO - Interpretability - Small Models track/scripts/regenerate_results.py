"""Rebuild every generated section of RESULTS.md, in the only order that works.

`train_text_baseline.py` rewrites RESULTS.md from scratch and `check_dataset_confound.py`
appends to it, so those two go first; every later script upserts its own section. Steps whose
inputs are missing on this machine (a run not downloaded here) fail loudly rather than silently
leaving a stale section. The three sections only the `nakib_v2` checkout can regenerate are
read before the rewrite and put back afterwards, unchanged (see CLAUDE.md, "Commands").

**Every script that upserts a section must appear in STEPS.** A missing one is not a stale
section, it is a deleted one: `train_text_baseline.py` rewrites the file from scratch, so any
section nobody regenerates afterwards is gone. On 1 October an audit found 18 of 37 sections in
exactly that state - E28 to E38 and E40 to E48 - which had been silently destroyed by every run of
this script since they were written. `scripts/check_results_coverage.py` now asserts the
invariant.

Run:  python scripts/regenerate_results.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "RESULTS.md"
sys.path.insert(0, str(ROOT / "train"))
from results_io import upsert_section  # noqa: E402

# Sections no script on this machine can regenerate, preserved verbatim across the rewrite.
# The first three come from the `nakib_v2` checkout; the last three were written from runs whose
# generating script lives on a collaborator's checkout. Without this, `train_text_baseline.py`'s
# rewrite silently deletes them.
CARRIED = ("Self-generated olympiad labels", "Labelling cost model", "E21: Stage A on olympiad labels",
           "E29: controlled DeepSeek-only transfer", "E34a — independent AIME label-yield gate",
           "E35 — external robustness labels, source-disjoint holdout")
STAGEA = "runs/e1b-stagea-deepseek/features/stagea-deepseek"
TRACESTATE = "runs/e4-tracestate-deepseek/features/tracestate-deepseek"

STEPS = [
    ["train/train_text_baseline.py"],
    ["train/check_dataset_confound.py"],
    ["train/train_stagea_probes.py", "--features-dir", STAGEA],
    ["train/train_stageb_features.py"],
    ["train/train_stagea_probes.py", "--features-dir", TRACESTATE, "--section", "E4: trace-state probes"],
    ["train/train_fusion.py", "--blocks", "text", "stagea"],
    ["train/train_fusion.py", "--blocks", "text", "stagea", "tracestate", "trace",
     "--section", "E4: fusion with trace states"],
    ["train/train_fusion.py", "--blocks", "text", "stagea", "agreement",
     "--section", "E14: fusion with test-time agreement"],
    ["train/eval_label_free.py"],
    ["train/eval_agreement.py"],
    ["train/summarize_labelling.py"],
    ["train/eval_train_main_v2.py"],
    ["train/eval_official_holdout.py"],
    ["train/eval_problem_level.py"],
    ["train/eval_familiarity.py"],
    ["train/compare_selflabels.py"],
    # E28-E38: added 1 Oct. These existed and were absent from this list, so every run of this
    # script deleted their sections (E48's audit).
    ["train/eval_e27_candidates.py"],
    ["train/eval_e28_stability.py"],
    ["train/eval_e30_geometry.py"],
    ["train/eval_e31_counterfactual.py"],
    ["train/eval_e32_counterfactual_math.py"],
    ["train/report_e37.py"],
    ["train/report_e37_runtime.py"],
    ["train/report_e38.py"],
    # E40-E48: the leaderboard arithmetic and the 1 Oct analyses. All CPU, all seconds to run.
    ["train/eval_batch_composition.py"],
    ["train/eval_prior_sensitivity.py"],
    ["train/eval_familiarity_shared.py"],
    ["train/eval_published_split.py"],
    ["train/eval_maxdrop_continuous.py"],
    ["train/eval_budget_feasibility.py"],
    ["train/eval_power_required.py"],
    # E50: the hidden set solved from every recorded leaderboard score. CPU, seconds.
    ["train/eval_hidden_state.py"],
    # E51: which perturbation families make a label spurious. CPU, needs the e24b runs.
    ["train/eval_label_mechanics.py"],
    # E52: contradiction detection. CPU, needs the e52 runs downloaded.
    ["train/eval_e52.py"],
    # E54: every held signal as a per-problem rule. CPU; needs the e35, e36, e49 and e52 runs.
    ["train/eval_e54_per_problem.py"],
    # E55: coverage and one-pass timing of the feature runs. CPU; needs the e55 runs.
    ["train/report_e55.py"],
]


def section(text: str, heading: str) -> str | None:
    lines = text.splitlines()
    if f"## {heading}" not in lines:
        return None
    start = lines.index(f"## {heading}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start + 1:end]).strip("\n")


def main() -> None:
    before = RESULTS.read_text(encoding="utf-8")
    carried = {heading: body for heading in CARRIED if (body := section(before, heading))}
    for step in STEPS:
        print(f"-> {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", *step], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    for heading, body in carried.items():
        upsert_section(RESULTS, heading, body)
    print("RESULTS.md regenerated")


if __name__ == "__main__":
    main()
