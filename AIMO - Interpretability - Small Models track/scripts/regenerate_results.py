"""Rebuild every generated section of RESULTS.md, in the only order that works.

`train_text_baseline.py` rewrites RESULTS.md from scratch and `check_dataset_confound.py`
appends to it, so those two go first; every later script upserts its own section. Steps whose
inputs are missing on this machine (a run not downloaded here) fail loudly rather than silently
leaving a stale section. The three sections only the `nakib_v2` checkout can regenerate are
read before the rewrite and put back afterwards, unchanged (see CLAUDE.md, "Commands").

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

# Generated on the `nakib_v2` checkout from data this machine does not hold.
CARRIED = ("Self-generated olympiad labels", "Labelling cost model", "E21: Stage A on olympiad labels")
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
