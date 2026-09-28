"""Regenerate E38's result table from the preserved Kaggle audit output.

Run: python train/report_e38.py
"""
import json
from pathlib import Path
from results_io import upsert_section

ROOT = Path(__file__).resolve().parents[1]
result = json.loads((ROOT / "kernels/runs/e38-prefix-audit-Tamim3/result.json").read_text())
body = ["Exact-prefix screening only; no correctness labels or robustness scores were read.", "",
        "| Source / model | Problems | Comparisons | Mismatches |", "| --- | --- | --- | --- |"]
for name, cell in result["cells"].items():
    body.append(f"| {name} | {cell['problems']} | {cell['comparisons']} | {cell['mismatches']} |")
body += ["", f"Total: {result['mismatches']:,} mismatches in {result['comparisons']:,} comparisons.",
         f"All frozen gates passed: **{result['passed']}**.", "",
         "The CPU gate failed. E38b is not launched; E37 retains independent budget generations.",
         "Reproduce with `train/audit_e38_prefix.py`; rebuild this table with `train/report_e38.py`."]
upsert_section(ROOT / "RESULTS.md", "E38a: exact-prefix runtime audit", "\n".join(body))
