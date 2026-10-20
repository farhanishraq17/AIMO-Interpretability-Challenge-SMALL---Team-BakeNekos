"""Assert that regenerating RESULTS.md cannot delete a section.

`train/train_text_baseline.py` rewrites `RESULTS.md` from scratch and every later step upserts its
own section, so a section whose producer is missing from `regenerate_results.py` is not left stale
- it is **deleted**. That failure is silent: the command prints success and the file simply gets
shorter. On 1 October an audit found 18 of 37 sections in that state.

This asserts the invariant: every `## ` heading in `RESULTS.md` is either produced by a script in
`STEPS`, passed as an explicit `--section` argument there, or listed in `CARRIED`.

Run it before trusting `regenerate_results.py`, and after adding any section.

Run:  python scripts/check_results_coverage.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGEN = ROOT / "scripts" / "regenerate_results.py"
RESULTS = ROOT / "RESULTS.md"


def main() -> None:
    regen = REGEN.read_text(encoding="utf-8")
    headings = [line[3:].strip() for line in RESULTS.read_text(encoding="utf-8").splitlines()
                if line.startswith("## ")]

    carried = re.search(r"CARRIED = \((.*?)\)", regen, re.S).group(1)
    steps = set(re.findall(r'"(train/[A-Za-z0-9_]+\.py)"', regen))

    # A heading can be covered three ways: named in CARRIED, passed as --section in STEPS, or
    # written by a script that STEPS runs.
    produced: dict[str, list[str]] = {}
    for path in sorted((ROOT / "train").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for heading in headings:
            if heading in text:
                produced.setdefault(heading, []).append(f"train/{path.name}")

    missing = []
    for heading in headings:
        if heading in carried or heading in regen:
            continue
        if any(script in steps for script in produced.get(heading, [])):
            continue
        missing.append((heading, produced.get(heading, [])))

    for heading, producers in missing:
        where = producers[0] if producers else "no producer found anywhere"
        print(f"UNCOVERED  {heading}  <- {where}", file=sys.stderr)

    if missing:
        print(f"\n{len(missing)} of {len(headings)} sections would be DELETED by "
              f"regenerate_results.py.\nAdd each producer to STEPS, or the section to CARRIED.",
              file=sys.stderr)
        raise SystemExit(1)
    print(f"all {len(headings)} RESULTS.md sections are covered by regenerate_results.py")


if __name__ == "__main__":
    main()
