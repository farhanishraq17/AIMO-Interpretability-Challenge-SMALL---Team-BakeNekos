"""E62's hand audit: 30 kept contradictions drawn at random and judged by reading, before any sample is read.

The keep rule trusts one checker model; §2 of `protocols/E62_PROTOCOL.md` asks whether a person agrees. `sheet` draws 30
kept variants (seed 62) from `data/e62/variants.parquet` and writes `docs/e62_hand_audit.md`: each one's original
problem, the inserted sentence, the generator's own reason, and a verdict line left `pending`. The auditor replaces it
by `genuine` (the sentence contradicts what the problem states or implies, so it has no answer as written) or
`not genuine: <why>`. `count` reads the verdicts back: the audit passes with at least 27 genuine.

Run:  python datagen/perturb/e62_audit.py sheet ; python datagen/perturb/e62_audit.py count
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SHEET = ROOT / "docs" / "e62_hand_audit.md"
N, SEED, PASS = 30, 62, 27


def sheet() -> None:
    e62 = ROOT / "data" / "e62"
    kept = pd.read_parquet(e62 / "variants.parquet").sample(N, random_state=SEED)
    reasons = pd.read_parquet(e62 / "audit.parquet").set_index(["pool_id", "sentence"])["why_impossible"]
    problems = pd.read_parquet(e62 / "eligible.parquet").set_index("pool_id")
    lines = [f"# E62 hand audit: {N} kept contradictions (seed {SEED})", "",
             f"Drawn by `datagen/perturb/e62_audit.py` from {len(pd.read_parquet(e62 / 'variants.parquet'))} kept "
             f"variants, before any contradiction sample was read. Pass: at least {PASS} genuine.", ""]
    for i, v in enumerate(kept.itertuples(index=False), 1):
        problem = problems.loc[v.pool_id]
        lines += [f"## {i}. `{v.variant_id}`", "", f"Answer {problem['answer']}. **Problem:**", "",
                  *(f"> {line}" for line in problem["problem"].splitlines()), "",
                  f"**Inserted:** {v.sentence}", "",
                  f"**Generator's reason:** {reasons.get((v.pool_id, v.sentence), '')}", "",
                  "**Verdict:** pending", ""]
    SHEET.write_text("\n".join(lines), encoding="utf-8")
    print(f"{N} variants -> {SHEET}")


def count() -> None:
    verdicts = re.findall(r"^\*\*Verdict:\*\* (.+)$", SHEET.read_text(encoding="utf-8"), re.M)
    genuine = sum(v.strip() == "genuine" for v in verdicts)
    pending = sum(v.strip() == "pending" for v in verdicts)
    print(f"{genuine} of {len(verdicts)} genuine, {pending} pending: "
          f"{'pending' if pending else 'pass' if genuine >= PASS else 'FAIL'} (pass: >= {PASS})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sheet", "count"])
    {"sheet": sheet, "count": count}[parser.parse_args().command]()
