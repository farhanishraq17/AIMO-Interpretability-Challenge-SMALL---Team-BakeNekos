"""E61: build the floor candidates, and score every constant policy on the official rows.

E61's premise is that the shipped `e27f` constants are worse than a one-line answer on the only
labelled olympiad data we have, and that the per-model split they encode should not exist
(Finding 38: the label is a property of the problem; E43 re-read: every model is majority
robust). This builder ships the floor and prints the table that justifies it, so the claim is
regenerated rather than quoted.

* `e61a-always-true`  `True` for all four small-track models, no model loaded. Gate A floor.
* `e61b-uncertainty-baseline` is **not** built here. It is the organizers' own
  `solutions/uncertainty-profiling`, packaged unmodified by the upstream builder, so that what
  we upload is their artifact and not our transcription of it:
      cd getting-started; python -m uv run scripts/build.py solutions --small

Nothing is fitted, so nothing needs the pinned venv to serialise - but run it there anyway, as
every other builder does, so the ZIPs are produced under one toolchain.

Run:  python submission/make_e61.py          # -> dist/solution-e61a-always-true-small.zip
      python submission/make_e61.py --score  # table only, no ZIP
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SUBMISSION = Path(__file__).resolve().parent
ROOT = SUBMISSION.parent
sys.path.insert(0, str(ROOT / "getting-started" / "scripts"))

from build import archive  # noqa: E402

DIST = ROOT / "dist"
INSTANCES = ROOT / "data" / "small_track" / "instances.parquet"

QWEN = "Qwen/Qwen3.5-4B"
SKYWORK = "Skywork/Skywork-OR1-Math-7B"
OLMO = "allenai/Olmo-3-7B-Think"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
SMALL_TRACK = (QWEN, SKYWORK, OLMO, DEEPSEEK)

# The policies worth comparing. Each maps a model_id to a constant prediction; `e27f` is the
# shipped incumbent, kept here so the comparison is regenerated and not transcribed.
POLICIES = {
    "e27f constants (shipped)": {QWEN: True, DEEPSEEK: True, SKYWORK: False, OLMO: False},
    "e61a always-True": dict.fromkeys(SMALL_TRACK, True),
    "always-False": dict.fromkeys(SMALL_TRACK, False),
}


def official_rows() -> pd.DataFrame:
    """The labelled `train-main-v2` small-track rows, as `ingest_public.py` wrote them.

    The official split decides any pair it covers (Finding 44), so these rows carry the
    organizers' own three-valued rule keyed on `max_drop`; band cases are unlabelled upstream
    and never appear here.
    """
    if not INSTANCES.exists():
        raise SystemExit(f"missing {INSTANCES}; run datagen/ingest_public.py first")
    d = pd.read_parquet(INSTANCES)
    d = d[d.model_id.isin(SMALL_TRACK)]
    return d[d.source_repos.astype(str).str.contains("train-main-v2")]


def score(rows: pd.DataFrame, policy: dict[str, bool]) -> tuple[int, float, float]:
    """Accuracy and balanced accuracy of a constant policy over `rows`."""
    pred = rows.model_id.map(policy)
    truth = rows.is_robust.astype(bool)
    correct = int((pred == truth).sum())
    tpr = float((pred & truth).sum()) / max(int(truth.sum()), 1)
    tnr = float((~pred & ~truth).sum()) / max(int((~truth).sum()), 1)
    return correct, correct / len(rows), 0.5 * (tpr + tnr)


def report(rows: pd.DataFrame) -> None:
    n, robust = len(rows), int(rows.is_robust.sum())
    print(f"official small-track rows: {n}  robust {robust}  spurious {n - robust}  "
          f"prior {robust / n:.3f}")
    print("\nper model:")
    for model in SMALL_TRACK:
        s = rows[rows.model_id == model]
        if len(s):
            r = int(s.is_robust.sum())
            print(f"  {model:42s} {r:3d}/{len(s):<3d} robust = {r / len(s):.3f}")
    print(f"\n{'policy':28s} {'correct':>10} {'accuracy':>9} {'balanced':>9}")
    for name, policy in POLICIES.items():
        correct, acc, bal = score(rows, policy)
        print(f"{name:28s} {correct:>6}/{n:<3d} {acc:>9.3f} {bal:>9.3f}")


def build() -> None:
    src = (SUBMISSION / "e61a-always-true" / "solution.py").read_bytes()
    entries = {"solution.py": src, "small.txt": b""}
    out = archive(DIST / "solution-e61a-always-true-small.zip", entries)
    print(f"\nwrote {out.name} ({out.stat().st_size / 1e3:.1f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", action="store_true", help="print the table only, build nothing")
    args = ap.parse_args()
    report(official_rows())
    if not args.score:
        build()


if __name__ == "__main__":
    main()
