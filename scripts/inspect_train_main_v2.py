"""Summarize `aimo-interp/train-main-v2`, the organizers' own labelled train split.

Why this exists: this table was published on 14 September 2026 and reworked on 16 September,
is linked from neither the competition website nor the `getting-started` README, and is the
only public source of labels for three of the four small-track models. Every number quoted in
PROGRESS Findings 29-31 is printed here so it can be regenerated for the report.

Two properties matter more than the counts and are checked explicitly:

* The label rule is not the one `ingest_public.py` applies. Here `is_robust` is three-valued
  and keyed on `max_drop` (robust <= 0.10, spurious >= 0.25, null between), not on
  `absolute_accuracy_decay`.
* `problem` is not a safe grouping key. Several rows are the same problem template with one
  constant changed, so an exact-text or SHA-1 grouping would split a template across folds.
  The clustering below reports how many templates the distinct strings collapse to.

Run:  uv run python ../scripts/inspect_train_main_v2.py
"""

from __future__ import annotations

import argparse
import difflib
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]

DATASET = "aimo-interp/train-main-v2"
# Pinned so the counts below are reproducible; the table was rewritten once already.
REVISION = "23791d9b120132087877b0459f953d5959fc8455"
PARQUET = "data/train-00000-of-00001.parquet"

SMALL_TRACK = {
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}

# The organizers' published rule, restated so the data can contradict it if it ever changes.
ROBUST_AT_OR_BELOW = 0.10
SPURIOUS_AT_OR_ABOVE = 0.25

# Two problems count as one template above this similarity. 0.90 separates the
# constant-swapped variants from genuinely different problems in this table.
TEMPLATE_SIMILARITY = 0.90


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def expected_label(max_drop: float) -> bool | None:
    if max_drop <= ROBUST_AT_OR_BELOW:
        return True
    if max_drop >= SPURIOUS_AT_OR_ABOVE:
        return False
    return None


def cluster_templates(problems: list[str]) -> dict[str, int]:
    """Union-find over pairwise similarity; returns problem text -> template id."""
    parent = list(range(len(problems)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(problems)):
        for j in range(i + 1, len(problems)):
            if difflib.SequenceMatcher(None, problems[i], problems[j]).ratio() > TEMPLATE_SIMILARITY:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    return {problems[i]: find(i) for i in range(len(problems))}


def tally(rows: list[dict]) -> tuple[int, int, int]:
    robust = sum(1 for r in rows if r["is_robust"] is True)
    spurious = sum(1 for r in rows if r["is_robust"] is False)
    unlabelled = sum(1 for r in rows if r["is_robust"] is None)
    return robust, spurious, unlabelled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=REVISION, help="dataset revision to pin")
    args = parser.parse_args()

    path = hf_hub_download(
        repo_id=DATASET, filename=PARQUET, repo_type="dataset", revision=args.revision
    )
    rows = pq.read_table(path).to_pylist()

    print(f"{DATASET} @ {args.revision[:12]}")
    print(f"rows: {len(rows)}\n")

    print("per configuration (compare against the dataset card's own table):")
    print(f"  {'model_id':38s} {'effort':8s}  {'n':>3s} {'rob':>4s} {'spur':>5s} {'null':>5s}")
    configs = sorted({(r["model_id"], r["reasoning_effort"]) for r in rows})
    for model_id, effort in configs:
        subset = [r for r in rows if r["model_id"] == model_id and r["reasoning_effort"] == effort]
        robust, spurious, unlabelled = tally(subset)
        print(f"  {model_id:38s} {effort:8s}  {len(subset):3d} {robust:4d} {spurious:5d} {unlabelled:5d}")

    robust, spurious, unlabelled = tally(rows)
    print(f"\nall rows:   robust={robust} spurious={spurious} null={unlabelled}"
          f" -> {robust / (robust + spurious):.4f} robust among labelled")

    small = [r for r in rows if r["model_id"] in SMALL_TRACK]
    robust, spurious, unlabelled = tally(small)
    print(f"small track: rows={len(small)} robust={robust} spurious={spurious} null={unlabelled}"
          f" -> {robust / (robust + spurious):.4f} robust among labelled")

    print("\nlabel rule (robust <= {:.2f}, spurious >= {:.2f}):".format(
        ROBUST_AT_OR_BELOW, SPURIOUS_AT_OR_ABOVE))
    violations = [r for r in rows if expected_label(r["max_drop"]) is not r["is_robust"]]
    print(f"  violations: {len(violations)}")
    for row in violations[:5]:
        print(f"    is_robust={row['is_robust']} max_drop={row['max_drop']}")
    band = sorted(r["max_drop"] for r in rows if r["is_robust"] is None)
    print(f"  unlabelled band: {band}")

    problems = sorted({r["problem"] for r in rows})
    templates = cluster_templates(problems)
    distinct_templates = len(set(templates.values()))
    print(f"\ngrouping: {len(problems)} distinct problem strings"
          f" -> {distinct_templates} templates at similarity > {TEMPLATE_SIMILARITY}")
    print("  (the dataset card says 28 problems; grouping on raw text would split templates)")

    members: dict[int, list[str]] = defaultdict(list)
    for text, template_id in templates.items():
        members[template_id].append(text)
    for texts in sorted(members.values(), key=len, reverse=True):
        if len(texts) < 2:
            continue
        a, b = texts[0], texts[1]
        diff = [
            line for line in difflib.unified_diff(a.split(), b.split(), lineterm="", n=0)
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]
        print(f"  template with {len(texts)} variants, first pair differs by: {' '.join(diff[:6])}")

    rows_per_template = Counter(templates[r["problem"]] for r in rows)
    print(f"  rows per template: max={max(rows_per_template.values())}"
          f" median={sorted(rows_per_template.values())[len(rows_per_template) // 2]}")

    covered = Counter(r["model_id"] for r in rows if r["model_id"] in SMALL_TRACK)
    print(f"\nsmall-track coverage:")
    for model_id in sorted(SMALL_TRACK):
        labelled = sum(
            1 for r in rows if r["model_id"] == model_id and r["is_robust"] is not None
        )
        print(f"  {model_id:38s} rows={covered.get(model_id, 0):3d} labelled={labelled:3d}")

    compare_local_prevalence()


def compare_local_prevalence() -> None:
    """Does the label rule explain why our set is spurious-heavy and theirs robust-heavy?

    Re-labels our own DeepSeek instances under the train-main-v2 thresholds. If the gap were a
    labelling artefact the two prevalences would converge; they do not, which leaves the
    problem population (Hendrycks MATH vs olympiad) as the explanation.
    """
    instances = ROOT / "data" / "small_track" / "instances.parquet"
    if not instances.exists():
        print("\n(skipping local comparison: run datagen/ingest_public.py first)")
        return

    rows = [
        r for r in pq.read_table(instances).to_pylist()
        if r["model_id"].startswith("deepseek")
    ]
    decays = [r["max_absolute_accuracy_decay"] for r in rows if r["max_absolute_accuracy_decay"] is not None]
    ours = sum(1 for d in decays if d <= 0) / len(decays)
    labelled = [d for d in decays if d <= ROBUST_AT_OR_BELOW or d >= SPURIOUS_AT_OR_ABOVE]
    theirs = sum(1 for d in labelled if d <= ROBUST_AT_OR_BELOW) / len(labelled)

    print(f"\nour DeepSeek instances re-labelled under the train-main-v2 rule (n={len(decays)}):")
    print(f"  robust under ours   (decay <= 0)                : {ours:.3f}")
    print(f"  robust under theirs (<= 0.10, band dropped n={len(labelled)}): {theirs:.3f}")
    print(f"  the rule shifts prevalence by {theirs - ours:+.3f}, so it does NOT explain the")
    print(f"  gap to train-main-v2's 0.706 robust; the problem population does.")


if __name__ == "__main__":
    main()
