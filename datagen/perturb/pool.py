"""Olympiad problem pool for self-generated robustness labels (PLAN.md §3.4, feeds E7).

The consolidated training set has one olympiad instance; the private test set is entirely
olympiad. This builds the problem side of the fix: public competition problems with a single
integer answer, taken from Hugging Face datasets pinned to exact revisions (all present in the
local cache, so `--offline` works).

Deliberately excluded:

* `HuggingFaceH4/MATH-500` and anything Hendrycks: the distribution we already have.
* `math-ai/aime26`: a copy of `MathArena/aime_2026`.
* OlympiadBench items that are multimodal, multi-answer, non-numeric, or carry a unit or a
  shared context.
* Any problem whose text already appears in `data/small_track/instances.parquet`.

Nothing here is labelled. Labels come later, from sampling the target model on originals and
variants (E7).

Run:  python datagen/perturb/pool.py --offline
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

PERTURB = Path(__file__).resolve().parent
ROOT = PERTURB.parents[1]
if str(PERTURB.parent) not in sys.path:
    sys.path.insert(0, str(PERTURB.parent))

from ingest_public import problem_key  # noqa: E402  - same key as the training set

OUT = ROOT / "data" / "olympiad_pool"
INSTANCES = ROOT / "data" / "small_track" / "instances.parquet"


@dataclass(frozen=True)
class Source:
    name: str
    tier: str  # rough difficulty: amc < aime < olympiad
    repo: str
    revision: str
    filename: str
    id_col: str
    problem_col: str
    answer_col: str


SOURCES = (
    Source("aime-2024", "aime", "Maxwell-Jia/AIME_2024", "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
           "aime_2024_problems.parquet", "ID", "Problem", "Answer"),
    Source("aime-2026", "aime", "MathArena/aime_2026", "d2de22f3c656b4f56cf8981212186377d1e23bc3",
           "data/train-00000-of-00001.parquet", "problem_idx", "problem", "answer"),
    Source("amc-validation", "amc", "AI-MO/aimo-validation-amc", "69d78a4a2c840e82d69af6bc742bda09005f6316",
           "data/train-00000-of-00001.parquet", "id", "problem", "answer"),
    Source("olympiadbench-en", "olympiad", "Hothan/OlympiadBench", "91184b52131e7fc9455fef848035173aea8cc01a",
           "OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet", "id", "question", "final_answer"),
)


def parse_integer(value: object) -> int | None:
    """A single integer answer, or None.

    Accepts '33', '142.0', "['$2$']" and '5,000'; rejects fractions, tuples, and multi-answers.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("["):
        try:
            items = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(items, (list, tuple)) or len(items) != 1:
            return None
        text = str(items[0]).strip()
    text = text.strip("$ ")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", text):
        text = text.replace(",", "")
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    return int(text) if re.fullmatch(r"-?\d+", text) else None


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().isin(["", "None", "nan"])


# Some sources were serialised without escaping backslashes, so "\frac" arrived as a form
# feed followed by "rac" (seen in Maxwell-Jia/AIME_2024). Restore the commands.
_CONTROL_ESCAPES = {"\x07": "\\a", "\x08": "\\b", "\x0b": "\\v", "\x0c": "\\f"}
_PATTERN_REPAIRS = (
    (re.compile(r"\t(?=imes|heta|au|an\b|ext|riangle|frac|op)"), "\\\\t"),
    (re.compile(r"\r(?=ight|floor|ceil|ho\b|angle)"), "\\\\r"),
    # The opposite failure, double escaping: "\\frac" where "\frac" was meant. A real LaTeX
    # line break ("\\") is followed by whitespace, not a command name, so it is left alone.
    (
        re.compile(
            r"\\\\(?=(?:d?frac|sqrt|cdot|times|left|right|mathcal|mathbb|text|leq?|geq?|pi|angle"
            r"|triangle|overline|binom|log|sin|cos|tan)(?![A-Za-z]))"
        ),
        "\\\\",
    ),
)


def repair_escapes(text: str) -> tuple[str, int]:
    """Undo unescaped- and double-escaped-backslash corruption; returns text and repair count."""
    repairs = 0
    for char, command in _CONTROL_ESCAPES.items():
        repairs += text.count(char)
        text = text.replace(char, command)
    for pattern, replacement in _PATTERN_REPAIRS:
        text, count = pattern.subn(replacement, text)
        repairs += count
    return text, repairs


def load_source(source: Source, offline: bool) -> tuple[pd.DataFrame, dict]:
    path = hf_hub_download(
        repo_id=source.repo,
        repo_type="dataset",
        revision=source.revision,
        filename=source.filename,
        local_files_only=offline,
    )
    frame = pd.read_parquet(path)
    stats = {"raw": len(frame), "escape_repairs": 0}
    if source.repo == "Hothan/OlympiadBench":
        keep = (
            (frame["modality"] == "Text-only")
            & (frame["is_multiple_answer"].astype(str) == "False")
            & (frame["answer_type"] == "Numerical")
            & _is_blank(frame["context"])
            & _is_blank(frame["unit"])
        )
        frame = frame[keep]
        stats["after_olympiadbench_filters"] = len(frame)

    rows = []
    for _, row in frame.iterrows():
        problem, repairs = repair_escapes(str(row[source.problem_col]).strip())
        stats["escape_repairs"] += repairs
        rows.append({
            "pool_id": f"{source.name}:{row[source.id_col]}",
            "source": source.name,
            "tier": source.tier,
            "source_repo": source.repo,
            "source_revision": source.revision,
            "problem": problem,
            "answer": parse_integer(row[source.answer_col]),
            "problem_key": problem_key(problem),
        })
    out = pd.DataFrame(rows)
    # AI-MO/aimo-validation-amc reuses `id` across contests: 34 ids named two different problems,
    # which made eligibility and labels ambiguous (Finding 29). Only colliding ids change, so
    # records keyed by every other pool_id stay valid.
    clash = out["pool_id"].duplicated(keep=False)
    out.loc[clash, "pool_id"] = out.loc[clash, "pool_id"] + ":" + out.loc[clash, "problem_key"].str[:8]
    stats["id_collisions_disambiguated"] = int(clash.sum())
    stats["integer_answer"] = int(out["answer"].notna().sum())
    return out[out["answer"].notna()].astype({"answer": "int64"}), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="read only from the local HF cache")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    frames, summary = [], {"sources": {}}
    for source in SOURCES:
        frame, stats = load_source(source, args.offline)
        frames.append(frame)
        summary["sources"][source.name] = {"repo": source.repo, "revision": source.revision, **stats}
        print(f"  {source.name:18s} {stats}")

    pool = pd.concat(frames, ignore_index=True)
    before = len(pool)
    pool = pool.drop_duplicates("problem_key", keep="first")
    duplicates = before - len(pool)

    overlap = 0
    if INSTANCES.is_file():
        known = set(pd.read_parquet(INSTANCES)["problem_key"])
        overlap = int(pool["problem_key"].isin(known).sum())
        pool = pool[~pool["problem_key"].isin(known)]
    pool = pool.sort_values("pool_id").reset_index(drop=True)

    summary.update(
        duplicates_removed=duplicates,
        training_overlap_removed=overlap,
        pool_size=len(pool),
        per_tier=pool["tier"].value_counts().to_dict(),
        per_source=pool["source"].value_counts().to_dict(),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(args.out_dir / "pool.parquet", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\npool: {len(pool)} problems ({duplicates} duplicates, {overlap} training overlaps removed)")
    print(f"per tier: {summary['per_tier']}")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
