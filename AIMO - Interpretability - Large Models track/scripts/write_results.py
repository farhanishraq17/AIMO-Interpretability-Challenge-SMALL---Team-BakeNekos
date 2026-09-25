"""Regenerate Results_Large_Track.md from the data mirror and the downloaded Kaggle outputs.

Why a generator: as in the small track, every number in a document must come from a script so it
can be rebuilt for the report. This one rewrites the whole file each run (no hand-kept sections):
label inventory, the official reference-problem table, the panel item list, the L0 hardware
probes, and the health of every downloaded L1 panel run (completions, truncation, boxed answers).
Nothing is graded here: the public tables carry no reference answers.

Run: getting-started/.venv/Scripts/python.exe "AIMO - Interpretability - Large Models track/scripts/write_results.py"
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data" / "hf"
RUNS = ROOT / "runs"
OUT = ROOT / "Results_Large_Track.md"


def table(frame: pd.DataFrame) -> str:
    """Markdown table without `tabulate`, which the pinned venv does not ship."""
    cell = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + " --- |" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def labels() -> str:
    train = pd.read_parquet(HF / "train-main-v2/data/train-00000-of-00001.parquet")
    labelled = train.dropna(subset=["is_robust"])
    grouped = train.groupby(["model_id", "reasoning_effort"])
    frame = pd.DataFrame({
        "rows": grouped.size(),
        "labelled": grouped.is_robust.count(),
        "robust": labelled.groupby(["model_id", "reasoning_effort"]).is_robust.sum().astype(int),
    }).reset_index()
    frame.loc[len(frame)] = ["all", "", frame.rows.sum(), frame.labelled.sum(), frame.robust.sum()]
    frame["robust_share"] = frame.robust / frame.labelled
    frame["constant_acc"] = frame.robust_share.where(frame.robust_share >= 0.5, 1 - frame.robust_share)
    return ("## Official labels: `train-main-v2`\n\n"
            f"{train.problem.nunique()} distinct problems. `constant_acc` is the best constant answer per row "
            "group, the floor any Main-Track method must beat.\n\n" + table(frame))


def reference() -> str:
    full = pd.read_parquet(HF / "aimo-interp-challenge-sample-full/data/validation-00000-of-00001.parquet")
    frame = full.groupby("model_id").agg(
        rows=("problem_id", "size"), problems=("problem_id", "nunique"),
        base_acc=("base_accuracy", "mean"), permuted_acc=("permuted_accuracy", "mean"),
        abs_decay=("absolute_accuracy_decay", "mean"),
    ).reset_index()
    return ("## Official reference problems: `aimo-interp-challenge-sample-full`\n\n"
            "Means over (problem, perturbation family) rows.\n\n" + table(frame))


def items() -> str:
    path = ROOT / "data/panel/items.jsonl"
    if not path.exists():
        return "## Panel items\n\nNot built; run `scripts/build_items.py`."
    frame = pd.read_json(path, lines=True)
    counts = frame.groupby(["source", "family"]).agg(items=("item_id", "size"), completions=("k", "sum")).reset_index()
    return (f"## Panel items (`data/panel/items.jsonl`)\n\n{len(frame)} items, {int(frame.k.sum())} "
            "completions per model and effort.\n\n" + table(counts))


def probes() -> str:
    rows = []
    for path in sorted(RUNS.glob("l0-h100-probe/s*/probe.json")):
        report = json.loads(path.read_text())
        job = json.loads(report["job"])
        gpus = report["nvidia_smi"].strip().splitlines()
        rows.append(dict(shard=job["shard"], account=job["account"], gpus=f"{len(gpus)}x {gpus[0].split(',')[0]}",
                         gpt_oss_120b_mounted="gpt-oss-120b" in report["inputs"], internet=report["internet"] is True))
    body = table(pd.DataFrame(rows)) if rows else "No probe output downloaded."
    return "## L0: what `NvidiaH100` actually allocates\n\n" + body


def panel_runs() -> str:
    rows = []
    for job in sorted({path.parts[-3] for path in RUNS.glob("l1-*/s*/records*.jsonl")}):
        # every shard and every continuation session of the job (one records file per session)
        frame = pd.concat([pd.read_json(path, lines=True) for path in RUNS.glob(f"{job}/s*/records*.jsonl")],
                          ignore_index=True)
        text = frame.content.fillna("")
        for (model, effort), part in frame.groupby(["model", "effort"]):
            answered = text[part.index].str.contains(r"\\boxed\{", regex=True)
            rows.append(dict(job=job, model=model, effort=effort, completions=len(part),
                             errors=int(part.error.notna().sum()),
                             truncated=float((part.done_reason == "length").mean()),
                             boxed=float(answered.mean()), mean_tokens=float(part.eval_count.mean())))
    body = table(pd.DataFrame(rows)) if rows else "No panel output downloaded yet."
    return ("## L1: panel run health\n\n`truncated` = hit `num_predict`; `boxed` = final message contains "
            "`\\boxed{}`.\n\n" + body)


def main() -> None:
    sections = [labels(), reference(), items(), probes(), panel_runs()]
    header = ("# Results: Large Models (Main) Track\n\nGenerated by `scripts/write_results.py`; "
              "do not hand-edit. Narrative and interpretation live in `Progress_Large_Track.md`.\n")
    OUT.write_text("\n\n".join([header, *sections]) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
