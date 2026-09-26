"""Regenerate Results_Large_Track.md from the data mirror and the downloaded Kaggle outputs.

Why a generator: as in the small track, every number in a document must come from a script so it
can be rebuilt for the report. This one rewrites the whole file each run (no hand-kept sections):
label inventory, the official reference-problem table, the panel item list, the L0 hardware
probes, and the health of every downloaded L1 panel run (completions, truncation, boxed answers).
Nothing is graded here: the public tables carry no reference answers.

Run: getting-started/.venv/Scripts/python.exe "AIMO - Interpretability - Large Models track/scripts/write_results.py"
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data" / "hf"
RUNS = ROOT / "runs"
OUT = ROOT / "Results_Large_Track.md"
# The small track's E24a key: an answer only where all 8 Skywork samples agreed (model-derived).
E24A_KEY = ROOT.parent / "data" / "official_check" / "pool_keyed.parquet"
# Official reference rows for the checkpoint each proxy stands in for (aliases in the public table).
OFFICIAL_ALIAS = {"gpt-oss:20b": "gpt-oss-120b:low", "deepseek-r1:8b": "qwen3-8b:low"}

sys.path.insert(0, str(ROOT.parent / "datagen" / "perturb"))
from answers import extract_answer  # noqa: E402  (the small track's grader: last \boxed{} -> int)


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


def panel_answers() -> str:
    """Label-free answer statistics; no official answers exist for these problems."""
    paths = [p for p in RUNS.glob("l1-*/s*/records*.jsonl") if "smoke" not in p.parts[-3]]
    if not paths:
        return "## L1: answer stability\n\nNo panel output downloaded yet."
    runs = pd.concat([pd.read_json(p, lines=True) for p in paths], ignore_index=True)
    runs = runs.drop_duplicates(["model", "effort", "item_id", "sample"])  # a base job and its helper
    runs["answer"] = runs.content.fillna("").map(extract_answer)
    key = {}
    if E24A_KEY.exists():
        pool = pd.read_parquet(E24A_KEY)
        key = {"tm2:" + hashlib.sha1(" ".join(t.split()).encode()).hexdigest()[:12]: int(a)
               for t, a in zip(pool.problem, pool.answer)}
    full = pd.read_parquet(HF / "aimo-interp-challenge-sample-full/data/validation-00000-of-00001.parquet")

    summary, families = [], []
    for (model, effort), part in runs.groupby(["model", "effort"]):
        originals = part[part.family == "original"].groupby("problem_id").answer.agg(list)
        originals = originals[originals.map(len) >= 2]
        # share of samples that give the problem's most common answer (no answer counts as its own)
        consistency = originals.map(lambda a: pd.Series([str(x) for x in a]).value_counts().iloc[0] / len(a))
        # a problem's own answer: kept only when every original sample agrees (the E24a rule)
        unanimous = {pid: a[0] for pid, a in originals.items() if a[0] is not None and len(set(a)) == 1}
        variants = part[(part.family != "original") & part.problem_id.isin(unanimous)].copy()
        variants["stable"] = variants.answer == variants.problem_id.map(unanimous)
        keyed = part[(part.family == "original") & part.problem_id.isin(key)]
        summary.append(dict(
            model=model, effort=effort, problems=len(originals), self_consistency=consistency.mean(),
            unanimous=len(unanimous), variant_problems=variants.problem_id.nunique(),
            variants_scored=len(variants), variant_stability=variants.stable.mean(),
            e24a_keyed=keyed.problem_id.nunique(),
            e24a_accuracy=(keyed.answer == keyed.problem_id.map(key)).mean() if len(keyed) else float("nan")))
        official = full[full.model_id == OFFICIAL_ALIAS.get(model, "")]
        for family, group in variants.groupby("family"):
            ref = official[official.permutation_type == family]
            families.append(dict(model=model, effort=effort, family=family, variants=len(group),
                                 stability=group.stable.mean(), official=OFFICIAL_ALIAS.get(model, "-"),
                                 official_base_acc=ref.base_accuracy.mean() if len(ref) else float("nan"),
                                 official_permuted_acc=ref.permuted_accuracy.mean() if len(ref) else float("nan")))
    return ("## L1: answer stability (label-free)\n\nNo official answers exist for the panel problems, so "
            "nothing here is accuracy against ground truth. `self_consistency`: share of a problem's "
            "original samples giving its most common answer. `variant_stability`: share of variants whose "
            "answer equals the problem's answer, over problems whose original samples all agree. "
            "`e24a_accuracy`: original samples matching the small track's E24a key (unanimous Skywork "
            "samples; model-derived, not official).\n\n" + table(pd.DataFrame(summary))
            + "\n\nPer perturbation family, next to the official rows for the checkpoint the proxy stands in "
            "for (official means over its (problem, family) rows; a different quantity from stability, "
            "shown for orientation only):\n\n" + table(pd.DataFrame(families)))


def main() -> None:
    sections = [labels(), reference(), items(), probes(), panel_runs(), panel_answers()]
    header = ("# Results: Large Models (Main) Track\n\nGenerated by `scripts/write_results.py`; "
              "do not hand-edit. Narrative and interpretation live in `Progress_Large_Track.md`.\n")
    OUT.write_text("\n\n".join([header, *sections]) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
