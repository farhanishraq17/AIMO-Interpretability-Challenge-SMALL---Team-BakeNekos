"""E58: does a full-budget answer-property labeller reproduce the official labels on the official rows?

`protocols/E58_PROTOCOL.md` fixed everything here before any E58 record existed: the variants
(`datagen/perturb/e58_variants.py`), the grading, the label rule and the gate.

**Grading.** A base or consistent-premise sample is correct iff it finished within the cap and its
answer is the reference answer. A contradiction sample is correct iff E52's grader calls it `detected`
(finished, no integer, an ending that says there is no valid answer; audited 30 of 30 in E52).

**Per (model, problem).** Eligible iff all base samples are correct. Each family's accuracy pools its
two variants. `max_drop = 1 - min(family accuracies)`, and the official rule labels the row robust iff
<= 0.10, spurious iff >= 0.25, band between. A row is labelled iff eligible, every family has at least
`MIN_FAMILY_SAMPLES` graded samples, and it is not band.

**Gate.** At least `MIN_LABELLED` labelled rows, and Cohen's kappa against the official labels at least
`MIN_KAPPA` with a problem-bootstrap 95% interval above 0. The official labels are read here and only here.

**Exploratory, added after the gate was read (18 Oct) and never part of it:** how many contradiction
samples mention the clash in their stored ending, and, in the organizers' public sample, how closely each
family's accuracy follows the original's (`exploratory`).

Writes the `E58: self-labelling gate on the official rows` section of RESULTS.md and
runs/e58-analysis/result.json.

E59 (`protocols/E59_PROTOCOL.md`) is the same labeller with a prompt that can decline; `--experiment e59`
grades and gates its runs with this same code and writes its own section and runs/e59-analysis/.

Run:  python train/eval_e58.py   (or --experiment e59)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from eval_e52 import grade, problem_bootstrap  # noqa: E402
from results_io import upsert_section  # noqa: E402

MODELS = {"qwen": "Qwen/Qwen3.5-4B", "deepseek": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
          "skywork": "Skywork/Skywork-OR1-Math-7B", "olmo": "allenai/Olmo-3-7B-Think"}
CONSTANTS = {"qwen": True, "deepseek": True, "skywork": False, "olmo": False}  # e27f
RECORDS = "runs/{experiment}-{short}-*/{experiment}-{short}-*/{phase}/shard_*/records.jsonl"
EXPERIMENT = "e58"  # set by --experiment
HEADINGS = {"e58": "E58: self-labelling gate on the official rows",
            "e59": "E59: the same gate with a prompt that can decline"}
ROBUST, SPURIOUS = 0.10, 0.25  # the official max_drop rule
MIN_FAMILY_SAMPLES = 4
MIN_LABELLED, MIN_KAPPA, SEED = 18, 0.30, 58


def family(category: str) -> str:
    return "no_solution" if category.startswith("contradict") else "consistent"


def correct(record: dict, cap: int) -> list[bool]:
    return [a == record["expected_answer"] and t < cap for a, t in zip(record["answers"], record["completion_tokens"])]


def label_row(base: dict | None, variants: list[dict], cap: int) -> dict:
    """One (model, problem) row from its base record and variant records (see the module docstring)."""
    row = {"status": "not_sampled", "label": None, "max_drop": None, "deciding": None}
    if base is None:
        return row
    row["eligible"] = bool(base["n_samples"]) and all(correct(base, cap))
    samples = {"no_solution": [], "consistent": []}
    outcomes = []
    for record in variants:
        if family(record["category"]) == "consistent":
            samples["consistent"] += correct(record, cap)
            continue
        for tail, tokens, answer in zip(record["sample_tails"], record["completion_tokens"], record["answers"]):
            outcome = grade(tail, tokens, cap)
            samples["no_solution"].append(outcome == "detected")
            outcomes.append("kept_original" if outcome == "answered" and answer == record["expected_answer"] else outcome)
    row.update({f"n_{f}": len(s) for f, s in samples.items()}, **{o: outcomes.count(o) for o in
               ("detected", "answered", "kept_original", "truncated", "other")})
    if not row["eligible"]:
        row["status"] = "ineligible"
    elif min(len(s) for s in samples.values()) < MIN_FAMILY_SAMPLES:
        row["status"] = "incomplete"
    else:
        accuracy = {f: float(np.mean(s)) for f, s in samples.items()}
        row.update({f"acc_{f}": a for f, a in accuracy.items()}, max_drop=1 - min(accuracy.values()),
                   deciding=min(accuracy, key=accuracy.get))
        row["label"] = "robust" if row["max_drop"] <= ROBUST + 1e-9 else "spurious" if row["max_drop"] >= SPURIOUS - 1e-9 else "band"
        row["status"] = "labelled" if row["label"] != "band" else "band"
    return row


def load(short: str, phase: str) -> list[dict]:
    paths = sorted(ROOT.glob(RECORDS.format(experiment=EXPERIMENT, short=short, phase=phase)))
    rows = [json.loads(line) for p in paths for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return list({r["item_id"]: r for r in rows}.values())  # a resumed shard may repeat an item


def run_cap(short: str) -> int:
    paths = ROOT.glob(RECORDS.format(experiment=EXPERIMENT, short=short, phase="base").replace("records.jsonl", "metadata.json"))
    caps = {json.loads(p.read_text(encoding="utf-8"))["args"]["max_new_tokens"] for p in paths}
    assert len(caps) == 1, f"{short}: runs disagree on the cap: {caps}"
    return caps.pop()


def kappa(frame: pd.DataFrame) -> float:
    if frame.official.nunique() == 1 and frame.self_robust.nunique() == 1:
        raise ValueError("kappa undefined: one class in both")
    return float(cohen_kappa_score(frame.official, frame.self_robust))


# Exploratory: the ending names the clash (the added premise, a contradiction, a suspected error).
NOTICE = re.compile(r"contradict|inconsisten|impossible|not (?:odd|even)|isn.t (?:odd|even)|but (?:the|it) "
                    r"(?:says|states|problem says)|which is even|which is odd|typo|mistake in the problem|"
                    r"cannot be (?:odd|even|greater)", re.IGNORECASE)


def exploratory() -> tuple[list[str], dict]:
    """Post-hoc reading of the failed gate; see the module docstring."""
    from scipy.stats import spearmanr
    from eval_label_mechanics import SAMPLE
    from huggingface_hub import hf_hub_download

    lines = ["**Exploratory (after the gate was read).** Contradiction samples whose stored ending mentions the clash:", "",
             "| model | answered: mention the clash | truncated: mention the clash |", "| --- | --- | --- |"]
    notice = {}
    for short in MODELS:
        cap, counts = run_cap(short), {"answered": [0, 0], "truncated": [0, 0]}
        for record in load(short, "variants"):
            if family(record["category"]) != "no_solution":
                continue
            for tail, tokens in zip(record["sample_tails"], record["completion_tokens"]):
                outcome = grade(tail, tokens, cap)
                if outcome in counts:
                    counts[outcome][0] += 1
                    counts[outcome][1] += bool(NOTICE.search(tail))
        notice[short] = counts
        lines.append(f"| {short} | {counts['answered'][1]} of {counts['answered'][0]} | {counts['truncated'][1]} of {counts['truncated'][0]} |")
    if EXPERIMENT != "e58":  # the organizers' sample is read once, in E58's section
        return lines, {"notice": notice}
    sample = pd.read_parquet(hf_hub_download(SAMPLE[0], SAMPLE[1], repo_type="dataset", revision=SAMPLE[2]))
    lines += ["", "Organizers' public sample: how closely variant accuracy follows the original's, per family "
              "(an answer-preserving family should follow it closely; one graded on detection need not):", "",
              "| family | rows | Spearman(base, variant) | variant accuracy where base <= 0.2 |", "| --- | --- | --- | --- |"]
    coupling = {}
    for name, group in sample.groupby("permutation_type"):
        if len(group) < 10:
            continue
        low = group[group.base_accuracy <= 0.2]
        coupling[name] = {"rows": len(group), "spearman": float(spearmanr(group.base_accuracy, group.permuted_accuracy).statistic),
                          "low_base_rows": len(low), "low_base_variant_accuracy": float(low.permuted_accuracy.mean())}
        c = coupling[name]
        lines.append(f"| {name} | {c['rows']} | {c['spearman']:+.2f} | {c['low_base_variant_accuracy']:.2f} (n = {c['low_base_rows']}) |")
    no_solution = sample[sample.permutation_type == "expert_no_solution"].groupby("model_id").permuted_accuracy.mean()
    lines += ["", "`expert_no_solution` accuracy by configuration in that sample: " +
              ", ".join(f"{m} {a:.2f}" for m, a in no_solution.sort_values().items()) + "."]
    return lines, {"notice": notice, "coupling": coupling, "no_solution_accuracy": no_solution.to_dict()}


def main() -> None:
    global EXPERIMENT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=sorted(HEADINGS), default="e58")
    EXPERIMENT = parser.parse_args().experiment
    out_dir = ROOT / "runs" / f"{EXPERIMENT}-analysis"
    targets = pd.read_parquet(ROOT / "data" / "e58" / "rows.parquet")
    official = pd.read_parquet(ROOT / "data" / "official_check" / "official_labels.parquet")
    rows = []
    for short, model_id in MODELS.items():
        base = {r["pool_id"]: r for r in load(short, "base")}
        if not base:
            raise SystemExit(f"no {EXPERIMENT} base records for {short}; download every {EXPERIMENT} kernel first")
        cap = run_cap(short)
        variants = load(short, "variants")
        for target in targets[targets.model_id == model_id].itertuples(index=False):
            row = label_row(base.get(target.pool_id), [v for v in variants if v["pool_id"] == target.pool_id], cap)
            rows.append({"model": short, "pool_id": target.pool_id, "problem_key": target.problem_key, "cap": cap, **row})
    frame = pd.DataFrame(rows).merge(official.rename(columns={"is_robust": "official"})[["model_id", "problem_key", "official"]]
                                     .assign(model=lambda d: d.model_id.map({v: k for k, v in MODELS.items()})),
                                     on=["model", "problem_key"], how="left", validate="one_to_one")
    assert frame.official.notna().all(), "an E58 row has no official label"
    labelled = frame[frame.status == "labelled"].assign(self_robust=lambda d: d.label == "robust",
                                                       constant=lambda d: d.model.map(CONSTANTS))
    try:
        k = kappa(labelled)
        low, high = problem_bootstrap(labelled, kappa, seed=SEED)
    except ValueError:
        k, low, high = float("nan"), float("nan"), float("nan")
    passes = bool(len(labelled) >= MIN_LABELLED and k >= MIN_KAPPA and low > 0)

    per_model = frame.groupby("model", sort=False).agg(
        rows=("pool_id", "size"), sampled=("status", lambda s: int((s != "not_sampled").sum())),
        eligible=("eligible", lambda s: int(s.fillna(False).sum())), labelled=("status", lambda s: int((s == "labelled").sum())),
        self_robust=("label", lambda s: int((s == "robust").sum())), self_spurious=("label", lambda s: int((s == "spurious").sum())),
        band=("label", lambda s: int((s == "band").sum())), official_robust=("official", "sum"),
        detected=("detected", "sum"), answered=("answered", "sum"), kept_original=("kept_original", "sum"),
        truncated=("truncated", "sum"), other=("other", "sum"), cap=("cap", "first")).reset_index()
    agreement = lambda column: float((labelled[column] == labelled.official).mean()) if len(labelled) else float("nan")  # noqa: E731
    result = {"rows": len(frame), "labelled": len(labelled), "kappa": k, "kappa_ci": [low, high], "passes": passes,
              "self_label_accuracy": agreement("self_robust"), "constants_accuracy_same_rows": agreement("constant"),
              "deciding_family": labelled.deciding.value_counts().to_dict(),
              "confusion": pd.crosstab(labelled.official, labelled.self_robust).to_dict() if len(labelled) else {},
              "per_model": per_model.to_dict("records"), "per_row": frame.drop(columns="model_id").to_dict("records")}
    explore_lines, result["exploratory"] = exploratory()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    table = ["| model | cap | rows | sampled | eligible | labelled | self robust / spurious / band | official robust "
             "| contradiction samples: detected / answered (kept original) / truncated / other |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in per_model.itertuples(index=False):
        table.append(f"| {m.model} | {m.cap:,} | {m.rows} | {m.sampled} | {m.eligible} | {m.labelled} | {m.self_robust} / "
                     f"{m.self_spurious} / {m.band} | {int(m.official_robust)} | {int(m.detected)} / {int(m.answered + m.kept_original)} "
                     f"({int(m.kept_original)}) / {int(m.truncated)} / {int(m.other)} |")
    body = "\n".join([
        "Answer-property variants (`datagen/perturb/e58_variants.py`) on the 27 official rows with a reference answer, "
        "sampled with the organizers' decoding at each model's T4 cap "
        f"(`protocols/{EXPERIMENT.upper()}_PROTOCOL.md`" + ("; the prompt can decline" if EXPERIMENT == "e59" else "") + ").",
        "", *table, "",
        f"**Gate: {'passes' if passes else 'fails'}.** {len(labelled)} of {len(frame)} rows labelled (needs {MIN_LABELLED}); "
        f"Cohen's kappa against the official labels {k:.3f}, 95% problem-bootstrap CI [{low:.3f}, {high:.3f}] "
        f"(needs >= {MIN_KAPPA} with the lower bound above 0).", "",
        f"On the labelled rows, self-labels agree with the official labels on {result['self_label_accuracy']:.3f}; "
        f"`e27f`'s constants agree on {result['constants_accuracy_same_rows']:.3f}. "
        f"Family that set `max_drop`: {result['deciding_family']}.", "", *explore_lines,
    ])
    upsert_section(ROOT / "RESULTS.md", HEADINGS[EXPERIMENT], body)
    print(body)


if __name__ == "__main__":
    main()
