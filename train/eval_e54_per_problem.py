"""E54: every signal we hold, scored as an absolute per-problem rule against the constants.

Since 15 Oct the organizers call `are_robust` once per problem (Finding 79), so a method must decide
each problem alone: a score and a fixed threshold. Every signal in this repository was only ever
scored as a within-batch ranking. This scores three of them as per-problem rules on five labelled
populations, under `protocols/E54_PROTOCOL.md`, where the populations, rules, thresholds, metric
and qualification were fixed before any number below was computed.

* S1 familiarity: robust iff mean token NLL is below the scoring model's median on the reference set
  (E52's 75 unlabelled ReliableMath AMC problems). The E35 populations use the `Qwen3-0.6B` proxy of
  the review, whose reference NLLs are computed here with the same function.
* S2 length: robust iff the statement is shorter than the reference median, in characters.
* S3 budget path: E37's frozen predictor (trained on MATH-Perturb), robust iff its probability is at
  least the out-of-fold threshold E36 chose on that training data and recorded (amendment 1).

Metric: pooled balanced accuracy with a 95% problem-bootstrap interval (2,000 draws, seed 54). Any
constant scores 0.5 within a model. A signal qualifies only with intervals above 0.5 on two
populations and no point below 0.5 anywhere.

Writes the `E54: per-problem rules against the constants` section of RESULTS.md and
runs/e54/result.json.

Run:  python train/eval_e54_per_problem.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for path in ("train", "scripts", "datagen"):
    sys.path.insert(0, str(ROOT / path))

from eval_e35_external import derive_labels, read_shards  # noqa: E402
from eval_e36_budget_path import build_examples, feature_columns, make_model  # noqa: E402
from eval_e37_gsm_confirmation import build_gsm_examples  # noqa: E402
from eval_e37_official import SMALL_MODELS, read_feature_records  # noqa: E402
from eval_familiarity import DEEPSEEK, INSTANCES, load_scores  # noqa: E402
from eval_problem_level import load_rows  # noqa: E402
from results_io import upsert_section  # noqa: E402
from review_generalization import PROXY, proxy_scores  # noqa: E402

REFERENCE = ROOT / "data" / "reliablemath" / "pool.parquet"
E52_FAMILIARITY = "runs/e52-features-*/e52-features/familiarity-{short}/shard_*/records.jsonl"
SHORT = {"Qwen/Qwen3.5-4B": "qwen", "Skywork/Skywork-OR1-Math-7B": "skywork",
         "allenai/Olmo-3-7B-Think": "olmo", DEEPSEEK: "deepseek"}
CONSTANTS = {"Qwen/Qwen3.5-4B": True, DEEPSEEK: True, "Skywork/Skywork-OR1-Math-7B": False,
             "allenai/Olmo-3-7B-Think": False}
MATH_PERTURB = ROOT / "runs" / "e35-external" / "math_perturb"
PUTNAM = ROOT / "runs" / "e35-external" / "putnam_axiom"
GSM = ROOT / "runs" / "e36-budget-path" / "gsm_plus"
E36_RESULT = ROOT / "runs" / "e36-budget-path" / "result.json"
# The feature runs E49 scored (Finding 74): rows 6-7 of DeepSeek come from the pre-registered fallback.
E49_RUNS = ["e49-qwen-Mashrafi", "e49-skywork-Nasif", "e49-olmo-r0-DIDHITI", "e49-olmo-r3-SALAM", "e49-olmo-r6-Labib",
            "e49-deepseek-r0-Nasif", "e49-deepseek-r2-Nasif", "e49-deepseek-r4-Samin", "e49-deepseek-r6-fallback",
            "e49-deepseek-r8-DIDHITI"]
PROXY_REFERENCE = ROOT / "runs" / "e54" / "proxy_reference.jsonl"
BOOT, SEED = 2_000, 54


def reference_nll() -> dict[str, float]:
    """Median mean NLL on the reference set, per scoring model (ours from E52, the proxy here)."""
    keys = set(pd.read_parquet(REFERENCE)["problem_key"])
    medians = {}
    for model, short in SHORT.items():
        rows = [json.loads(line) for path in ROOT.glob(E52_FAMILIARITY.format(short=short))
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        values = [r["mean_nll"] for r in rows if r["problem_key"] in keys]
        assert len(values) == len(keys), f"{model}: {len(values)} of {len(keys)} reference NLLs"
        medians[model] = float(np.median(values))
    medians[PROXY] = float(np.median(proxy_reference()))
    return medians


def proxy_reference() -> list[float]:
    """The proxy's mean NLL on the reference set, computed once with E23's function and cached."""
    pool = pd.read_parquet(REFERENCE)
    if not PROXY_REFERENCE.is_file():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModelForCausalLM

        from extract_familiarity import familiarity
        from tokenizer_check import load_checked_tokenizer

        tokenizer, _ = load_checked_tokenizer(PROXY, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(PROXY, dtype=torch.float32, local_files_only=True).eval()
        PROXY_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        with PROXY_REFERENCE.open("w", encoding="utf-8") as handle:
            for row in pool.itertuples():
                handle.write(json.dumps({"key": row.problem_key, **familiarity(model, tokenizer, row.problem)}) + "\n")
    values = [json.loads(line)["mean_nll"] for line in PROXY_REFERENCE.read_text(encoding="utf-8").splitlines()]
    assert len(values) == len(pool)
    return values


def balanced(frame: pd.DataFrame, decision: str) -> dict:
    """Pooled balanced accuracy with a problem-bootstrap 95% interval."""
    y, d = frame["is_robust"].astype(bool).to_numpy(), frame[decision].astype(bool).to_numpy()
    groups = frame["group"].to_numpy(str)
    ids = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in ids}
    rng, values = np.random.default_rng(SEED), []
    for _ in range(BOOT):
        pick = np.concatenate([index[g] for g in rng.choice(ids, len(ids))])
        if np.unique(y[pick]).size == 2:
            values.append(balanced_accuracy_score(y[pick], d[pick]))
    low, high = np.percentile(values, [2.5, 97.5])
    return {"n": int(len(y)), "problems": int(len(ids)), "robust_share": float(y.mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, d)), "ci": [float(low), float(high)],
            "accuracy": float(accuracy_score(y, d))}


def populations(medians: dict[str, float], length_median: float) -> dict[str, dict[str, pd.DataFrame]]:
    """Per population, a frame per signal with columns is_robust, group and the decision."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    nll = load_scores()[["problem_key", "scored_model_id", "mean_nll"]]

    rows = load_rows()
    rows = rows[rows["model_id"].isin(SHORT)].merge(
        nll, left_on=["problem_key", "model_id"], right_on=["problem_key", "scored_model_id"], validate="one_to_one")
    rows["group"] = rows["template"]
    rows["S1"] = rows["mean_nll"] < rows["model_id"].map(medians)
    rows["S2"] = rows["problem"].str.len() < length_median
    rows["constants"] = rows["model_id"].map(CONSTANTS)
    out["P1 official rows"] = {"S1": rows, "S2": rows, "constants": rows}

    instances = pd.read_parquet(INSTANCES)
    math = instances[(instances["model_id"] == DEEPSEEK) & instances["dataset_id"].str.startswith("hendrycks")]
    math = math.merge(nll[nll["scored_model_id"] == DEEPSEEK], on="problem_key", validate="one_to_one")
    math["group"] = math["problem_key"]
    math["S1"] = math["mean_nll"] < medians[DEEPSEEK]
    math["S2"] = math["problem"].str.len() < length_median
    out["P2 DeepSeek MATH"] = {"S1": math, "S2": math}

    proxy = proxy_scores()
    for name, source in (("P3 Putnam-AXIOM", "putnam_axiom"), ("P5 MATH-Perturb", "math_perturb")):
        frame = derive_labels(read_shards(ROOT / "runs" / "e35-external" / source), source)
        frame = frame.merge(proxy[proxy["src"] == source][["key", "mean_nll"]], left_on="problem_id", right_on="key",
                            validate="one_to_one")
        frame["group"] = frame["problem_id"].astype(str)
        frame["S1"] = frame["mean_nll"] < medians[PROXY]
        frame["S2"] = frame["problem"].str.len() < length_median
        out[name] = {"S1": frame, "S2": frame}

    # S3: E37's frozen predictor and the threshold E36 recorded (amendment 1): recomputing the
    # out-of-fold threshold here gives a different value, because fold assignment depends on the
    # library build, while the fitted model itself reproduces E36 exactly (checked below).
    train = build_examples(read_shards(MATH_PERTURB), "math_perturb")
    columns = feature_columns(train)
    e36 = json.loads(E36_RESULT.read_text(encoding="utf-8"))
    threshold = float(e36["oof_threshold"])
    model = make_model().fit(train[columns], train["is_robust"].to_numpy(bool))
    predict = lambda frame: model.predict_proba(frame[columns])[:, 1] >= threshold  # noqa: E731

    putnam = build_examples(read_shards(PUTNAM), "putnam_axiom")
    auc = roc_auc_score(putnam["is_robust"].astype(bool), model.predict_proba(putnam[columns])[:, 1])
    assert abs(auc - e36["holdout_auc"]) < 1e-6, f"the predictor no longer reproduces E36 ({auc} vs {e36['holdout_auc']})"
    putnam["group"], putnam["S3"] = putnam["problem_id"].astype(str), predict(putnam)
    out["P3 Putnam-AXIOM"]["S3"] = putnam

    gsm_raw = read_shards(GSM)
    gsm = build_gsm_examples(gsm_raw)
    gsm["group"], gsm["S3"] = gsm["problem_id"].astype(str), predict(gsm)
    text = gsm_raw.drop_duplicates("problem_id").set_index("problem_id")["question"]
    gsm["S2"] = gsm["problem_id"].map(text).str.len() < length_median
    out["P4 GSM-Plus"] = {"S3": gsm, "S2": gsm}

    features = read_feature_records([ROOT / "runs" / run for run in E49_RUNS])
    labels = pd.read_parquet(ROOT / "data" / "train_main" / "rows.parquet")
    labels = labels.loc[labels["model_id"].isin(SMALL_MODELS) & labels["is_labelled"].astype(bool),
                        ["model_id", "reasoning_effort", "problem_key", "is_robust"]]
    official = features.drop(columns=["is_robust", "budget_summaries"], errors="ignore").merge(
        labels, on=["model_id", "reasoning_effort", "problem_key"], validate="one_to_one")
    official = official.merge(rows[["model_id", "problem_key", "template"]], on=["model_id", "problem_key"])
    official["group"], official["S3"] = official["template"], predict(official)
    out["P1 official rows"]["S3"] = official
    return out, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()

    medians = reference_nll()
    length_median = float(pd.read_parquet(REFERENCE)["problem"].str.len().median())
    pops, threshold = populations(medians, length_median)

    signals = {"S1": "familiarity, NLL below reference median", "S2": "length below reference median",
               "S3": "budget path, P at least E36's threshold"}
    result: dict = {"thresholds": {"nll_reference_median": medians, "length_reference_median": length_median,
                                   "budget_path_threshold": threshold}, "cells": {}, "qualifies": {}}
    lines = ["Every rule fixed in `protocols/E54_PROTOCOL.md` before this table was computed. Balanced accuracy "
             "of the per-problem decisions, 95% problem-bootstrap interval; any constant scores 0.5.", "",
             "| population | rows | robust | " + " | ".join(f"{s} {signals[s]}" for s in signals) + " |",
             "| --- | --- | --- |" + " --- |" * len(signals)]
    for pop, frames in pops.items():
        cells, size = [], None
        for s in signals:
            if s not in frames:
                cells.append("—")
                continue
            m = balanced(frames[s], s)
            result["cells"].setdefault(s, {})[pop] = m
            size = size or m
            cells.append(f"{m['balanced_accuracy']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")
        lines.append(f"| {pop} | {size['n']} | {size['robust_share']:.2f} | " + " | ".join(cells) + " |")

    constants = balanced(pops["P1 official rows"]["constants"], "constants")
    lines += ["", f"On P1 the per-model constants score accuracy {constants['accuracy']:.3f} (balanced "
              f"{constants['balanced_accuracy']:.3f}); the signals there score accuracy "
              + ", ".join(f"{s} {result['cells'][s]['P1 official rows']['accuracy']:.3f}" for s in signals
                          if 'P1 official rows' in result["cells"].get(s, {})) + "."]
    result["p1_constants"] = constants

    lines += ["", "| signal | populations with interval above 0.5 | lowest point | qualifies |", "| --- | --- | --- | --- |"]
    for s in signals:
        cells = result["cells"][s]
        above = [p for p, m in cells.items() if m["ci"][0] > 0.5]
        lowest = min(m["balanced_accuracy"] for m in cells.values())
        ok = len(above) >= 2 and lowest >= 0.5
        result["qualifies"][s] = ok
        lines.append(f"| {s} | {len(above)} ({', '.join(p.split()[0] for p in above) or 'none'}) | {lowest:.3f} | "
                     f"**{'yes' if ok else 'no'}** |")
    verdict = [s for s, ok in result["qualifies"].items() if ok]
    lines += ["", f"**Decision (§5): {('qualifies: ' + ', '.join(verdict)) if verdict else 'nothing qualifies; the constants stay the per-problem floor and E56 is not run'}.**",
              "", f"Thresholds: reference NLL medians {', '.join(f'{SHORT.get(k, k)} {v:.3f}' for k, v in medians.items())}; "
              f"reference length median {length_median:.0f} characters; budget-path threshold {threshold:.4f}."]

    body = "\n".join(lines)
    print(body)
    out = ROOT / "runs" / "e54"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E54: per-problem rules against the constants", body)
        print("\nwrote the E54 section of RESULTS.md")


if __name__ == "__main__":
    main()
