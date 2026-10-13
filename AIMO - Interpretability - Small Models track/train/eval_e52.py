"""E52: do our models notice an impossible premise, and can one forward pass predict which ones?

Finding 76 located the family that decides olympiad labels: `expert_no_solution`, an added premise
that contradicts the rest. This scores that behaviour on ReliableMath's AMC 2023 contradiction
rewrites (`datagen/perturb/reliablemath_pool.py`), sampled with the organizers' decoding settings,
and answers the three questions of `protocols/E52_PROTOCOL.md`. Everything below - the grader, the
label thresholds, the probe recipe, the gate - was fixed in that protocol before any record was read.

**Grading one rewrite sample.**

* `truncated`: the sample used the whole token budget. It never flagged the contradiction within
  budget, so it counts as **not detected** (the protocol also reports the rate without them).
* `answered`: it finished with an integer final answer (`answers.extract_answer`); not detected.
  `kept_original` is the subset that gave the original problem's answer, the clearest case of
  answering from the pattern rather than the premises.
* `detected`: it finished, gave no integer, and its ending says there is no valid answer.
* `other`: it finished with neither; not detected, and counted so the audit can see it.

**Per (model, problem).** Eligible iff every base sample solved the original. Detection rate =
detected / all rewrite samples of the problem. Mirroring the official rule against a base accuracy
of 1.0, `robust_like` iff the rate >= 0.90 (a drop of at most 0.10) and `spurious_like` iff
<= 0.75 (a drop of at least 0.25); with about four samples per problem nothing falls between.

Writes the `E52: contradiction detection` section of RESULTS.md and runs/e52-analysis/result.json.

Run:  python train/eval_e52.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "datagen" / "perturb"))

from answers import extract_answer  # noqa: E402
from results_io import upsert_section  # noqa: E402

MODELS = {"qwen": "Qwen/Qwen3.5-4B", "skywork": "Skywork/Skywork-OR1-Math-7B"}
RECORDS = "runs/e52-{short}-*/e52-{short}*/{phase}/shard_*/records.jsonl"
FEATURES = "runs/e52-features-*/e52-features/{kind}-{short}"
POOL = ROOT / "data" / "reliablemath" / "pool.parquet"
OUT = ROOT / "runs" / "e52-analysis"
CAP = 16384  # --max-new-tokens of every E52 sampling run
ROBUST, SPURIOUS = 0.90, 0.75
MIN_PAIRS = 60  # gate 3 is only read with at least this many eligible (model, problem) pairs
NO_SOLUTION = re.compile(
    r"no (?:valid |real |such |possible |integer )?(?:solution|answer|value)|not solvable|unsolvable|"
    r"cannot be (?:determined|solved|satisfied|found)|can't be (?:determined|solved)|does ?n[o']t exist|"
    r"no such|impossible|contradict|inconsistent|ill-posed|not well-defined|undefined|"
    r"\\emptyset|\\varnothing|\\text\{none\}|\bnone\b",
    re.IGNORECASE,
)


def grade(tail: str, tokens: int, cap: int = CAP) -> str:
    """One rewrite sample: `truncated`, `answered`, `detected` or `other` (see the module docstring)."""
    if tokens >= cap:
        return "truncated"
    if extract_answer(tail) is not None:
        return "answered"
    ending = tail.rsplit("</think>", 1)[-1] if "</think>" in tail else tail[-600:]
    return "detected" if NO_SOLUTION.search(ending) else "other"


def load_records(short: str, phase: str) -> list[dict]:
    paths = sorted(ROOT.glob(RECORDS.format(short=short, phase=phase)))
    rows = [json.loads(line) for p in paths for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return list({r["item_id"]: r for r in rows}.values())  # a resumed shard may repeat an item


def pairs(short: str) -> pd.DataFrame:
    """One row per (model, problem): eligibility and rewrite outcomes."""
    base = {r["pool_id"]: r for r in load_records(short, "base")}
    rows = []
    for record in load_records(short, "variants"):
        for tail, tokens, answer in zip(record["sample_tails"], record["completion_tokens"], record["answers"]):
            outcome = grade(tail, tokens)
            rows.append({"pool_id": record["pool_id"], "outcome": outcome,
                         "kept_original": outcome == "answered" and answer == record["expected_answer"]})
    samples = pd.DataFrame(rows, columns=["pool_id", "outcome", "kept_original"])
    out = pd.DataFrame({"pool_id": list(base)})
    out["eligible"] = [base[p]["n_correct"] == base[p]["n_samples"] > 0 for p in out.pool_id]
    counts = samples.pivot_table(index="pool_id", columns="outcome", aggfunc="size", fill_value=0)
    out = out.merge(counts, left_on="pool_id", right_index=True, how="left").fillna(0)
    for column in ("truncated", "answered", "detected", "other"):
        out[column] = out.get(column, 0)
    out["kept_original"] = out.pool_id.map(samples.groupby("pool_id").kept_original.sum()).fillna(0)
    out["n_rewrite_samples"] = out[["truncated", "answered", "detected", "other"]].sum(axis=1)
    out["rate"] = out.detected / out.n_rewrite_samples.replace(0, np.nan)
    finished = out.n_rewrite_samples - out.truncated
    out["rate_finished"] = out.detected / finished.replace(0, np.nan)
    out["label"] = np.select([out.rate >= ROBUST, out.rate <= SPURIOUS], ["robust_like", "spurious_like"], "band")
    out["model"] = short
    return out


def problem_bootstrap(frame: pd.DataFrame, statistic, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval resampling problems, so a problem's pairs across models move together."""
    rng = np.random.default_rng(seed)
    problems = frame.pool_id.unique()
    values = []
    for _ in range(n_boot):
        draw = rng.choice(problems, size=len(problems))
        sample = pd.concat([frame[frame.pool_id == p] for p in draw])
        try:
            values.append(statistic(sample))
        except ValueError:  # one class only in this draw
            continue
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def familiarity(short: str) -> pd.Series:
    paths = sorted(ROOT.glob(FEATURES.format(kind="familiarity", short=short) + "/shard_*/records.jsonl"))
    rows = [json.loads(line) for p in paths for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows).drop_duplicates("problem_key").set_index("problem_key")["mean_nll"]


def probe_scores(short: str, keys: list[str], y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Out-of-fold layer-ensemble probability (Finding 11's recipe): one L2 logistic probe per layer on
    mean-pooled prompt states, standardised in-fold, averaged over every layer."""
    directory = next(ROOT.glob(FEATURES.format(kind="stagea", short=short)))
    bundle = np.load(directory / "features.npz", allow_pickle=True)
    index = {k: i for i, k in enumerate(bundle["problem_keys"])}
    X = bundle["X_mean"][[index[k] for k in keys]].astype(np.float64)
    scores = np.zeros(len(keys))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in folds.split(X, y):
        per_layer = [make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
                     .fit(X[train, layer], y[train]).predict_proba(X[test, layer])[:, 1] for layer in range(X.shape[1])]
        scores[test] = np.mean(per_layer, axis=0)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()

    pool = pd.read_parquet(POOL).set_index("pool_id")
    frame = pd.concat([pairs(short) for short in MODELS], ignore_index=True)
    frame["problem_key"] = frame.pool_id.map(pool.problem_key)
    frame["length"] = frame.pool_id.map(pool.problem.str.len())
    eligible = frame[frame.eligible & (frame.n_rewrite_samples > 0)].copy()
    total = eligible[["truncated", "answered", "detected", "other"]].sum()

    lines = ["**Q1: how often each model notices the contradiction**, on problems it solved in every base sample:", "",
             "| model | problems sampled | eligible | detection rate [95% CI] | without truncated samples | "
             "robust-like | answered the original's answer |", "| --- | --- | --- | --- | --- | --- | --- |"]
    result = {"outcomes": total.astype(int).to_dict(), "models": {}}
    for short in MODELS:
        part = eligible[eligible.model == short]
        if part.empty:
            lines.append(f"| {short} | {int((frame.model == short).sum())} | 0 | - | - | - | - |")
            continue
        low, high = problem_bootstrap(part, lambda f: f.rate.mean())
        kept = part.kept_original.sum() / part.n_rewrite_samples.sum()
        lines.append(f"| {short} | {int((frame.model == short).sum())} | {len(part)} | {part.rate.mean():.3f} "
                     f"[{low:.3f}, {high:.3f}] | {part.rate_finished.mean():.3f} | "
                     f"{(part.label == 'robust_like').mean():.3f} | {kept:.3f} |")
        result["models"][short] = {"eligible": len(part), "rate": part.rate.mean(), "ci": [low, high],
                                   "robust_like": float((part.label == "robust_like").mean())}
    lines += ["", "Rewrite samples by outcome: " + ", ".join(f"{k} {int(v)}" for k, v in total.items()) + "."]

    wide = eligible.pivot_table(index="pool_id", columns="model", values="rate")
    if set(MODELS) <= set(wide.columns) and len(wide.dropna()) >= 5:
        shared = wide.dropna()
        rho = shared.corr(method="spearman").iloc[0, 1]
        kappa = cohen_kappa_score(shared.iloc[:, 0] >= ROBUST, shared.iloc[:, 1] >= ROBUST)
        lines += ["", f"**Q2: is detection a property of the problem?** On {len(shared)} problems both models solved: "
                  f"Spearman of detection rates {rho:.3f}, Cohen's kappa of robust-like {kappa:.3f}."]
        result["q2"] = {"n": len(shared), "spearman": rho, "kappa": kappa}

    labelled = eligible[eligible.label != "band"].copy()
    labelled["y"] = (labelled.label == "robust_like").astype(int)
    lines += ["", f"**Q3: does one forward pass on the original predict it?** {len(labelled)} labelled pairs "
              f"({int(labelled.y.sum())} robust-like). Gate: the probe's pooled AUC has a problem-bootstrap CI lower "
              f"bound above 0.5, with at least {MIN_PAIRS} pairs.", "",
              "| signal | direction | AUC [95% CI] |", "| --- | --- | --- |"]
    scored = []
    for short in MODELS:
        part = labelled[labelled.model == short].copy()
        if part.y.nunique() < 2 or part.y.value_counts().min() < 5:
            continue
        part["nll"] = -part.problem_key.map(familiarity(short))  # e27k's direction: familiar = robust
        part["short"] = -part.length
        part["probe"] = probe_scores(short, list(part.problem_key), part.y.to_numpy())
        scored.append(part)
    gate = {"pairs": len(labelled), "passed": False}
    if scored:
        both = pd.concat(scored)
        for column, direction in (("probe", "learned in-fold"), ("nll", "familiar = robust (e27k)"),
                                  ("short", "shorter = robust")):
            auc = roc_auc_score(both.y, both[column])
            low, high = problem_bootstrap(both, lambda f, c=column: roc_auc_score(f.y, f[c]))
            lines.append(f"| {column} | {direction} | {auc:.3f} [{low:.3f}, {high:.3f}] |")
            gate[column] = {"auc": auc, "ci": [low, high]}
        gate["passed"] = bool(len(labelled) >= MIN_PAIRS and gate["probe"]["ci"][0] > 0.5)
    lines += ["", f"**Gate 3: {'PASS' if gate['passed'] else 'FAIL'}.**"]
    result["gate"] = gate

    body = "\n".join(lines)
    print(body)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E52: contradiction detection", body)
        print("\nwrote the E52 section of RESULTS.md")


if __name__ == "__main__":
    main()
