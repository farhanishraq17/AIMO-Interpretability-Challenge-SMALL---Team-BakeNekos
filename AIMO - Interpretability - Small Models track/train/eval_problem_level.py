"""E22: the problem-level acceptance harness every later gate is scored on.

Two findings decide how official labels must be used. Robustness is a property of the problem,
not the model: the four small-track models agree on 20 of 20 shared official problems
(Finding 38). And `train-main-v2` repeats a template with one constant swapped, sometimes under
a different label (Finding 34). So the unit of evaluation is the **problem**, with labels pooled
across every configuration that labelled it (gpt-oss included), and every split and resample
is **by template**, never by row or by text.

Scored here, all leave-one-template-out (LOTO) on the official problems:

1. constants - the reference any per-problem signal must beat;
2. the shipped MATH-trained text block, unchanged (Finding 39's model, now problem-level);
3. char tf-idf refitted on the official problem labels;
4. the same, with DeepSeek's MATH instances added to every training fold;
5. problem length alone, the confound every text signal has to clear (Finding 22).

Every AUC carries a template-cluster bootstrap CI and a label-permutation p-value. With about
30 labelled problems, only a strong signal can clear chance, so read the CI, not the point.

Other scripts import `load_problems`, `score` and `loto` so E23/E24 signals are scored the
same way. Writes the `E22: problem-level acceptance harness` section of RESULTS.md.

Run:  uv run python ../train/eval_problem_level.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "datagen"))

from ingest_public import problem_key  # noqa: E402
from inspect_train_main_v2 import cluster_templates  # noqa: E402
from results_io import upsert_section  # noqa: E402
from train_stagea_probes import text_model  # noqa: E402

DATASET = "aimo-interp/train-main-v2"
REVISION = "23791d9b120132087877b0459f953d5959fc8455"
PARQUET = "data/train-00000-of-00001.parquet"
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
ARTIFACT = ROOT / "submission" / "e11-text-stagea" / "artifacts" / "e11_deepseek.joblib"
RESULTS = ROOT / "RESULTS.md"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
SMALL_TRACK = (
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    DEEPSEEK,
)


def load_rows(revision: str = REVISION) -> pd.DataFrame:
    """Labelled `train-main-v2` rows with `problem_key` and a template id."""
    table = pd.read_parquet(hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=revision))
    table["problem_key"] = table["problem"].map(problem_key)
    texts = sorted(table["problem"].map(lambda t: " ".join(t.split())).unique())
    template = cluster_templates(texts)
    table["template"] = table["problem"].map(lambda t: template[" ".join(t.split())])
    return table[table["is_robust"].notna()].reset_index(drop=True)


def load_problems(revision: str = REVISION) -> pd.DataFrame:
    """One row per labelled problem: its pooled label, template and who labelled it.

    The pooled label is the majority over every configuration; the rare problem whose
    configurations split evenly gets no label, like the organizers' own ambiguous band.
    """
    rows = load_rows(revision)
    rows["is_robust"] = rows["is_robust"].astype(bool)
    grouped = rows.groupby("problem_key")
    problems = pd.DataFrame({
        "problem": grouped["problem"].first(),
        "template": grouped["template"].first(),
        "n_labels": grouped.size(),
        "n_robust": grouped["is_robust"].sum(),
        "small_track": grouped["model_id"].agg(lambda m: bool(set(m) & set(SMALL_TRACK))),
    })
    share = problems["n_robust"] / problems["n_labels"]
    problems["label"] = np.where(share > 0.5, 1.0, np.where(share < 0.5, 0.0, np.nan))
    return problems[problems["label"].notna()].reset_index()


def score(
    y: np.ndarray, s: np.ndarray, groups: np.ndarray, rng: np.random.Generator,
    n_boot: int = 2000, n_perm: int = 2000,
) -> dict[str, float]:
    """ROC-AUC with a template-cluster bootstrap CI and a label-permutation p-value."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    auc = roc_auc_score(y, s)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(unique, size=len(unique))])
        if y[idx].any() and not y[idx].all():
            boot.append(roc_auc_score(y[idx], s[idx]))
    low, high = np.percentile(boot, [2.5, 97.5])
    null = np.array([roc_auc_score(rng.permutation(y), s) for _ in range(n_perm)])
    p_value = (1 + np.sum(null >= auc)) / (1 + n_perm)
    return {"auc": float(auc), "low": float(low), "high": float(high), "p": float(p_value)}


def loto(problems: pd.DataFrame, fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]) -> np.ndarray:
    """Leave-one-template-out predictions: a template's siblings never train its own prediction."""
    out = np.full(len(problems), np.nan)
    for template in problems["template"].unique():
        held = (problems["template"] == template).to_numpy()
        train = problems[~held]
        if train["label"].nunique() < 2:
            out[held] = train["label"].mean()
            continue
        out[held] = fit_predict(train, problems[held])
    return out


def official_text(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = text_model().fit(list(train["problem"]), train["label"].astype(bool))
    return model.predict_proba(list(test["problem"]))[:, 1]


def official_plus_math(math: pd.DataFrame) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        texts = list(train["problem"]) + list(math["problem"])
        labels = list(train["label"].astype(bool)) + list(math["is_robust"].astype(bool))
        return text_model().fit(texts, labels).predict_proba(list(test["problem"]))[:, 1]
    return fit_predict


def length_only(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    feature = lambda frame: np.log(frame["problem"].str.len().to_numpy())[:, None]  # noqa: E731
    model = LogisticRegression(class_weight="balanced").fit(feature(train), train["label"].astype(bool))
    return model.predict_proba(feature(test))[:, 1]


def math_agreement() -> str:
    """Does the problem-level reading hold on MATH? DeepSeek against Llama on shared problems."""
    instances = pd.read_parquet(INSTANCES)
    table = instances.pivot_table(index="problem_key", columns="model_id", values="is_robust", aggfunc="first")
    if DEEPSEEK not in table or LLAMA not in table:
        return "No shared DeepSeek/Llama problems."
    shared = table[[DEEPSEEK, LLAMA]].dropna().astype(bool)
    agree = int((shared[DEEPSEEK] == shared[LLAMA]).sum())
    deepseek_spurious = 1 - float(instances.loc[instances["model_id"] == DEEPSEEK, "is_robust"].mean())
    return (
        f"On the {len(shared)} Hendrycks MATH problems labelled for both DeepSeek and Llama-3.1-8B, the two "
        f"agree on **{agree}** ({agree / len(shared):.2f}). Llama is spurious on every problem it has, so "
        f"agreement here is just DeepSeek's spurious rate on these problems, against {deepseek_spurious:.2f} "
        "over all of DeepSeek's: a weak test that cannot confirm Finding 38 outside `train-main-v2`."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    problems = load_problems(args.revision)
    y = problems["label"].astype(bool).to_numpy()
    groups = problems["template"].to_numpy()
    math = pd.read_parquet(INSTANCES)
    math = math[(math["model_id"] == DEEPSEEK) & math["dataset_id"].str.startswith("hendrycks")]

    shipped = joblib.load(ARTIFACT)["text_model"]
    candidates = {
        "shipped MATH text block (no refit)": shipped.predict_proba(list(problems["problem"]))[:, 1],
        "char tf-idf refit on official labels (LOTO)": loto(problems, official_text),
        f"char tf-idf, official + {len(math)} DeepSeek MATH rows (LOTO)": loto(problems, official_plus_math(math)),
        "problem length only (LOTO)": loto(problems, length_only),
    }

    lines = [
        f"`{DATASET}` @ `{args.revision[:12]}`: **{len(problems)} labelled problems in "
        f"{problems['template'].nunique()} templates** ({int(y.sum())} robust), labels pooled over every "
        f"configuration; {int(problems['small_track'].sum())} are labelled for a small-track model. "
        "Leave-one-template-out; 95% CIs resample templates. Regenerated by `train/eval_problem_level.py`.",
        "",
        "| Signal | ROC-AUC | 95% CI | Permutation p | Balanced acc. at 0.5 |",
        "| --- | --- | --- | --- | --- |",
        "| always robust / always spurious | 0.500 | — | — | 0.500 |",
    ]
    for name, prediction in candidates.items():
        result = score(y, prediction, groups, rng)
        balanced = balanced_accuracy_score(y, prediction >= 0.5)
        lines.append(
            f"| {name} | {result['auc']:.3f} | [{result['low']:.3f}, {result['high']:.3f}] "
            f"| {result['p']:.3f} | {balanced:.3f} |"
        )
    lines += [
        "",
        "A signal passes only with a CI lower bound above 0.5 (gates G-M and G2, `PLAN.md` §11).",
        "",
        "### Does agreement hold on MATH too?",
        "",
        math_agreement(),
    ]
    upsert_section(RESULTS, "E22: problem-level acceptance harness", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
