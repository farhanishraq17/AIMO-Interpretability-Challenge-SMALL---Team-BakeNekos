"""Review, 6 Oct: does the shipped ranking generalize past the populations it was chosen on?

`e27k` ranks each batch by the scored model's mean token NLL of the bare problem. Its evidence is
DeepSeek's Hendrycks MATH rows and the 14-case leaderboard; the 34 official olympiad rows point
the other way (Finding 58). This script adds what was missing from that argument:

1. **How much of the ranking is problem length.** Spearman of mean NLL against token count, per
   model, on the E23 features.
2. **Two more labelled populations.** E35 already holds problem-level robustness labels for
   MATH-Perturb (233) and Putnam-AXIOM (48). No competition checkpoint is cached locally, so
   familiarity is scored with `Qwen/Qwen3-0.6B` on CPU. Finding 68 says the ranking is a problem
   property; the proxy's agreement with each competition model is printed first so the reader
   can judge that substitution.
3. **The sign of E22's length signal.** E22 reports a leave-one-template-out *fitted* length
   model; the raw direction and the fitted coefficient are printed beside it.
4. **What the robust count is worth at test-set prevalence.** Expected accuracy of calling the
   top fraction `c` of a batch robust, under a binormal ranking of a given AUC.

Nothing here is written to RESULTS.md, and no official label is used to choose anything.

Run:  python train/review_generalization.py        # first run scores ~470 problems on CPU, ~3 min
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for path in ("train", "scripts", "datagen"):
    sys.path.insert(0, str(ROOT / path))

from eval_e35_external import derive_labels, read_shards  # noqa: E402
from eval_familiarity import DEEPSEEK, INSTANCES, load_scores  # noqa: E402
from eval_label_free import within_length_auc  # noqa: E402
from eval_problem_level import length_only, load_problems, load_rows, loto  # noqa: E402

PROXY = "Qwen/Qwen3-0.6B"
CACHE = ROOT / "runs" / "review-generalization" / "proxy_nll.jsonl"
E35 = ROOT / "runs" / "e35-external"


def external(source: str) -> pd.DataFrame:
    return derive_labels(read_shards(E35 / source), source)


def proxy_scores() -> pd.DataFrame:
    """Familiarity under the proxy for every E23 problem and both E35 populations; resumable."""
    from extract_familiarity import familiarity, load_problems as e23_problems

    items = [("e23", r.problem_key, r.problem) for r in e23_problems(INSTANCES).itertuples()]
    for source in ("math_perturb", "putnam_axiom"):
        items += [(source, r.problem_id, r.problem) for r in external(source).itertuples()]
    done = set()
    if CACHE.is_file():
        done = {(j["src"], j["key"]) for j in map(json.loads, CACHE.read_text(encoding="utf-8").splitlines())}
    todo = [item for item in items if item[:2] not in done]
    if todo:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModelForCausalLM

        from tokenizer_check import load_checked_tokenizer

        print(f"scoring {len(todo)} problems with {PROXY} on CPU")
        tokenizer, _ = load_checked_tokenizer(PROXY, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(PROXY, dtype=torch.float32, local_files_only=True).eval()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as handle:
            for source, key, text in todo:
                handle.write(json.dumps({"src": source, "key": key, **familiarity(model, tokenizer, text)}) + "\n")
                handle.flush()
    return pd.read_json(CACHE, lines=True, dtype={"key": str})


def residual(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    design = np.c_[np.ones(len(covariate)), covariate]
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def auc_row(name: str, y, frame: pd.DataFrame) -> str:
    """AUCs oriented so > 0.5 means the shipped direction: longer / more familiar is robust."""
    y = np.asarray(y, dtype=bool)
    chars = frame["problem"].str.len().to_numpy()
    nll = frame["mean_nll"].to_numpy()
    cells = [
        roc_auc_score(y, np.log(chars)),
        roc_auc_score(y, -nll),
        roc_auc_score(y, -frame["suffix_nll"].to_numpy()),
        roc_auc_score(y, -residual(nll, np.log(frame["n_tokens"].to_numpy()))),
        within_length_auc(y, -nll, chars),
    ]
    return f"| {name} | {len(y)} | {y.mean():.2f} | " + " | ".join(f"{c:.3f}" for c in cells) + " |"


def expected_accuracy(c: float, prevalence: float, auc: float) -> float:
    """Call the top fraction `c` robust; robust scores ~ N(d, 1), spurious ~ N(0, 1)."""
    if c >= 1:
        return prevalence
    d = np.sqrt(2) * norm.ppf(auc)
    low, high = -10.0, 10.0
    for _ in range(80):
        t = (low + high) / 2
        called = prevalence * norm.sf(t - d) + (1 - prevalence) * norm.sf(t)
        low, high = (t, high) if called > c else (low, t)
    return prevalence * norm.sf(t - d) + (1 - prevalence) * norm.cdf(t)


def main() -> None:
    e23 = load_scores()
    proxy = proxy_scores()
    mine = proxy[proxy["src"] == "e23"].set_index("key")

    print("## 1. Mean NLL is mostly problem length\n")
    print("| scoring model | Spearman(mean NLL, tokens), official | same, MATH | Spearman with proxy |")
    print("| --- | --- | --- | --- |")
    official_keys = set(load_rows()["problem_key"])
    for model, part in e23.groupby("scored_model_id"):
        on = part["problem_key"].isin(official_keys)
        joined = part.set_index("problem_key").join(mine, rsuffix="_proxy", how="inner")
        print(f"| `{model}` | {spearmanr(part.loc[on, 'mean_nll'], part.loc[on, 'n_tokens'])[0]:.2f} "
              f"| {spearmanr(part.loc[~on, 'mean_nll'], part.loc[~on, 'n_tokens'])[0]:.2f} "
              f"| {spearmanr(joined['mean_nll'], joined['mean_nll_proxy'])[0]:.3f} |")

    print(f"\n## 2. The shipped direction on four labelled populations (scored by `{PROXY}`)\n")
    print("| population | n | robust | longer | familiar (mean NLL) | suffix NLL | NLL, length-residualised "
          "| NLL within length quintiles |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    instances = pd.read_parquet(INSTANCES)
    math = instances[(instances["model_id"] == DEEPSEEK) & instances["dataset_id"].str.startswith("hendrycks")]
    math = math.merge(mine, left_on="problem_key", right_index=True)
    print(auc_row("DeepSeek, Hendrycks MATH", math["is_robust"], math))
    problems = load_problems().merge(mine, left_on="problem_key", right_index=True)
    print(auc_row("official, problem level (E22 pooling)", problems["label"], problems))
    rows = load_rows()
    rows = rows[rows["model_id"].isin(e23["scored_model_id"].unique())].merge(mine, left_on="problem_key", right_index=True)
    print(auc_row("official, small-track rows", rows["is_robust"], rows))
    for source in ("math_perturb", "putnam_axiom"):
        frame = external(source).merge(proxy[proxy["src"] == source], left_on="problem_id", right_on="key")
        print(auc_row(f"E35 {source}", frame["is_robust"], frame))

    print("\n## 3. The sign of E22's length signal\n")
    problems = load_problems()
    y = problems["label"].astype(bool).to_numpy()
    log_length = np.log(problems["problem"].str.len().to_numpy())
    coefficient = LogisticRegression(class_weight="balanced").fit(log_length[:, None], y).coef_[0, 0]
    print(f"- raw AUC, longer = robust: {roc_auc_score(y, log_length):.3f}")
    print(f"- E22's leave-one-template-out fitted AUC: {roc_auc_score(y, loto(problems, length_only)):.3f}")
    print(f"- coefficient on log length, all 33 problems: {coefficient:+.3f} (negative: shorter is robust)")

    print("\n## 4. Expected accuracy of calling the top fraction c robust\n")
    fractions = (0.5, 0.6, 0.7, 1.0)
    print("| ranking AUC | test prevalence | " + " | ".join(f"c = {c:.1f}" for c in fractions) + " |")
    print("| --- | --- |" + " --- |" * len(fractions))
    for auc in (0.4, 0.5, 0.6, 0.7):
        for prevalence in (0.5, 0.6, 0.7):
            cells = " | ".join(f"{expected_accuracy(c, prevalence, auc):.3f}" for c in fractions)
            print(f"| {auc:.1f} | {prevalence:.1f} | {cells} |")


if __name__ == "__main__":
    main()
