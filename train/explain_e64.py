"""E64 explained: how noisy its official check is, why it reverses on MATH, and what its ridge reads.

E64 ships (Finding 88), E65 found it reversing on DeepSeek's MATH rows (Finding 89), and E66's better-fitted refit
ranked the official rows worse (Finding 90). Nothing here is a gate or changes what ships: every number is exploratory,
computed after those reads, and labelled so. Three parts:

* noise: E64's fit, unchanged, on random subsets of its 497 problems and on one-sample targets. How far the 34-row AUC
  moves when the method does not change, which is the yardstick any refit has to beat;
* MATH: which perturbation family sets DeepSeek's MATH labels, which problems it was run on, and E64 and familiarity
  on the problems it never touched;
* inside the ridge: single layers, the top principal components, the four models' agreement on problems outside the
  fit, and how much of the score interpretable text features reproduce.

Writes runs/explain-e64/result.json and the `E64 explained: noise, MATH and what the ridge reads` section of RESULTS.md.

Run:  python train/explain_e64.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for path in ("train", "scripts", "datagen", "submission/e64-difficulty"):
    sys.path.insert(0, str(ROOT / path))

import eval_e64 as e64  # noqa: E402
import eval_e65 as e65  # noqa: E402
from eval_e61 import SHORT  # noqa: E402
from eval_familiarity import DEEPSEEK, load_scores  # noqa: E402
from eval_problem_level import load_rows  # noqa: E402
from ingest_public import problem_key  # noqa: E402
from results_io import upsert_section  # noqa: E402

OUT = ROOT / "runs" / "explain-e64"
HEADING = "E64 explained: noise, MATH and what the ridge reads"
SIZES, DRAWS, CURVE_SEED = (125, 250, 375), 5, 67
# The DeepSeek MATH tables that keep the variant texts (`qwen3-8b:low` is DeepSeek's Codabench alias).
# Revisions pinned to the snapshots these numbers were read from.
VARIANT_REPOS = {"aimo-interp/augmented-sample-math-agg": "f972ced0705096f8d7ca7fac30825900b8b7fb6a",
                 "aimo-interp/augmented-sample-math-agg-filtered": "0607cd06cbd789dd33655ff718bbfae0d3edba7c",
                 "aimo-interp/augmented-sample-math-full-filtered": "f09e41ed0648b0ff7cac220c6ce9150a36e2e8b0"}
VARIANT_FILE = "data/validation-00000-of-00001.parquet"
MATH_REPO, MATH_REVISION = "EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3"
DEPTHS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
RANKS = (1, 5, 20, 50)
TEXT_ALPHAS = np.logspace(-2, 4, 13)
TOPICS = {"geometry": r"triangle|circle|angle|polygon|square|rectangle|radius|tangent|perpendicular|parallel|area|hexagon|quadrilateral",
          "counting": r"how many|number of ways|probability|arrangements?|choose|subsets?|colou?r|permutations?|games?",
          "number_theory": r"remainder|divisib|prime|integers?|digits?|gcd|mod|divisors?|multiple",
          "algebra": r"polynomial|roots?|real numbers?|equations?|f\(|function|inequalit|sequence",
          "proof": r"\bprove|\bshow that|find all|determine all",
          "extremal": r"maxim|minim|largest|smallest|greatest|least"}


def official_auc(rows: pd.DataFrame, predicted: dict[str, pd.Series]) -> float:
    score = [-predicted[m].loc[k] for m, k in zip(rows["model_id"], rows["problem_key"])]
    return float(roc_auc_score(rows["is_robust"].astype(bool), score))


def load_features() -> dict[str, tuple[pd.Index, np.ndarray]]:
    feats = {}
    for model, short in SHORT.items():
        bundle = np.load(next(ROOT.glob(e64.FEATURES.format(short=short))), allow_pickle=True)
        feats[model] = (pd.Index([str(k) for k in bundle["problem_keys"]]), bundle["X_mean"].astype(np.float32))
    return feats


def refit(feats, y: pd.Series, train: list[str]) -> tuple[dict[str, pd.Series], float]:
    """E64's fit on `train` only: predictions on every E55 problem, and the mean Spearman on the unused problems."""
    held = y.index.difference(train)
    predicted, spearman = {}, []
    for model, (keys, X) in feats.items():
        fits = e64.layer_fit(X[keys.get_indexer(train)], y.loc[train].to_numpy())
        predicted[model] = pd.Series(e64.layer_predict(fits, X), index=keys)
        if len(held):
            spearman.append(spearmanr(predicted[model].loc[held], y.loc[held]).statistic)
    return predicted, float(np.mean(spearman)) if spearman else float("nan")


def noise(feats, rows: pd.DataFrame) -> dict:
    tokens = {json.loads(line)["pool_id"].removeprefix("e62:"): json.loads(line)["completion_tokens"]
              for path in ROOT.glob(e64.BASE) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    log = {k: np.log(np.minimum(v, e64.CAP)) for k, v in tokens.items()}
    targets = {"both samples (E64)": pd.Series({k: float(v.mean()) for k, v in log.items()}),
               "first sample only": pd.Series({k: float(v[0]) for k, v in log.items()}),
               "second sample only": pd.Series({k: float(v[1]) for k, v in log.items()})}
    out = {"targets": {name: official_auc(rows, refit(feats, y, list(y.index))[0]) for name, y in targets.items()}}
    y, rng, curve = targets["both samples (E64)"], np.random.default_rng(CURVE_SEED), {}
    for n in SIZES:
        reads = [refit(feats, y, list(rng.choice(y.index, n, replace=False))) for _ in range(DRAWS)]
        aucs = [official_auc(rows, p) for p, _ in reads]
        curve[n] = {"spearman": float(np.mean([s for _, s in reads])), "auc_mean": float(np.mean(aucs)),
                    "auc_sd": float(np.std(aucs)), "auc_min": float(min(aucs)), "auc_max": float(max(aucs))}
        print(f"noise n={n}: {curve[n]}", flush=True)
    out["curve"] = curve
    return out


def numbers_kept(math_keys: set[str]) -> pd.DataFrame:
    """Per decay-causing variant in DeepSeek's MATH tables: its family and the share of the original's numbers it keeps.

    The tables store only the variants that lowered accuracy, so this describes failures, not every variant drawn."""
    out = []
    for repo, revision in VARIANT_REPOS.items():
        table = pd.read_parquet(hf_hub_download(repo, VARIANT_FILE, repo_type="dataset", revision=revision))
        for original, variants in zip(table["original_problem"], table["permutations_causing_decay"]):
            if problem_key(original) not in math_keys or not isinstance(variants, str):
                continue
            numbers = set(re.findall(r"\d+", original))
            for variant in json.loads(variants):
                if numbers:
                    kept = len(numbers & set(re.findall(r"\d+", variant["question"]))) / len(numbers)
                    out.append((problem_key(original), variant["question"], variant["permutation_type"], kept))
    # The three tables overlap; a variant counts once.
    return pd.DataFrame(out, columns=["problem_key", "question", "family", "kept"]).drop_duplicates(["problem_key", "question"])


def math_levels() -> pd.Series:
    snapshot = Path(snapshot_download(MATH_REPO, repo_type="dataset", revision=MATH_REVISION,
                                      allow_patterns="*/*.parquet"))
    table = pd.concat(pd.read_parquet(path) for path in sorted(snapshot.glob("*/*.parquet")))
    table["problem_key"] = table["problem"].map(problem_key)
    return table.drop_duplicates("problem_key").set_index("problem_key")["level"].str.extract(r"(\d)")[0].astype(float)


def math_check() -> dict:
    math = e65.math_frame(dict(np.load(ROOT / "runs" / "e64" / "artifact.npz")))
    nll = load_scores()
    math = math.merge(nll.loc[nll["scored_model_id"] == DEEPSEEK, ["problem_key", "mean_nll"]], on="problem_key",
                      validate="one_to_one")
    math["template"] = math["problem_key"]
    math["easier"], math["familiar"] = -math["difficulty"], -math["mean_nll"]
    math["level"] = math["problem_key"].map(math_levels())
    assert math["level"].notna().all()
    # Instance-level families include those inside aggregate rows, so a problem labelled only by an aggregate that
    # covered `domain` counts as having it.
    math["domain"] = math["perturbation_types"].map(lambda types: "domain" in list(types))
    untouched = math[~math["domain"]]

    kept = numbers_kept(set(math["problem_key"]))
    by_family = kept.groupby("family")["kept"].agg(variants="size", none_kept=lambda s: float((s == 0).mean()))
    return {
        "rows": len(math),
        "robust_share": {"all": float(math["is_robust"].mean()), "with domain": float(math.loc[math["domain"], "is_robust"].mean()),
                         "without domain": float(untouched["is_robust"].mean())},
        "n": {"with domain": int(math["domain"].sum()), "without domain": int(len(untouched))},
        "auc_all": {s: e64.template_auc(math, s) for s in ("easier", "familiar")},
        "auc_untouched": {s: e64.template_auc(untouched, s) for s in ("easier", "familiar")},
        "domain_given_easier": float(roc_auc_score(math["domain"], math["easier"])),
        "domain_given_familiar": float(roc_auc_score(math["domain"], math["familiar"])),
        "domain_share_by_level": math.groupby("level")["domain"].mean().to_dict(),
        "spearman_level": float(spearmanr(math["difficulty"], math["level"]).statistic),
        "numbers_kept": by_family.to_dict("index"),
    }


def cv_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions of E64's layer-averaged ridge on E64's folds; X is (problems, layers, dims)."""
    oof = np.zeros(len(y))
    for fit, test in KFold(5, shuffle=True, random_state=e64.SEED).split(X):
        oof[test] = e64.layer_predict(e64.layer_fit(X[fit], y[fit]), X[test])
    return oof


def hand_features(text: str) -> dict[str, float]:
    """Interpretable surface features: size, how much is mathematics, and which topic words appear."""
    prose = re.sub(r"\$[^$]*\$", " ", text)
    numbers = [float(n) for n in re.findall(r"\d+", text)[:50]]
    features = {"log_chars": np.log(len(text)), "log_words": np.log1p(len(re.findall(r"[A-Za-z]{3,}", prose))),
                "math_spans": text.count("$") / 2, "digits": len(re.findall(r"\d", text)),
                "log_max_number": np.log1p(max(numbers, default=0.0)),
                "share_math": len("".join(re.findall(r"\$[^$]*\$", text))) / len(text)}
    features.update({name: float(bool(re.search(pattern, text, re.I))) for name, pattern in TOPICS.items()})
    return features


def inside(feats, rows: pd.DataFrame, y: pd.Series) -> dict:
    train = list(y.index)
    yt = y.to_numpy()
    out: dict = {"layers": [], "rank": {}}

    # Single layers, each fitted alone, grouped by relative depth.
    single = []
    for model, (keys, X) in feats.items():
        pos = keys.get_indexer(train)
        for layer in range(X.shape[1]):
            Xl = X[:, [layer]]
            spearman = spearmanr(cv_predict(Xl[pos], yt), yt).statistic
            pred = pd.Series(e64.layer_predict(e64.layer_fit(Xl[pos], yt), Xl), index=keys)
            single.append((model, layer / (X.shape[1] - 1), spearman, pred))
        print(f"layers: {SHORT[model]} done", flush=True)
    single = pd.DataFrame(single, columns=["model", "depth", "spearman", "pred"])
    single["band"] = pd.cut(single["depth"], DEPTHS, include_lowest=True)
    for band, group in single.groupby("band", observed=True):
        pred = {m: pd.concat(list(g["pred"]), axis=1).mean(axis=1) for m, g in group.groupby("model")}
        out["layers"].append({"depth": f"{max(band.left, 0.0):.2f}-{band.right:.2f}", "layers": len(group),
                              "spearman": float(group["spearman"].mean()), "auc": official_auc(rows, pred)})
    best = single.loc[single.groupby("model")["spearman"].idxmax()]
    out["best_layer"] = {"depth": {SHORT[m]: float(d) for m, d in zip(best["model"], best["depth"])},
                         "spearman": float(best["spearman"].mean()),
                         "auc": official_auc(rows, dict(zip(best["model"], best["pred"])))}

    # Rank: the same fit on each layer's top-k principal components (PCA on all E55 problems, no label).
    # Components are nested, so one decomposition per layer serves every k.
    pcs = {}
    for model, (keys, X) in feats.items():
        pcs[model] = np.stack([PCA(max(RANKS), random_state=0).fit_transform(StandardScaler().fit_transform(X[:, layer]))
                               for layer in range(X.shape[1])], axis=1)
    for k in RANKS:
        pred, spearman = {}, []
        for model, (keys, _) in feats.items():
            Z = pcs[model][:, :, :k]
            pos = keys.get_indexer(train)
            spearman.append(spearmanr(cv_predict(Z[pos], yt), yt).statistic)
            pred[model] = pd.Series(e64.layer_predict(e64.layer_fit(Z[pos], yt), Z), index=keys)
        out["rank"][k] = {"spearman": float(np.mean(spearman)), "auc": official_auc(rows, pred)}
        print(f"rank {k}: {out['rank'][k]}", flush=True)

    # The four models' agreement where none was fitted, and what interpretable text reproduces.
    shipped = pd.read_csv(ROOT / "runs" / "e64" / "predicted_difficulty.csv", index_col=0)
    held = shipped.index.difference(train)
    corr = shipped.loc[held].corr("spearman").to_numpy()
    out["agreement"] = {"problems": len(held), "min_pairwise_spearman": float(corr[np.triu_indices(4, 1)].min()),
                        "mean_pairwise_spearman": float(corr[np.triu_indices(4, 1)].mean())}
    out["e64"] = {"auc": official_auc(rows, {m: shipped[m] for m in SHORT})}

    text = pd.read_parquet(ROOT / "data" / "e55" / "instances.parquet").drop_duplicates("problem_key")
    text = text.set_index("problem_key")["problem"].loc[shipped.index]
    H = pd.DataFrame([hand_features(t) for t in text], index=shipped.index)
    pos = shipped.index.get_indexer(train)
    surrogate = make_pipeline(StandardScaler(), RidgeCV(alphas=TEXT_ALPHAS))
    oof = np.zeros(len(yt))
    for fit, test in KFold(5, shuffle=True, random_state=e64.SEED).split(pos):
        oof[test] = surrogate.fit(H.to_numpy()[pos[fit]], yt[fit]).predict(H.to_numpy()[pos[test]])
    surrogate.fit(H.to_numpy()[pos], yt)
    hand = pd.Series(surrogate.predict(H.to_numpy()), index=shipped.index)
    out["hand"] = {"features": H.shape[1], "spearman": float(spearmanr(oof, yt).statistic),
                   "auc": official_auc(rows, {m: hand for m in SHORT}),
                   "coef": dict(sorted(zip(H.columns, map(float, surrogate[-1].coef_)), key=lambda t: -abs(t[1])))}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    V = vectorizer.fit_transform(text.tolist())
    oof = np.zeros(len(yt))
    for fit, test in KFold(5, shuffle=True, random_state=e64.SEED).split(pos):
        oof[test] = RidgeCV(alphas=TEXT_ALPHAS).fit(V[pos[fit]], yt[fit]).predict(V[pos[test]])
    ridge = RidgeCV(alphas=TEXT_ALPHAS).fit(V[pos], yt)
    words = pd.Series(ridge.predict(V), index=shipped.index)
    order, names = np.argsort(ridge.coef_), vectorizer.get_feature_names_out()
    out["tfidf"] = {"spearman": float(spearmanr(oof, yt).statistic), "auc": official_auc(rows, {m: words for m in SHORT}),
                    "spearman_with_e64": float(spearmanr(words.loc[held], shipped.loc[held].mean(axis=1)).statistic),
                    "easy": [str(n) for n in names[order[:15]]], "hard": [str(n) for n in names[order[::-1][:15]]]}

    # How much of E64's own score the hand features explain, and whether the rest still ranks the official rows.
    residual, r2 = {}, {}
    for model in SHORT:
        fit = make_pipeline(StandardScaler(), RidgeCV(alphas=TEXT_ALPHAS)).fit(H.to_numpy(), shipped[model].to_numpy())
        r2[SHORT[model]] = float(fit.score(H.to_numpy(), shipped[model].to_numpy()))
        residual[model] = shipped[model] - fit.predict(H.to_numpy())
    out["explained"] = {"r2": r2, "residual_auc": official_auc(rows, residual)}
    return out


def render(result: dict) -> str:
    def ci(m: dict) -> str:
        return f"{m['auc']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"

    math, inner, noise_ = result["math"], result["inside"], result["noise"]
    lines = ["Exploratory, computed after E64-E66 were read; nothing here is a gate or changes what ships. E64's fit and "
             "arrays are used unchanged unless a row says otherwise. AUCs on the 34 official small-track rows unless "
             "stated; intervals are row bootstraps (2,000 draws, seed 64).", "",
             "### How far the official AUC moves when the method does not",
             "", "E64's fit on a different target or a random subset of its 497 problems; held-out Spearman is on the "
             f"problems left out, mean over {DRAWS} draws (seed {CURVE_SEED}).", "",
             "| fit | held-out Spearman | official AUC |", "| --- | --- | --- |"]
    lines += [f"| all 497, {name} | - | {auc:.3f} |" for name, auc in noise_["targets"].items()]
    lines += [f"| {n} problems | {c['spearman']:.3f} | {c['auc_mean']:.3f} (sd {c['auc_sd']:.3f}, range "
              f"{c['auc_min']:.3f}-{c['auc_max']:.3f}) |" for n, c in noise_["curve"].items()]

    lines += ["", "### Why E64 reverses on DeepSeek's MATH rows", "",
              f"E65's {math['rows']} rows (E64's shipped DeepSeek arithmetic, older label rule). `domain` = the family was "
              "run on the problem, including inside aggregate rows.", "",
              "| rows | n | robust share | AUC, easier = robust | AUC, familiar = robust |", "| --- | --- | --- | --- | --- |",
              f"| all | {math['rows']} | {math['robust_share']['all']:.2f} "
              f"| {ci(math['auc_all']['easier'])} | {ci(math['auc_all']['familiar'])} |",
              f"| `domain` run | {math['n']['with domain']} | {math['robust_share']['with domain']:.2f} | | |",
              f"| **`domain` never run** | {math['n']['without domain']} | {math['robust_share']['without domain']:.2f} "
              f"| **{ci(math['auc_untouched']['easier'])}** | {ci(math['auc_untouched']['familiar'])} |", "",
              f"Which problems got `domain`: AUC {math['domain_given_easier']:.3f} from E64's easier-ness, "
              f"{math['domain_given_familiar']:.3f} from familiarity; share by MATH level "
              + ", ".join(f"L{int(float(k))} {v:.2f}" for k, v in math["domain_share_by_level"].items())
              + f". Spearman of E64's predicted difficulty with the MATH level: {math['spearman_level']:.3f}.", "",
              "Variants that lowered accuracy, as stored in DeepSeek's MATH tables (failures only), that keep none of "
              "the original's numbers: "
              + ", ".join(f"{f} {v['none_kept']:.2f} of {v['variants']}" for f, v in math["numbers_kept"].items()) + "."]

    lines += ["", "### What the ridge reads", "",
              "Single layers, each fitted alone (E64's fit on one layer), grouped by relative depth; CV Spearman on "
              "E64's folds, official AUC of the band's layers averaged.", "",
              "| depth | layers (4 models) | CV Spearman | official AUC |", "| --- | --- | --- | --- |"]
    lines += [f"| {b['depth']} | {b['layers']} | {b['spearman']:.3f} | {b['auc']:.3f} |" for b in inner["layers"]]
    best = inner["best_layer"]
    lines += ["", f"Best single layer per model (depth {', '.join(f'{m} {d:.2f}' for m, d in best['depth'].items())}): "
              f"CV Spearman {best['spearman']:.3f}, official AUC {best['auc']:.3f}. E64 (all layers averaged): "
              f"official AUC {inner['e64']['auc']:.3f}.", "",
              "| top principal components per layer | CV Spearman | official AUC |", "| --- | --- | --- |"]
    lines += [f"| {k} | {r['spearman']:.3f} | {r['auc']:.3f} |" for k, r in inner["rank"].items()]
    agree, hand, tfidf = inner["agreement"], inner["hand"], inner["tfidf"]
    lines += ["", f"The four models' predictions on the {agree['problems']} problems outside the fit: pairwise Spearman "
              f"{agree['min_pairwise_spearman']:.3f} to {agree['mean_pairwise_spearman']:.3f} (min to mean).", "",
              "| text surrogate of the same target | CV Spearman | official AUC |", "| --- | --- | --- |",
              f"| {hand['features']} hand features (size, share of mathematics, topic words) | {hand['spearman']:.3f} | {hand['auc']:.3f} |",
              f"| word 1-2-gram tf-idf, ridge | {tfidf['spearman']:.3f} | {tfidf['auc']:.3f} |", "",
              "Hand-feature weights (standardized): " + ", ".join(f"{k} {v:+.2f}" for k, v in list(hand["coef"].items())[:6]) + ".",
              f"Tf-idf words for short traces: {', '.join(tfidf['easy'])}. For long traces: {', '.join(tfidf['hard'])}. "
              f"Spearman of the tf-idf score with E64's (mean of models) outside the fit: {tfidf['spearman_with_e64']:.3f}.", "",
              "The hand features explain R2 " + ", ".join(f"{m} {v:.2f}" for m, v in inner["explained"]["r2"].items())
              + f" of E64's score over the 548 E55 problems; the rest scores official AUC {inner['explained']['residual_auc']:.3f}."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()
    e64.SEED = 64

    rows = load_rows()
    rows = rows[rows["model_id"].isin(SHORT)].copy()
    feats = load_features()
    result = {"math": math_check()}
    print(json.dumps(result["math"], indent=1, default=str), flush=True)
    result["inside"] = inside(feats, rows, e64.difficulty())
    result["noise"] = noise(feats, rows)

    body = render(result)
    print(body)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", HEADING, body)
        print(f"\nwrote the {HEADING} section of RESULTS.md")


if __name__ == "__main__":
    main()
