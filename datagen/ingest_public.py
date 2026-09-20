"""Consolidate every public aimo-interp table into a small-track training set.

Why this exists: the public *labelled* tables are tiny, but two larger tables
(`aimo-interp-challenge-sample-full`, `augmented-sample-math-full-filtered`) ship the
accuracy-decay columns without a `model_is_robust` column. `scripts/infer_label_rule.py`
showed the organizers' label is exactly `absolute_accuracy_decay == 0` (379/379 agreement),
so those rows can be labelled and reused instead of regenerated on GPUs.

Outputs, under data/small_track/:
  perturbation_rows.parquet  one row per (model, problem, perturbation_type)
  instances.parquet          one row per (model, problem) - the unit we predict at eval
  splits.json                grouped CV folds, grouped by problem so no problem leaks
  summary.json               provenance and counts

Run:  uv run python ../datagen/ingest_public.py
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "small_track"

# Codabench aliases and raw checkpoint names that denote the same weights.
MODEL_CANONICAL = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "meta-llama__Llama-3.1-8B-Instruct_low": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "allenai/Olmo-3-7B-Instruct": "allenai/Olmo-3-7B-Instruct",
    # The Competition Phase passes ids verbatim; train-main-v2 uses the same spellings.
    "Qwen/Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think",
}

# Open-weight models below the 10B track cap: the four the Competition Phase scores, plus two
# that only appear in the older public tables (Llama-3.1-8B-Instruct, Olmo-3-7B-Instruct).
SMALL_MODELS = {
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "meta-llama/Llama-3.1-8B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
}

N_FOLDS = 5
DECAY = "absolute_accuracy_decay"

# The task is only defined where the model already answers the original correctly.
# 295 of 298 organizer-published rows sit at base_accuracy == 1.0, so we mirror that
# curation instead of training on problems the model never solved.
MIN_BASE_ACCURACY = 1.0

# The organizers' own rule, stated in `aimo-interp/train-main-v2` (published 17 Sep 2026):
# robust iff the largest drop is <= 0.10, spurious iff >= 0.25, and no label in between --
# in training, validation and test alike (PROGRESS.md Finding 30). Our earlier `decay <= 0`
# was recovered from older tables that happen to contain nothing between 0 and 0.3, so both
# rules agree on every published row; they differ on rows we derive ourselves.
ROBUST_MAX_DROP = 0.10
SPURIOUS_MIN_DROP = 0.25
AMBIGUOUS_MAX_DECAY = SPURIOUS_MIN_DROP  # kept for importers; the band is (0.10, 0.25)

# That split is also authoritative wherever it covers a (model, problem) pair: it is labelled
# under a different rule from the older derived tables, so the two must not be aggregated.
OFFICIAL_SPLIT_REPO = "aimo-interp/train-main-v2"


def canonical_model(raw: str) -> str | None:
    return MODEL_CANONICAL.get(raw.strip())


def problem_key(text: str) -> str:
    normalized = " ".join((text or "").split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def parse_perturbations(value: object) -> list[str]:
    """Normalize the permutation_type column, which is sometimes a JSON list string."""
    if value is None:
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return sorted({str(v) for v in parsed})
    return [text] if text else []


def load_tables() -> list[tuple[str, list[dict]]]:
    api = HfApi()
    tables: list[tuple[str, list[dict]]] = []
    for dataset in sorted(api.list_datasets(author="aimo-interp"), key=lambda d: d.id):
        repo = dataset.id
        info = api.dataset_info(repo)
        files = [
            s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".parquet")
        ]
        rows: list[dict] = []
        for name in files:
            path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name)
            rows.extend(pq.read_table(path).to_pylist())
        if rows:
            tables.append((repo, rows))
            print(f"  {repo}: {len(rows)} rows")
    return tables


def build_perturbation_rows(tables: list[tuple[str, list[dict]]]) -> list[dict]:
    """Flatten every source table into deduplicated per-perturbation records."""
    best: dict[tuple[str, str, str], dict] = {}
    skipped_models: dict[str, int] = defaultdict(int)

    for repo, rows in tables:
        for row in rows:
            raw_model = row.get("model_id")
            if not raw_model:
                continue
            model = canonical_model(str(raw_model))
            if model is None or model not in SMALL_MODELS:
                skipped_models[str(raw_model)] += 1
                continue
            decay = row.get(DECAY)
            if decay is None:
                # `train-main-v2` (17 Sep) reports `max_drop`, the largest drop across
                # perturbation types, which is what an instance label already aggregates. Rows
                # in the organizers' unlabelled band carry is_robust = None and are skipped.
                decay = row.get("max_drop")
                if decay is None or row.get("is_robust") is None:
                    continue

            published = row.get("model_is_robust")
            if published is None:
                published = row.get("is_robust")
            decay_value = float(decay)
            derived = decay_value <= ROBUST_MAX_DROP
            label = bool(published) if published is not None else derived

            base_accuracy = row.get("base_accuracy")
            # The organizers' own split ships only cases they already curated, without a
            # base-accuracy column; its rows are eligible by construction.
            eligible = True if base_accuracy is None and row.get("max_drop") is not None                 else (base_accuracy or 0.0) >= MIN_BASE_ACCURACY
            clear_cut = decay_value <= ROBUST_MAX_DROP or decay_value >= SPURIOUS_MIN_DROP

            problem = row.get("original_problem") or row.get("problem") or ""
            pkey = problem_key(str(problem))
            perturbations = parse_perturbations(row.get("permutation_type"))
            # An aggregate row covering several perturbations is stored under a
            # composite key so it never overwrites a single-perturbation record.
            pert_key = "+".join(perturbations) if perturbations else "UNSPECIFIED"

            record = {
                "model_id": model,
                "problem_key": pkey,
                "problem": str(problem),
                "problem_id": str(row.get("problem_id") or ""),
                "dataset_id": str(row.get("dataset_id") or repo.split("/")[-1]),
                "perturbation_key": pert_key,
                "perturbation_types": perturbations,
                "is_perturbation_aggregate": len(perturbations) != 1,
                "base_accuracy": row.get("base_accuracy"),
                "permuted_accuracy": row.get("permuted_accuracy"),
                "absolute_accuracy_decay": float(decay),
                "relative_accuracy_decay": row.get("relative_accuracy_decay"),
                "n_base_predictions": row.get("n_base_predictions"),
                "n_permuted_predictions": row.get("n_permuted_predictions"),
                "n_detrimental_permutations": row.get("n_detrimental_permutations"),
                "is_robust": label,
                "label_source": "published" if published is not None else "derived",
                "is_eligible": eligible,
                "is_clear_cut": clear_cut,
                "source_repo": repo,
            }

            key = (model, pkey, pert_key)
            previous = best.get(key)
            # Prefer published labels, then the better-supported measurement.
            if previous is None:
                best[key] = record
                continue
            prefer_new = (
                previous["label_source"] == "derived" and record["label_source"] == "published"
            ) or (
                previous["label_source"] == record["label_source"]
                and (record["n_permuted_predictions"] or 0) > (previous["n_permuted_predictions"] or 0)
            )
            if prefer_new:
                best[key] = record

    if skipped_models:
        print("\n  skipped non-small models:")
        for model, count in sorted(skipped_models.items(), key=lambda kv: -kv[1]):
            print(f"    {count:5d}  {model}")
    return list(best.values())


def build_instances(rows: list[dict]) -> list[dict]:
    """Collapse to the evaluation unit: one (model, problem) prediction.

    Mirrors the aggregate semantics of the organizers' `*-agg` tables: an instance is
    robust only when no observed perturbation degraded accuracy. Only eligible rows
    (model already correct on the original) contribute to the label.
    """
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["problem_key"])].append(row)

    instances = []
    for (model, pkey), group in grouped.items():
        eligible = [r for r in group if r["is_eligible"]]
        # The organizers' own split wins outright where it covers a pair. It is labelled under
        # a different rule (`max_drop`, three-valued) from the older derived tables, and mixing
        # the two silently overrode two official DeepSeek labels with older measurements of the
        # same problems: `cd15d0c0…` was flipped robust -> spurious and `49981d06…` was pushed
        # into the ambiguous band and dropped from training.
        official = [r for r in group if r["source_repo"] == OFFICIAL_SPLIT_REPO]
        scored = official or eligible or group
        max_decay = max(r["absolute_accuracy_decay"] for r in scored)
        perturbations = sorted({p for r in scored for p in r["perturbation_types"]})
        published = [r for r in scored if r["label_source"] == "published"]
        instances.append(
            {
                "model_id": model,
                "problem_key": pkey,
                "problem": scored[0]["problem"],
                "problem_id": scored[0]["problem_id"],
                "dataset_id": scored[0]["dataset_id"],
                "is_robust": max_decay <= ROBUST_MAX_DROP,
                "max_absolute_accuracy_decay": max_decay,
                "n_perturbation_rows": len(scored),
                "perturbation_types": perturbations,
                "has_published_label": bool(published),
                "is_eligible": bool(eligible),
                "is_clear_cut": max_decay <= ROBUST_MAX_DROP or max_decay >= SPURIOUS_MIN_DROP,
                "source_repos": sorted({r["source_repo"] for r in scored}),
            }
        )
    return instances


def make_splits(instances: list[dict]) -> dict:
    """Grouped folds: every problem lives in exactly one fold, so no text leaks."""
    keys = sorted({i["problem_key"] for i in instances})
    folds: dict[str, int] = {}
    for key in keys:
        bucket = int(hashlib.sha1(key.encode()).hexdigest(), 16) % N_FOLDS
        folds[key] = bucket
    per_fold = defaultdict(lambda: {"instances": 0, "robust": 0})
    for inst in instances:
        fold = folds[inst["problem_key"]]
        per_fold[fold]["instances"] += 1
        per_fold[fold]["robust"] += int(inst["is_robust"])
    return {
        "n_folds": N_FOLDS,
        "group_by": "problem_key",
        "problem_fold": folds,
        "fold_stats": {str(k): v for k, v in sorted(per_fold.items())},
    }


def write_parquet(rows: list[dict], path: Path) -> None:
    normalized = [dict(r) for r in rows]
    for row in normalized:
        for field in ("perturbation_types", "source_repos"):
            if field in row and row[field] is not None:
                row[field] = list(row[field])
    pq.write_table(pa.Table.from_pylist(normalized), path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("downloading public tables:")
    tables = load_tables()

    print("\nflattening to perturbation rows:")
    rows = build_perturbation_rows(tables)
    print(f"\n  kept {len(rows)} deduplicated perturbation rows")

    instances = build_instances(rows)
    print(f"  collapsed to {len(instances)} (model, problem) instances")

    # Training set mirrors the organizers' curation: model correct on the original,
    # and the robust/spurious verdict unambiguous.
    train = [i for i in instances if i["is_eligible"] and i["is_clear_cut"]]
    print(
        f"  train-ready: {len(train)} instances "
        f"(dropped {sum(1 for i in instances if not i['is_eligible'])} ineligible, "
        f"{sum(1 for i in instances if i['is_eligible'] and not i['is_clear_cut'])} ambiguous)"
    )

    splits = make_splits(train)

    write_parquet(rows, OUT_DIR / "perturbation_rows.parquet")
    write_parquet(instances, OUT_DIR / "instances.parquet")
    write_parquet(train, OUT_DIR / "instances_train.parquet")
    (OUT_DIR / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    per_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "instances": 0,
            "robust": 0,
            "spurious": 0,
            "perturbation_rows": 0,
            "train_instances": 0,
            "train_robust": 0,
        }
    )
    for row in rows:
        per_model[row["model_id"]]["perturbation_rows"] += 1
    for inst in instances:
        stats = per_model[inst["model_id"]]
        stats["instances"] += 1
        stats["robust" if inst["is_robust"] else "spurious"] += 1
    for inst in train:
        stats = per_model[inst["model_id"]]
        stats["train_instances"] += 1
        stats["train_robust"] += int(inst["is_robust"])

    label_sources = defaultdict(int)
    for row in rows:
        label_sources[row["label_source"]] += 1

    summary = {
        "label_rule": f"is_robust == (max drop <= {ROBUST_MAX_DROP}); spurious iff >= {SPURIOUS_MIN_DROP}",
        "label_rule_agreement": "379/379 on published labels (scripts/infer_label_rule.py)",
        "eligibility_rule": f"base_accuracy >= {MIN_BASE_ACCURACY}",
        "ambiguity_rule": f"{ROBUST_MAX_DROP} < drop < {SPURIOUS_MIN_DROP} carries no label (organizers' rule)",
        "source_repos": [repo for repo, _ in tables],
        "perturbation_rows": len(rows),
        "instances": len(instances),
        "train_instances": len(train),
        "label_sources": dict(label_sources),
        "per_model": {k: dict(v) for k, v in per_model.items()},
        "fold_stats": splits["fold_stats"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nper-model totals:")
    for model, stats in sorted(per_model.items(), key=lambda kv: -kv[1]["instances"]):
        print(
            f"  {model}\n"
            f"      all:   instances={stats['instances']:4d} "
            f"robust={stats['robust']:4d} spurious={stats['spurious']:4d} "
            f"perturbation_rows={stats['perturbation_rows']:4d}\n"
            f"      train: instances={stats['train_instances']:4d} "
            f"robust={stats['train_robust']:4d} "
            f"spurious={stats['train_instances'] - stats['train_robust']:4d}"
        )
    print(f"\nlabel sources: {dict(label_sources)}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
