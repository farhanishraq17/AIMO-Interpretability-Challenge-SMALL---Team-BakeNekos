"""Profile every public aimo-interp dataset: schema, size, and model coverage.

Writes a machine-readable summary to data/dataset_profiles.json so the data plan
can be grounded in measured row counts instead of guesses.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dataset_profiles.json"

SMALL_MODEL_HINTS = ("qwen3-8b", "olmo", "7b", "8b")


def profile(repo: str) -> dict:
    api = HfApi()
    info = api.dataset_info(repo)
    parquets = [
        s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".parquet")
    ]
    result: dict = {"repo": repo, "sha": info.sha, "parquet_files": parquets}

    total_rows = 0
    schema: list[str] = []
    model_counts: Counter = Counter()
    perturbation_counts: Counter = Counter()
    label_counts: Counter = Counter()
    problem_ids: set[str] = set()

    for name in parquets:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name)
        table = pq.read_table(path)
        total_rows += table.num_rows
        if not schema:
            schema = [f"{f.name}:{f.type}" for f in table.schema]
        columns = set(table.column_names)
        if "model_id" in columns:
            model_counts.update(table.column("model_id").to_pylist())
        for candidate in ("permutation_type", "perturbation_type"):
            if candidate in columns:
                perturbation_counts.update(
                    str(v) for v in table.column(candidate).to_pylist()
                )
                break
        for candidate in ("model_is_robust", "is_robust"):
            if candidate in columns:
                label_counts.update(str(v) for v in table.column(candidate).to_pylist())
                break
        if "problem_id" in columns:
            problem_ids.update(str(v) for v in table.column("problem_id").to_pylist())

    result.update(
        rows=total_rows,
        schema=schema,
        unique_problems=len(problem_ids),
        models=dict(model_counts.most_common()),
        perturbation_types=dict(perturbation_counts.most_common()),
        labels=dict(label_counts),
    )
    result["small_model_rows"] = {
        model: count
        for model, count in model_counts.items()
        if any(hint in model.lower() for hint in SMALL_MODEL_HINTS)
    }
    return result


def main() -> None:
    api = HfApi()
    repos = sorted(d.id for d in api.list_datasets(author="aimo-interp"))
    profiles = []
    for repo in repos:
        print("=" * 78)
        print(repo)
        try:
            data = profile(repo)
        except Exception as exc:  # noqa: BLE001 - keep profiling the rest
            print(f"  FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
            continue
        profiles.append(data)
        print(f"  rows={data['rows']}  unique_problems={data['unique_problems']}")
        print(f"  labels={data['labels']}")
        print("  schema:")
        for field in data["schema"]:
            print(f"    {field}")
        print("  models:")
        for model, count in data["models"].items():
            mark = " <-- SMALL" if model in data["small_model_rows"] else ""
            print(f"    {count:6d}  {model}{mark}")
        if data["perturbation_types"]:
            print("  perturbation types:")
            for kind, count in data["perturbation_types"].items():
                print(f"    {count:6d}  {kind[:70]}")
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
