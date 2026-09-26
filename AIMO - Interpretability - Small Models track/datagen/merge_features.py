"""Merge per-shard extraction outputs into a single training matrix.

Consumes the `shard_XX_of_YY/` directories produced by `extract_features.py` (downloaded
from each Kaggle account) and emits one consolidated bundle:

    features.npz   X_<pooling> arrays of shape (n_instances, n_layers, hidden_dim)
    records.parquet  per-instance labels, uncertainty scalars, and fold ids

Verifies that shards agree on dtype, layer count, and prompt hash before merging, so a
misconfigured account cannot silently poison the training set.

Run:  python datagen/merge_features.py --features-dir features/deepseek-8b
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ROOT / "data" / "small_track" / "splits.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    shard_dirs = sorted(p for p in args.features_dir.glob("shard_*_of_*") if p.is_dir())
    if not shard_dirs:
        raise SystemExit(f"no shard_*_of_* directories under {args.features_dir}")

    metas: list[dict] = []
    records: dict[str, dict] = {}
    for shard in shard_dirs:
        meta_path = shard / "metadata.json"
        if meta_path.is_file():
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        else:
            print(f"  warning: {shard.name} has no metadata.json (incomplete shard?)")

        records_path = shard / "records.jsonl"
        if not records_path.is_file():
            print(f"  warning: {shard.name} has no records.jsonl, skipping")
            continue
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                # A resumed shard can re-append a row; last write wins.
                records[record["problem_key"]] = record
        print(f"  {shard.name}: {len(list((shard / 'vectors').glob('*.npz')))} vectors")

    # Cross-shard consistency: differing dtype or prompt makes features incomparable.
    for field in ("dtype", "n_layers", "hidden_dim", "system_prompt_sha1", "model_id"):
        values = {json.dumps(m.get(field)) for m in metas}
        if len(values) > 1:
            raise SystemExit(f"shards disagree on {field}: {sorted(values)}")
    if metas:
        print(
            f"\nconsistent across {len(metas)} shards: model={metas[0]['model_id']} "
            f"dtype={metas[0]['dtype']} layers={metas[0]['n_layers']} "
            f"hidden={metas[0]['hidden_dim']}"
        )

    keys = sorted(records)
    if not keys:
        raise SystemExit("no records found")

    poolings = metas[0]["poolings"] if metas else ["last", "mean", "max"]
    stacks: dict[str, list[np.ndarray]] = {p: [] for p in poolings}
    kept: list[str] = []
    for key in keys:
        path = next(
            (s / "vectors" / f"{key}.npz" for s in shard_dirs if (s / "vectors" / f"{key}.npz").is_file()),
            None,
        )
        if path is None:
            print(f"  warning: record {key} has no vector file, dropped")
            continue
        with np.load(path) as payload:
            for pooling in poolings:
                stacks[pooling].append(payload[pooling])
        kept.append(key)

    fold_map: dict[str, int] = {}
    if args.splits.is_file():
        fold_map = json.loads(args.splits.read_text(encoding="utf-8"))["problem_fold"]

    table_rows = []
    for key in kept:
        row = dict(records[key])
        row["fold"] = fold_map.get(key, -1)
        table_rows.append(row)

    out_dir = args.out_dir or args.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "features.npz",
        **{f"X_{p}": np.stack(stacks[p]) for p in poolings},
        problem_keys=np.asarray(kept),
    )
    pq.write_table(pa.Table.from_pylist(table_rows), out_dir / "records.parquet")

    labels = Counter(r["is_robust"] for r in table_rows)
    folds = Counter(r["fold"] for r in table_rows)
    shape = np.stack(stacks[poolings[0]]).shape
    print(f"\nmerged {len(kept)} instances")
    print(f"  X_{poolings[0]} shape: {shape}  (instances, layers, hidden_dim)")
    print(f"  labels: robust={labels.get(True, 0)} spurious={labels.get(False, 0)}")
    print(f"  folds: {dict(sorted(folds.items()))}")
    print(f"  wrote {out_dir / 'features.npz'} and {out_dir / 'records.parquet'}")


if __name__ == "__main__":
    main()
