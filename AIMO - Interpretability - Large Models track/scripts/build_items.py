"""Build the problem list the Kaggle model panel samples (data/panel/items.jsonl).

Two sources, both official:

* `train-main-v2`: the 36 distinct Main-Track training problems (labels for all five evaluated
  models, gpt-oss-120b at three efforts). Originals only: their perturbations are not public.
* `aimo-interp-challenge-sample-full`: 10 AIMO-2025 reference problems. Its
  `permutations_causing_decay` column carries the **official perturbed question texts** with the
  per-variant accuracy each model got. Running our panel on the same texts measures how well a
  proxy reproduces official outcomes (e.g. gpt-oss-20b vs `gpt-oss-120b:low`, and our DeepSeek
  pipeline vs the official `qwen3-8b:low` rows, which are the same checkpoint).

Originals get k=4 samples; each official variant gets k=1, as in the organisers' protocol
(10 variants x 1 prediction per family). Originals sort first so a slow model covers them before
any variant. The official per-variant outcomes go to data/panel/official_variants.jsonl for scoring.

Run: getting-started/.venv/Scripts/python.exe "AIMO - Interpretability - Large Models track/scripts/build_items.py"
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data" / "hf"
OUT = ROOT / "data" / "panel"


def key(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode()).hexdigest()[:12]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items: dict[str, dict] = {}
    train = pd.read_parquet(HF / "train-main-v2/data/train-00000-of-00001.parquet")
    for text in sorted(train.problem.unique()):
        pid = f"tm2:{key(text)}"
        items[pid] = dict(item_id=pid, problem_id=pid, family="original", text=text, k=4, source="train-main-v2")

    full = pd.read_parquet(HF / "aimo-interp-challenge-sample-full/data/validation-00000-of-00001.parquet")
    outcomes = []
    for row in full.itertuples():
        pid = f"ref:{row.problem_id}"
        items.setdefault(pid, dict(item_id=pid, problem_id=pid, family="original", text=row.original_problem,
                                   k=4, source="aimo-interp-challenge-sample-full"))
        for variant in json.loads(row.permutations_causing_decay or "[]"):
            vid = f"{pid}:{variant['permutation_type']}:{key(variant['question'])}"
            items.setdefault(vid, dict(item_id=vid, problem_id=pid, family=variant["permutation_type"],
                                       text=variant["question"], k=1, source="aimo-interp-challenge-sample-full"))
            outcomes.append(dict(item_id=vid, model_id=row.model_id, variant_accuracy=variant["variant_accuracy"],
                                 n_predictions=variant["n_predictions"], permutation_source=variant["permutation_source"],
                                 row_base_accuracy=row.base_accuracy, row_permuted_accuracy=row.permuted_accuracy))

    ordered = sorted(items.values(), key=lambda it: (it["family"] != "original", it["item_id"]))
    (OUT / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in ordered), encoding="utf-8")
    (OUT / "official_variants.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outcomes), encoding="utf-8")
    frame = pd.DataFrame(ordered)
    print(frame.groupby(["source", "family"]).size().to_string())
    print(f"{len(ordered)} items, {int(frame.k.sum())} completions per model-effort; {len(outcomes)} official variant outcomes")


if __name__ == "__main__":
    main()
