"""E39: is the incumbent stable, and how long does it take? A GPU audit of the shipped e27k ZIP.

`e27k` scores 0.86 and was never GPU-checked - `check_e27d.py` audited `e27d`, its inverted
sibling, before the tilts existed. Two things are unmeasured and both matter before the 27 Oct
freeze:

1. **Runtime.** Ties break on average runtime, and the 3600 s budget covers every model batch
   together. Nobody has timed a load-plus-forward-pass sweep for all four checkpoints.
2. **Stability of the ranking.** `e27k` splits each batch at its median NLL, so a prediction
   only changes if a problem crosses the split. Two conditions differ between how the ranking
   was measured and how it will run, and neither has been tested:
   - **dtype.** Features were extracted on T4s, where bf16 is emulated (Finding 13); evaluation
     runs bf16 on a Blackwell card. This measures fp16 against bf16 on the same T4, which bounds
     how precision-sensitive the ranking is without reproducing the evaluation card exactly.
   - **prompt form.** The shipped policy scores the *bare* problem text. Nothing checked whether
     the ranking survives the model's own chat template, which is the form the probe baseline and
     every other repo pipeline uses.

Truncation is deliberately not an arm: the longest labelled problem is 641 tokens against the
solution's 2048 cap, so it never fires.

Reported per model: seconds, whether two identical calls return identical predictions, and for
each arm the Spearman correlation of the NLL ranking against the shipped baseline plus the number
of predictions that flip. Flips are what matters; a correlation of 0.99 that moves one problem
across the median still changes a case.

Run (on Kaggle, via kaggle_launch):  python submission/audit_e27k.py --models Qwen/Qwen3.5-4B ...
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "dist" / "solution-e27k-tilted-small.zip"
OFFICIAL = ROOT / "data" / "official_check"
TRAIN_MAIN = ROOT / "data" / "train_main" / "rows.parquet"


def nll(model_id: str, problems: list[str], dtype_name: str, chat_template: bool) -> list[float]:
    """Mean token NLL, with the two knobs the shipped `problem_nll` fixes.

    Kept deliberately close to the shipped function so `baseline` below can assert the two agree;
    a drifting copy would make every arm meaningless.
    """
    import torch
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(Path("/tmp/e27k")))
    from tokenizer_check import load_checked_tokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    tokenizer, _ = load_checked_tokenizer(model_id, local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="auto", local_files_only=False
    ).eval()
    try:
        scores = []
        with torch.inference_mode():
            for problem in problems:
                text = problem
                if chat_template:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True
                    )
                ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
                ids = ids.to(model.device)
                scores.append(float(model(input_ids=ids, labels=ids, use_cache=False).loss))
        return scores
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def spearman(a: list[float], b: list[float]) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/e39_audit.json"))
    args = parser.parse_args()

    unpacked = Path("/tmp/e27k")
    zipfile.ZipFile(ZIP).extractall(unpacked)
    sys.path.insert(0, str(unpacked))
    import solution  # noqa: E402  (the packaged file, not the repo copy)

    assert solution.ROBUST_HALF == "familiar" and solution.TILT, "not the e27k ZIP"

    # Every model scores all 36 official problems, not just its own labelled rows: Skywork has 3
    # of those, and a 3-problem batch cannot show whether a ranking is stable. A 36-problem batch
    # is also closer to what the harness passes. Labels are only used for the agreement column.
    rows = pd.read_parquet(TRAIN_MAIN).drop_duplicates("problem_key")
    problems = list(rows["problem"])
    keys = list(rows["problem_key"])
    labels = pd.read_parquet(OFFICIAL / "official_labels.parquet")
    report = {"tilt": solution.TILT, "problems": len(problems), "models": {}}

    for model_id in args.models:
        snapshot_download(model_id)
        labelled = labels[labels["model_id"] == model_id].set_index("problem_key")["is_robust"]

        started = time.time()
        first = solution.are_robust(model_id, problems)
        seconds = time.time() - started
        second = solution.are_robust(model_id, problems)  # same call twice: is it deterministic?
        base_nll = solution.problem_nll(model_id, problems)  # raises rather than falling back

        scored = [(p, bool(labelled[k])) for p, k in zip(first, keys) if k in labelled.index]
        entry = {
            "problems": len(problems),
            "seconds_per_batch": seconds,
            "seconds_per_problem": seconds / max(len(problems), 1),
            "deterministic": first == second,
            "native_bools": all(type(p) is bool for p in first),
            "predictions": first,
            "labelled_rows": len(scored),
            # On this model's own labelled subset only, and read as orientation: the split is over
            # a 36-problem batch, not the batch the leaderboard scores.
            "agreement_with_official": (sum(p == o for p, o in scored) / len(scored)) if scored else None,
            "arms": {},
        }

        # The baseline arm reproduces the shipped path with this file's own loader, so any
        # disagreement means the copy drifted and the other arms cannot be trusted.
        check = nll(model_id, problems, "bfloat16", chat_template=False)
        entry["baseline_matches_shipped"] = spearman(check, base_nll) > 0.999

        for name, dtype_name, template in (("fp16", "float16", False), ("chat_template", "bfloat16", True)):
            # One arm failing (a tokenizer without a chat template, an fp16 overflow) must not
            # cost the whole kernel: four models share it and each costs a model load.
            try:
                scores = nll(model_id, problems, dtype_name, template)
            except Exception as exc:
                entry["arms"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {name}: FAILED {type(exc).__name__}: {exc}", flush=True)
                continue
            predictions = solution.split_half(scores, model_id)
            entry["arms"][name] = {
                "spearman_vs_baseline": spearman(scores, base_nll),
                "flips": sum(p != q for p, q in zip(predictions, first)),
                "predictions": predictions,
            }
            print(f"  {name}: spearman {entry['arms'][name]['spearman_vs_baseline']:.4f}, "
                  f"{entry['arms'][name]['flips']} of {len(problems)} predictions flip", flush=True)

        report["models"][model_id] = entry
        print(f"{model_id}: {len(problems)} problems in {seconds:.1f}s "
              f"({entry['seconds_per_problem']:.2f}s each), deterministic={entry['deterministic']}, "
              f"baseline_matches_shipped={entry['baseline_matches_shipped']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
