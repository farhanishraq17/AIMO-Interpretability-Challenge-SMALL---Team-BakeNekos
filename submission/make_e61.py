"""Build the two E61 ZIPs: familiarity as an absolute per-problem threshold.

`train/eval_e61.py` computes both rules' thresholds under `protocols/E61_PROTOCOL.md` and writes them to
runs/e61/thresholds.json; this writes them into `e61-familiarity/solution.py` as literals, so the ZIP
holds no data file, and archives it with `tokenizer_check.py` and `small.txt`:

* solution-e61a-amc-median-small.zip: E54's frozen 75-problem AMC median, strict comparison (control)
* solution-e61b-quantile-small.zip: the p_m quantile of E55's 548 problems (the candidate)
* solution-e63-official-quantile-small.zip: the same quantiles over the 36 `train-main-v2` problems, Qwen at its
  constant (`protocols/E63_PROTOCOL.md`; thresholds from `train/eval_e63.py`)

Deterministic: the starter kit's `archive` fixes timestamps and entry order.

Run:  python submission/make_e61.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SUBMISSION = Path(__file__).resolve().parent
ROOT = SUBMISSION.parent
sys.path.insert(0, str(ROOT / "getting-started" / "scripts"))

from build import archive  # noqa: E402

DIST = ROOT / "dist"
# rule -> (ZIP name, strict comparison, thresholds file, models with a cutoff; the rest return their constant)
VARIANTS = {"E61a": ("e61a-amc-median", True, ROOT / "runs" / "e61" / "thresholds.json", 4),
            "E61b": ("e61b-quantile", False, ROOT / "runs" / "e61" / "thresholds.json", 4),
            "E63": ("e63-official-quantile", False, ROOT / "runs" / "e63" / "thresholds.json", 3)}


def main() -> None:
    source = (SUBMISSION / "e61-familiarity" / "solution.py").read_text(encoding="utf-8")
    tokenizer_check = (ROOT / "datagen" / "tokenizer_check.py").read_bytes()
    for rule, (name, strict, path, n_models) in VARIANTS.items():
        rules = json.loads(path.read_text(encoding="utf-8"))
        literal = "{" + ", ".join(f"{model!r}: {value!r}" for model, value in sorted(rules[rule].items())) + "}"
        variant = (source.replace("THRESHOLDS: dict[str, float] = {}", f"THRESHOLDS: dict[str, float] = {literal}")
                         .replace("STRICT = False", f"STRICT = {strict}"))
        assert literal in variant and f"STRICT = {strict}" in variant and len(rules[rule]) == n_models, rule
        entries = {"solution.py": variant.encode(), "tokenizer_check.py": tokenizer_check, "small.txt": b""}
        out = archive(DIST / f"solution-{name}-small.zip", entries)
        print(f"wrote {out.name} ({out.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
