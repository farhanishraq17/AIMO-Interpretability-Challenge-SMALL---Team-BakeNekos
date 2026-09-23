"""Build the pre-registered E28 familiarity candidates for the Small Models Track."""

from __future__ import annotations

import sys
from pathlib import Path

SUBMISSION = Path(__file__).resolve().parent
ROOT = SUBMISSION.parent
DIST = ROOT / "dist"
SOURCE = SUBMISSION / "e28-familiarity" / "solution.py"
sys.path.insert(0, str(ROOT / "getting-started" / "scripts"))

from build import archive  # noqa: E402


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tokenizer_check = (ROOT / "datagen" / "tokenizer_check.py").read_bytes()
    variants = {
        "e28a-suffix-nll": "suffix_nll",
        "e28b-mean-suffix": "mean_suffix",
        "e28c-min-k-20": "min_k_20",
        "e28d-min-k-gap": "min_k_gap",
    }
    for name, score_mode in variants.items():
        variant = source.replace('SCORE_MODE = "suffix_nll"', f'SCORE_MODE = "{score_mode}"')
        assert f'SCORE_MODE = "{score_mode}"' in variant
        out = archive(DIST / f"solution-{name}-small.zip", {
            "solution.py": variant.encode(),
            "tokenizer_check.py": tokenizer_check,
            "small.txt": b"",
        })
        print(f"wrote {out.name} ({out.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
