"""M11a: the first Main-Track ZIP (PLAN.md §4.12).

The Main Track scores five models. Four are the small track's checkpoints, where `e27k` already
scores 0.86 on that track's leaderboard, so they ship its policy unchanged: load the model being
scored, one forward pass over each bare problem, call the more familiar half robust, with Skywork
and Olmo calling one fewer robust than the median (small-track Findings 60, 61).

The fifth model, `openai/gpt-oss-120b`, gets the constant `True` at every effort. Two reasons:

* It is the best constant in the official data at every effort - robust shares 0.714 (low),
  0.667 (medium) and 0.867 (high), so always-robust scores 0.732 over all 71 labelled rows (LF-1).
* It is also the fastest answer. Listing only the four small models in `SPLIT_MODELS` means the
  62 GiB checkpoint is never loaded, which matters because ties break on runtime and the 3600 s
  budget covers all five models together.

No GPU is needed to build this: the policy ships no fitted estimator, only source. The forward-pass
path is the same source that is live on the small-track leaderboard; what is new here is the
gpt-oss branch and the *absence* of `small.txt`, and neither touches a GPU.

**`small.txt` must not be in a Main-Track ZIP** - with it, the score is 0.00 (small-track
Finding 21). The assertions at the end of `main` fail the build rather than let that ship.

Run:  python submission/make_m11a.py       # -> dist/solution-m11a-constant-gptoss-main.zip
"""

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SMALL_TRACK = ROOT.parent / "AIMO - Interpretability - Small Models track"
POLICY = SMALL_TRACK / "submission" / "e27d-familiarity" / "solution.py"
TOKENIZER_CHECK = SMALL_TRACK / "datagen" / "tokenizer_check.py"
DIST = ROOT / "dist"
OUT = DIST / "solution-m11a-constant-gptoss-main.zip"

sys.path.insert(0, str(SMALL_TRACK / "getting-started" / "scripts"))
from build import archive  # noqa: E402  (deterministic ZIP: fixed mtimes, sorted entries)

SMALL_MODELS = (
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
)
GPT_OSS = "openai/gpt-oss-120b"
# e27k's counts, and the constants its fallback path uses, plus gpt-oss-120b as always-robust.
TILT = '{"Skywork/Skywork-OR1-Math-7B": -1, "allenai/Olmo-3-7B-Think": -1}'
FALLBACK = ('{"Qwen/Qwen3.5-4B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", '
            f'"{GPT_OSS}"}}')

EDITS = (
    # e27k: the familiar half is robust, with per-model robust counts.
    ('ROBUST_HALF = "unfamiliar"', 'ROBUST_HALF = "familiar"'),
    ("TILT = {}", f"TILT = {TILT}"),
    # Main Track: the four small models take the split; gpt-oss-120b takes its constant without
    # ever being loaded, because `uses_split` is false for it.
    ('SPLIT_MODELS = "all"', f'SPLIT_MODELS = "{",".join(SMALL_MODELS)}"'),
    ('FALLBACK_ROBUST = {"Qwen/Qwen3.5-4B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}',
     f"FALLBACK_ROBUST = {FALLBACK}"),
)


def main() -> None:
    source = POLICY.read_text(encoding="utf-8")
    for old, new in EDITS:
        if source.count(old) != 1:
            raise SystemExit(f"expected exactly one {old!r} in {POLICY.name}; the policy moved")
        source = source.replace(old, new)

    entries = {"solution.py": source.encode(), "tokenizer_check.py": TOKENIZER_CHECK.read_bytes()}
    out = archive(OUT, entries)

    names = ZipFile(out).namelist()
    assert names == ["solution.py", "tokenizer_check.py"], names
    assert not any(name.lower() == "small.txt" for name in names), "Main-Track ZIP must not mark the small track"
    # The four models must reach the split and gpt-oss must not, or this is not M11a.
    assert all(model in source for model in SMALL_MODELS) and f'"{GPT_OSS}"' in source
    assert f'SPLIT_MODELS = "{",".join(SMALL_MODELS)}"' in source and GPT_OSS not in source.split("FALLBACK_ROBUST")[0]

    print(f"wrote {out} ({out.stat().st_size / 1e3:.1f} KB)")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
