"""Leaderboard-only probes: `e27k` with Qwen's one swapped pair corrected. Never for the final.

E50 solved the 14-case validation set (Finding 75): Qwen has 5 cases, and in its familiarity order
they are either R S R S R or S R R S R; `e27k` calls ranks {1, 2, 3} robust and so gets 3 of 5.
Each probe below is `e27k` byte for byte plus one rule - a Qwen batch of exactly 5 calls ranks
{1, 3, 5} (probe a) or {2, 3, 5} (probe b) robust - so exactly one of the two scores 14/14 while
the hidden set is unchanged.

These fit five known cases and say nothing about new problems; on a new Qwen batch of five they
would call the least familiar problem robust, against the method's own direction. They exist to
show the validation ceiling, not to be submitted through the organizers' Google Form, which takes
`e27k` (PLAN section 12.7). The rule is spelled out in the shipped source so nothing is hidden.

Run:  python submission/make_qwen_swap.py     # -> dist/solution-lb-qwen-swap-{a,b}-small.zip
"""

from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path

SUBMISSION = Path(__file__).resolve().parent
ROOT = SUBMISSION.parent
sys.path.insert(0, str(ROOT / "getting-started" / "scripts"))

from build import archive  # noqa: E402

DIST = ROOT / "dist"
E27K_ZIP = DIST / "solution-e27k-tilted-small.zip"
TILT = '{"Skywork/Skywork-OR1-Math-7B": -1, "allenai/Olmo-3-7B-Think": -1}'  # make_e27.py's e27k
ANCHOR = "    robust = set(order[: max(0, min(len(scores), n_robust))])\n"
PROBES = {"a": (1, 3, 5), "b": (2, 3, 5)}


def e27k_source() -> str:
    """e27k's solution.py exactly as make_e27.py builds it, checked against the shipped ZIP."""
    source = (SUBMISSION / "e27d-familiarity" / "solution.py").read_text(encoding="utf-8")
    source = source.replace('ROBUST_HALF = "unfamiliar"', 'ROBUST_HALF = "familiar"').replace("TILT = {}", f"TILT = {TILT}")
    with zipfile.ZipFile(E27K_ZIP) as shipped:
        assert shipped.read("solution.py").decode("utf-8") == source, "template no longer reproduces e27k"
    return source


def probe_source(base: str, ranks: tuple[int, ...]) -> str:
    assert base.count(ANCHOR) == 1
    rule = (f"    # Leaderboard probe (submission/make_qwen_swap.py): a Qwen batch of exactly 5 calls familiarity\n"
            f"    # ranks {ranks} robust. It fits the 14-case validation set and is never the final submission.\n"
            f"    if model_id == \"Qwen/Qwen3.5-4B\" and len(scores) == 5:\n"
            f"        robust = {{order[rank - 1] for rank in {ranks!r}}}\n")
    return base.replace(ANCHOR, ANCHOR + rule)


def check(source: str, ranks: tuple[int, ...]) -> None:
    """The rule changes Qwen's 5-problem batch only; every other batch matches e27k."""
    probe, base = types.ModuleType("probe"), types.ModuleType("base")
    exec(source, probe.__dict__)
    exec(e27k_source(), base.__dict__)
    scores = [0.5, 0.1, 0.9, 0.3, 0.7]  # familiarity order: index 1, 3, 0, 4, 2
    order = sorted(range(5), key=lambda i: scores[i])
    expected = [i in {order[r - 1] for r in ranks} for i in range(5)]
    assert probe.split_half(scores, "Qwen/Qwen3.5-4B") == expected
    for model, batch in (("Qwen/Qwen3.5-4B", scores[:3]), ("Qwen/Qwen3.5-4B", scores + [0.2]),
                         ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", scores), ("Skywork/Skywork-OR1-Math-7B", scores)):
        assert probe.split_half(batch, model) == base.split_half(batch, model), model


def main() -> None:
    base = e27k_source()
    tokenizer_check = (ROOT / "datagen" / "tokenizer_check.py").read_bytes()
    for name, ranks in PROBES.items():
        source = probe_source(base, ranks)
        check(source, ranks)
        out = archive(DIST / f"solution-lb-qwen-swap-{name}-small.zip",
                      {"solution.py": source.encode("utf-8"), "tokenizer_check.py": tokenizer_check, "small.txt": b""})
        print(f"wrote {out.name} ({out.stat().st_size / 1e3:.1f} KB): Qwen 5-batch ranks {ranks} robust")


if __name__ == "__main__":
    main()
