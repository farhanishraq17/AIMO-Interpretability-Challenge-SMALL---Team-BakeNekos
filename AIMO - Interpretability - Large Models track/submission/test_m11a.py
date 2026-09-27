"""What the built M11a ZIP does, without a GPU.

Only the split path needs a model; every branch below is reachable on CPU. Run after
`make_m11a.py`, which writes the ZIP this reads.

Run:  python submission/test_m11a.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "dist" / "solution-m11a-constant-gptoss-main.zip"
GPT_OSS = "openai/gpt-oss-120b"
SMALL = ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")


def load(extract_to: Path):
    """Import the ZIP's own solution.py, so the test reads what would be uploaded."""
    with zipfile.ZipFile(ZIP) as bundle:
        assert "small.txt" not in [n.lower() for n in bundle.namelist()], "Main-Track ZIP marked small"
        bundle.extractall(extract_to)
    sys.path.insert(0, str(extract_to))
    import solution

    return solution


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        s = load(Path(tmp))

        # gpt-oss-120b: always robust, and never loaded - the constant branch is taken first.
        problems = [f"problem {i}" for i in range(9)]
        for effort in ("low", "medium", "high", "default"):
            out = s.are_robust(GPT_OSS, problems, effort)
            assert out == [True] * 9, (effort, out)
            assert all(isinstance(v, bool) for v in out), "native bools only"
        assert not s.uses_split(GPT_OSS) and s.uses_split(SMALL[0])

        # The split: the familiar (low-NLL) end is robust, ceil(n/2) of the batch plus the tilt.
        nll = [5.0, 1.0, 4.0, 2.0, 3.0]  # familiar order: 1, 3, 4, 2, 0
        assert s.split_half(nll, SMALL[0]) == [False, True, False, True, True]  # 3 robust
        assert s.split_half(nll, "Skywork/Skywork-OR1-Math-7B") == [False, True, False, True, False]  # 2
        assert s.split_half(nll, "allenai/Olmo-3-7B-Think") == [False, True, False, True, False]
        assert sum(s.split_half([0.0] * 8, SMALL[0])) == 4, "even batch: half robust"
        assert s.split_half([], SMALL[0]) == [], "empty batch stays valid"

        # Every unsplit or failing batch still returns one native bool per problem.
        for model, expected in [(SMALL[0], True), ("Skywork/Skywork-OR1-Math-7B", False),
                                ("allenai/Olmo-3-7B-Think", False), (SMALL[3], True), (GPT_OSS, True),
                                ("some/unlisted-model", False)]:
            out = s.are_robust(model, problems)  # no weights cached locally, so the split falls back
            assert out == [expected] * 9, (model, out[:3])
            assert len(out) == len(problems)

    print("m11a: ok")


if __name__ == "__main__":
    main()
