"""Build constant-prediction ZIPs that measure per-model base rates on the live leaderboard.

E11 v1 scored 0.47 accuracy on Codabench while predicting False for the three small-track
models we have no labels for. That is only possible if the hidden validation set is not
mostly spurious, unlike our training data. The leaderboard reports only overall accuracy,
but constant submissions separate the models:

    probe-all-false            accuracy = 1 - (overall robust rate)
    probe-true-<model>         True for that model only; compared with all-false, the score
                               rises if that model's cases are mostly robust and falls if not

Each probe is a few hundred bytes and spends one of the 100 submissions.

Run:  python submission/make_probes.py      # -> dist/probes/solution-probe-*-small.zip
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "getting-started" / "scripts"))

from build import archive  # noqa: E402

OUT = ROOT / "dist" / "probes"
SMALL_TRACK_MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "qwen35-4b": "Qwen/Qwen3.5-4B",
    "skywork-7b": "Skywork/Skywork-OR1-Math-7B",
    "olmo3-think": "allenai/Olmo-3-7B-Think",
}

TEMPLATE = '''"""Leaderboard probe: {description}."""

ROBUST_MODELS = {robust!r}


def are_robust(model_id, problems, reasoning_effort="default"):
    return [model_id in ROBUST_MODELS] * len(problems)
'''


def write_probe(name: str, robust: set[str], description: str) -> Path:
    source = TEMPLATE.format(description=description, robust=sorted(robust))
    return archive(OUT / f"solution-probe-{name}-small.zip", {"solution.py": source.encode(), "small.txt": b""})


def main() -> None:
    paths = [write_probe("all-false", set(), "every case predicted False")]
    for short, model_id in SMALL_TRACK_MODELS.items():
        paths.append(write_probe(f"true-{short}", {model_id}, f"True only for {model_id}"))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
