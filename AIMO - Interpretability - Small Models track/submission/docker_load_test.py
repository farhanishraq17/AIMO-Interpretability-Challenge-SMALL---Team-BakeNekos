"""E42: does the shipped ZIP load and run inside `Dockerfile.competition`?

`COMPETITION.md` section 14 requires every upload to succeed in the competition image and not only
in the local venv, and no submission in this project has ever been checked that way. The failure
it guards against is silent and total: a ZIP that imports cleanly under our pins but raises on the
worker scores 0.00, and participant tracebacks are hidden on Codabench.

This runs *inside* the container, with networking off, and answers four questions:

1. Do the pinned runtime versions match what `COMPETITION.md` says they are?
2. Does the ZIP's own `solution.py` import under them - every import resolved, no stray dependency
   that only exists in our venv?
3. Does `are_robust` return exactly one **native** `bool` per problem, for every evaluated
   `model_id` and for an unknown one?
4. Does it stay valid when the weights are absent, which is what the fallback path is for? The
   container has no model cache, so the split cannot run and the constants must answer instead.

What it cannot check is the forward-pass path itself, which needs the worker's cached weights and
a GPU. E39 covers that on Kaggle.

Run (inside the image):  python /work/docker_load_test.py /work/solution.zip
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

EXPECTED = {  # COMPETITION.md section 11
    "torch": "2.12.1",
    "transformers": "5.13.0",
    "sklearn": "1.8.0",
    "pandas": "3.0.3",
    "joblib": "1.5.3",
    "huggingface_hub": "1.22.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "accelerate": "1.13.0",
}
MODELS = [
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]
PROBLEMS = [
    "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$",
    "An isosceles trapezoid has an inscribed circle tangent to each of its four sides.",
    "Triangle $ABC$ has side length $AB = 20$ and circumradius $R = 26$.",
]


def versions() -> dict[str, str]:
    import importlib

    found = {}
    for name in EXPECTED:
        try:
            found[name] = importlib.import_module(name).__version__
        except Exception as exc:  # a missing pin is itself the finding
            found[name] = f"MISSING ({type(exc).__name__})"
    return found


def main() -> None:
    archive = Path(sys.argv[1])
    report: dict[str, object] = {"zip": archive.name}

    names = sorted(zipfile.ZipFile(archive).namelist())
    report["contents"] = names
    report["solution_at_root"] = "solution.py" in names
    report["has_small_txt"] = "small.txt" in [n.lower() for n in names]

    found = versions()
    report["versions"] = found
    report["versions_match"] = {k: (found[k] == v) for k, v in EXPECTED.items()}

    unpacked = Path("/tmp/zip")
    zipfile.ZipFile(archive).extractall(unpacked)
    sys.path.insert(0, str(unpacked))
    import solution  # noqa: E402  (the packaged file, which is the point)

    report["imported"] = True
    results = {}
    for model_id in [*MODELS, "some/model-we-never-saw"]:
        predictions = solution.are_robust(model_id, PROBLEMS, "default")
        results[model_id] = {
            "predictions": predictions,
            "length_ok": len(predictions) == len(PROBLEMS),
            "native_bools": all(type(p) is bool for p in predictions),
        }
    report["are_robust"] = results
    # The two-argument form is the legacy caller and must keep working (COMPETITION.md section 4).
    legacy = solution.are_robust(MODELS[0], PROBLEMS)
    report["legacy_two_arg_call"] = {"predictions": legacy, "native_bools": all(type(p) is bool for p in legacy)}
    report["empty_batch"] = solution.are_robust(MODELS[0], [])

    ok = (
        report["solution_at_root"]
        and all(report["versions_match"].values())
        and all(r["length_ok"] and r["native_bools"] for r in results.values())
        and report["legacy_two_arg_call"]["native_bools"]
        and report["empty_batch"] == []
    )
    report["PASS"] = ok
    print(json.dumps(report, indent=2))
    print("\nDOCKER LOAD TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
