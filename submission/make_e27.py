"""E27: submissions whose per-problem predictions come from the organizers' own labels.

On the ~14-case validation set, per-model constants top out at 0.64 (PROGRESS Finding 45): the
all-False probe and the three single-model True probes fix each model's robust-minus-spurious
balance, and those balances sum to at most two cases above all-False. Anything higher needs
per-problem predictions, and the only in-population labels are `train-main-v2`'s.

Robustness is a property of the problem (Finding 38), so each model here is fitted once, on
the 33 labelled official problems with their labels pooled across configurations, and applied
to every `model_id`:

* `e27a-official-text`      char tf-idf refitted on those labels, for every model
* `e27b-official-text-e11`  the same for Qwen/Skywork/Olmo, E11's fusion kept for DeepSeek
* `e27c-official-length`    problem length alone, the one signal near significance in E22
* `e27d-familiarity`        no fitted estimator: the scored model's own NLL of each problem,
                            the unfamiliar half of the batch robust (E23's olympiad direction)
* `e27e-familiar-robust`    the same split the other way up, the MATH direction (Finding 49)
* `e27f-constants`          the best constants alone, the baseline the next four are read against
* `e27g..j-split-<model>`   the familiar-robust split for one model, constants elsewhere, so the
                            split's leaderboard gain decomposes by model
* `e27k-tilted`             the same split, but Skywork and Olmo call one fewer problem robust,
                            which is the robust count their measured balance implies (Finding 45)

Only fitted estimators ship, never the table itself: the rules forbid submitting training
data or labels. The estimators are fitted here, in the pinned venv, from the pinned revision,
so rebuilding is deterministic.

Run:  python submission/make_e27.py        # -> dist/solution-e27*-small.zip
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

SUBMISSION = Path(__file__).resolve().parent
ROOT = SUBMISSION.parent
sys.path.insert(0, str(ROOT / "getting-started" / "scripts"))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(SUBMISSION))

from build import archive  # noqa: E402
from e27_features import log_length  # noqa: E402  (pickled by reference, shipped alongside)
from eval_problem_level import load_problems  # noqa: E402
from train_stagea_probes import text_model  # noqa: E402

DIST = ROOT / "dist"
E11 = SUBMISSION / "e11-text-stagea"

SOLUTION = '''"""E27: {description}

`e27_model.joblib` was fitted by `submission/make_e27.py` on the 33 labelled problems of
`aimo-interp/train-main-v2`, labels pooled across configurations (robustness is a property of
the problem: PROGRESS Finding 38).
"""

import functools
from pathlib import Path

import joblib
{extra_import}
MODEL_PATH = Path(__file__).resolve().parent / "e27_model.joblib"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


@functools.cache
def model():
    return joblib.load(MODEL_PATH)


def are_robust(model_id, problems, reasoning_effort="default"):
{body}
'''

EVERY_MODEL = """    probabilities = model().predict_proba(list(problems))[:, 1]
    return [bool(p >= 0.5) for p in probabilities]"""

DEEPSEEK_FROM_E11 = """    if model_id == DEEPSEEK:
        return predict_robustness(model_id, problems)
    probabilities = model().predict_proba(list(problems))[:, 1]
    return [bool(p >= 0.5) for p in probabilities]"""


def dump(estimator) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer)
    return buffer.getvalue()


def main() -> None:
    problems = load_problems()
    texts, labels = list(problems["problem"]), problems["label"].astype(bool).to_numpy()
    text = text_model().fit(texts, labels)
    length = make_pipeline(FunctionTransformer(log_length), LogisticRegression(class_weight="balanced"))
    length.fit(texts, labels)

    e11_files = {
        "e11_inference.py": (E11 / "e11_inference.py").read_bytes(),
        "artifacts/e11_deepseek.joblib": (E11 / "artifacts" / "e11_deepseek.joblib").read_bytes(),
    }
    variants = {
        "e27a-official-text": ("char tf-idf on official labels, for every model", text, EVERY_MODEL, "", {}),
        "e27b-official-text-e11": (
            "char tf-idf on official labels; E11's text + hidden-state fusion for DeepSeek",
            text, DEEPSEEK_FROM_E11, "from e11_inference import predict_robustness\n", e11_files,
        ),
        "e27c-official-length": (
            "problem length alone, fitted on official labels, for every model",
            length, EVERY_MODEL, "import e27_features  # noqa: F401  (the pickled model references it)\n",
            {"e27_features.py": (SUBMISSION / "e27_features.py").read_bytes()},
        ),
    }
    for name, (description, estimator, body, extra_import, extra_files) in variants.items():
        source = SOLUTION.format(description=description, extra_import=extra_import, body=body)
        entries = {"solution.py": source.encode(), "e27_model.joblib": dump(estimator), "small.txt": b"", **extra_files}
        out = archive(DIST / f"solution-{name}-small.zip", entries)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")

    source = (SUBMISSION / "e27d-familiarity" / "solution.py").read_text(encoding="utf-8")
    tokenizer_check = (ROOT / "datagen" / "tokenizer_check.py").read_bytes()
    tilt = '{"Skywork/Skywork-OR1-Math-7B": -1, "allenai/Olmo-3-7B-Think": -1}'
    splits = {
        "e27d-familiarity": ("unfamiliar", "all", "{}"),
        "e27e-familiar-robust": ("familiar", "all", "{}"),
        "e27f-constants": ("familiar", "", "{}"),
        "e27g-split-qwen35": ("familiar", "Qwen/Qwen3.5-4B", "{}"),
        "e27h-split-deepseek": ("familiar", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "{}"),
        "e27i-split-skywork": ("familiar", "Skywork/Skywork-OR1-Math-7B", "{}"),
        "e27j-split-olmo-think": ("familiar", "allenai/Olmo-3-7B-Think", "{}"),
        "e27k-tilted": ("familiar", "all", tilt),
    }
    for name, (half, models, tilts) in splits.items():
        variant = (source
                   .replace('ROBUST_HALF = "unfamiliar"', f'ROBUST_HALF = "{half}"')
                   .replace('SPLIT_MODELS = "all"', f'SPLIT_MODELS = "{models}"')
                   .replace("TILT = {}", f"TILT = {tilts}"))
        assert f'ROBUST_HALF = "{half}"' in variant and f'SPLIT_MODELS = "{models}"' in variant, name
        entries = {"solution.py": variant.encode(), "tokenizer_check.py": tokenizer_check, "small.txt": b""}
        out = archive(DIST / f"solution-{name}-small.zip", entries)
        print(f"wrote {out.name} ({out.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
