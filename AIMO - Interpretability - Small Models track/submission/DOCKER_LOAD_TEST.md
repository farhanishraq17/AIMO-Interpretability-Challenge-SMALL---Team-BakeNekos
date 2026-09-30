# Docker load test — instructions

**What this is:** a check that our competition submission loads and runs inside the *exact* image
Codabench uses, rather than only in our own Python environment.

**How urgent is it? Not very.** The ZIP being tested already scored 0.86 on the live leaderboard,
which means it has already run inside the real container. This is a reproducibility check for the
November technical report, and a template for checking any future rebuild *before* it is uploaded.
**Nothing about the competition deadline depends on it.** If it is inconvenient, say so and skip it.

**Time:** about 20 minutes, most of it downloading the base image (~4 GB).
**You need:** Docker running, ~10 GB free disk, an internet connection for the build step only.

---

## 1. Make a folder with three files

Put these in one empty folder — call it anything, say `aimo-docker-test`:

| File | Where it comes from |
| --- | --- |
| `solution-e27k-tilted-small.zip` | sent with this file |
| `docker_load_test.py` | sent with this file |
| `Dockerfile.competition` | copy the block in §2 below into a file with exactly this name |

Nothing else is needed. You do **not** need to clone our repository.

## 2. `Dockerfile.competition`

This is the competition image, unchanged from the organizers' `aimo-interp/getting-started`
repository. Copy it verbatim — the pinned digest matters, because it is what makes this image the
same one Codabench runs.

```dockerfile
# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime@sha256:f48553a21cd5944e8093a74801e403283311b460a5b602ec02d15fe98f04ff5f

RUN python -m pip install --no-cache-dir --break-system-packages \
        "accelerate==1.13.0" \
        "huggingface-hub==1.22.0" \
        "joblib==1.5.3" \
        "pandas==3.0.3" \
        "safetensors==0.8.0" \
        "scikit-learn==1.8.0" \
        "tokenizers==0.22.2" \
        "transformers==5.13.0" \
    && python -c "import sklearn, torch, transformers; from transformers import Qwen3ForCausalLM; print(torch.__version__, transformers.__version__, sklearn.__version__, Qwen3ForCausalLM.__name__)"

ENV HF_HUB_OFFLINE=1 \
    HF_HOME=/app/data/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false
```

If you would rather not copy-paste it, the original is at
<https://github.com/aimo-interp/getting-started/blob/main/Dockerfile.competition>.

## 3. Build the image

From inside that folder:

```bash
docker build -f Dockerfile.competition -t aimo-competition:local .
```

This downloads the PyTorch base image and installs the pinned libraries. It is the slow step.
It should end with a line like `2.12.1+cu132 5.13.0 1.8.0 Qwen3ForCausalLM`. The `+cu132` is PyTorch's CUDA local-version suffix and is expected — the test compares the release portion.

## 4. Run the test

**Linux / macOS:**

```bash
docker run --rm --network none -v "$PWD":/work -w /work aimo-competition:local python docker_load_test.py solution-e27k-tilted-small.zip
```

**Windows PowerShell:**

```powershell
docker run --rm --network none -v "${PWD}:/work" -w /work aimo-competition:local python docker_load_test.py solution-e27k-tilted-small.zip
```

`--network none` is deliberate: the real evaluation container has no internet, and we want any
code that secretly needs the network to fail here rather than on the leaderboard.

## 5. What you should see

A JSON report, then a final line. The **last line is the whole result**:

```
DOCKER LOAD TEST: PASS
```

The exit code is 0 on pass and 1 on fail, so `echo $?` (or `$LASTEXITCODE` in PowerShell) works too.

Inside the JSON, the things that matter:

| Field | Should be |
| --- | --- |
| `versions_match` | every entry `true` — the release portion of each version matches the pin, ignoring any `+cu132`/`+cpu` local suffix |
| `imported` | `true` — our `solution.py` imports under those versions |
| `are_robust` → each model → `native_bools` | `true` — real Python `bool`, not NumPy bool or int |
| `are_robust` → each model → `length_ok` | `true` — one prediction per problem |
| `legacy_two_arg_call` | present and `native_bools: true` |
| `empty_batch` | `[]` |
| `marker_correct` | `true` — a `-small` ZIP carries `small.txt` and a `-main` ZIP does not. Either mistake scores 0.00 |

**Expect to see warnings on stderr about models falling back.** That is correct and is part of the
test. The container has no model weights cached, so our solution cannot run its normal path and
must fall back to constants — proving the fallback works is one of the four things being checked.
Lines like `e27d: OSError: ... does not appear to have a file named ...; falling back to constants`
are the expected output, not an error.

## 6. What to send back

Whatever the result:

- the **last line** (`PASS` or `FAIL`)
- the full JSON report, or just the `versions` and `versions_match` blocks if it is long
- if it failed, the traceback

If it says `FAIL`, that is a genuine and useful finding — it means our submission behaves
differently in the competition image than in our environment, and we would want to know before
the 1 November deadline.

## 7. Cleaning up

```bash
docker rmi aimo-competition:local
docker image prune
```

The base image is a few GB, so this is worth doing if you are short on disk.

---

### For context: what is actually being tested

Our submission implements `are_robust(model_id, problems, reasoning_effort)`, which predicts for
each (model, problem) pair whether that model's answer survives answer-preserving perturbations.
The real run loads the model being scored and takes one forward pass per problem. None of that can
happen here — no GPU, no cached weights — and it is already checked separately on GPU hardware.

What this test covers is the part that fails *silently* on the leaderboard: a submission that
imports cleanly under our library versions but raises under the competition's, or returns a NumPy
bool instead of a Python one, scores **0.00** with no visible traceback. Four questions, then:

1. Does the image really have the pinned versions?
2. Does our `solution.py` import under them?
3. Does it return exactly one native `bool` per problem, for every model and for an unknown one?
4. When the weights are missing, does the fallback still return something valid?
