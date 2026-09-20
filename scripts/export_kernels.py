"""Export every Kaggle kernel we have run into the repository, so runs survive the accounts.

`datagen/kaggle_launch.py` renders kernels into `kaggle_runs/`, which is git-ignored because
each rendered script carries a base64 copy of the code bundle (hundreds of KB, duplicating
files that are already tracked here). If the Kaggle accounts change or a kernel is deleted,
nothing outside this repository should be needed to rebuild and rerun the experiment.

This writes, for each registry entry:

    kernels/<name>.py      the exact kernel source, with the base64 bundle replaced by a
                           comment naming the files it contained (they are all tracked here)
    kernels/<name>.json    account, kernel slug, accelerator, pip pins, argv steps, bundle
                           file list and SHA-1, mounted datasets
    kernels/runs/<name>/   provenance of the runs already downloaded: run_manifest.json and
                           every metadata.json under runs/<name>/ (small files only)
    kernels/README.md      index: experiment, account, kernel URL, and what it produced

Everything here is generated; rerun this script after changing `EXPERIMENTS`. There are no
Jupyter notebooks in this project: every Kaggle run is a script kernel rendered from the
registry, which is what makes it reproducible.

Run:  python scripts/export_kernels.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))

import kaggle_launch as launcher  # noqa: E402

OUT = ROOT / "kernels"
BUNDLE_PLACEHOLDER = (
    '"""  # bundle stripped for the repository copy: these files are tracked here\n'
    "{files}\n"
    '"""'
)


def bundle_files(experiment: launcher.Experiment) -> tuple[list[str], str | None]:
    """The repo files the kernel embeds, and the bundle SHA-1 when they are all present."""
    names = [*launcher.BUNDLE_FILES]
    for directory in launcher.BUNDLE_DIRS:
        names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / directory).rglob("*.py"))]
    for pattern in experiment.bundle_globs:
        names += [Path(m).relative_to(ROOT).as_posix() for m in sorted(ROOT.glob(pattern))]
    names.append(launcher.SYSTEM_PROMPT_PATH)
    try:
        _, sha1 = launcher.build_bundle(experiment.bundle_globs)
    except SystemExit:  # generated inputs missing locally; regenerate with datagen/ingest_public.py
        sha1 = None
    return sorted(set(names)), sha1


def render(name: str, experiment: launcher.Experiment, files: list[str], sha1: str | None) -> str:
    payload = {
        "name": name,
        "account": experiment.account,
        "pip": launcher.PIP_PINS,
        "steps": [list(step) for step in experiment.steps],
    }
    listing = "\n".join(f"  {f}" for f in files)
    return (
        launcher.KERNEL_TEMPLATE.replace("__NAME__", name)
        .replace("__PURPOSE__", experiment.purpose)
        .replace("__EXPERIMENT_JSON__", json.dumps(json.dumps(payload)))
        .replace("__BUNDLE_SHA1__", sha1 or "unavailable-locally")
        .replace('"__BUNDLE_B64__"', BUNDLE_PLACEHOLDER.format(files=listing))
    )


def copy_provenance(name: str) -> dict:
    """Manifest and metadata of a downloaded run; records and feature dumps stay out."""
    source = launcher.RUNS_DIR / name
    target = OUT / "runs" / name
    copied: list[str] = []
    for path in sorted(source.rglob("*.json")):
        if path.name not in ("run_manifest.json", "metadata.json"):
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        copied.append(destination.relative_to(OUT).as_posix())
    if not copied:
        return {}
    manifest = target / "run_manifest.json"
    summary = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    return {
        "files": copied,
        "seconds": summary.get("seconds"),
        "returncodes": [step.get("returncode") for step in summary.get("steps", [])],
    }


def main() -> None:
    if (OUT / "runs").exists():
        shutil.rmtree(OUT / "runs")
    OUT.mkdir(exist_ok=True)

    index = ["| Experiment | Account | Kernel | Downloaded run |", "| --- | --- | --- | --- |"]
    for name, experiment in launcher.EXPERIMENTS.items():
        files, sha1 = bundle_files(experiment)
        (OUT / f"{name}.py").write_text(render(name, experiment, files, sha1), encoding="utf-8")
        # No `Secrets.env` holds every alias and no checkout holds every downloaded run, so an
        # export from one machine must never overwrite what another machine recorded. Both the
        # kernel slug and the run provenance fall back to the committed values.
        previous = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8")) \
            if (OUT / f"{name}.json").is_file() else {}
        try:
            ref = launcher.kernel_ref(name)
        except KeyError:  # this machine has no token for that alias
            ref = previous.get("kernel") or f"<{experiment.account}>/aimo-small-{name}"
        provenance = copy_provenance(name) or previous.get("downloaded_run", {})
        (OUT / f"{name}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "account": experiment.account,
                    "kernel": ref,
                    "accelerator": experiment.accelerator,
                    "purpose": experiment.purpose,
                    "pip": launcher.PIP_PINS,
                    "steps": [list(step) for step in experiment.steps],
                    "dataset_sources": list(experiment.dataset_sources),
                    "bundle_sha1": sha1,
                    "bundle_files": files,
                    "downloaded_run": provenance,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        run_note = (
            f"{provenance['seconds'] / 3600:.1f} h, exit {provenance['returncodes']}"
            if provenance.get("seconds") is not None
            else ("downloaded" if provenance else "not downloaded")
        )
        index.append(f"| `{name}` | {experiment.account} | https://www.kaggle.com/code/{ref} | {run_note} |")

    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Kaggle kernels, as run",
                "",
                "Generated by `scripts/export_kernels.py`; do not edit by hand. One `<name>.py` per",
                "experiment, exactly as pushed, except that the embedded base64 bundle is replaced by",
                "the list of repository files it held. `<name>.json` carries the account, kernel slug,",
                "runtime pins, argv steps and bundle SHA-1; `runs/<name>/` keeps the manifests and",
                "metadata of runs already downloaded.",
                "",
                "To rerun any experiment from this repository alone:",
                "",
                "```powershell",
                "$py = \"getting-started\\.venv\\Scripts\\python.exe\"",
                "& $py datagen\\ingest_public.py                       # regenerate data/ inputs",
                "& $py datagen\\kaggle_launch.py build <experiment>     # render, no network",
                "& $py datagen\\kaggle_launch.py push <experiment>      # needs that alias in Secrets.env",
                "```",
                "",
                "The account only decides where a kernel runs. To run one elsewhere, change its",
                "`account` in `datagen/kaggle_launch.py` and push again.",
                "",
                *index,
                "",
                "There are no Jupyter notebooks in this project: every Kaggle run is a script kernel",
                "rendered from the registry, so nothing exists only in a Kaggle account.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"exported {len(launcher.EXPERIMENTS)} kernels to {OUT}")


if __name__ == "__main__":
    main()
