"""Mirror every public dataset of the `aimo-interp` HF organisation, plus the website, into data/.

The competition publishes its data on the Hugging Face Hub, not on Codabench, and some of it is
linked from nowhere (`train-main-v2` was found by listing the org). So enumerate the org instead of
trusting a hand-kept list, pin each snapshot's revision, and record what could not be fetched
(e.g. `problems-public`, which answers 401) in data/MANIFEST.json.

Run: getting-started/.venv/Scripts/python.exe "AIMO - Interpretability - Large Models track/scripts/download_data.py"
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ORG = "aimo-interp"
# Known from COMPETITION.md even if the org listing hides them; failures are recorded, not raised.
EXTRA = ["aimo-interp/problems-public"]
WEBSITE = "https://aimo-interp.github.io/"


def main() -> None:
    api = HfApi()
    manifest: dict = {"fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "datasets": {}}
    listed = [d.id for d in api.list_datasets(author=ORG)]
    for repo in sorted(set(listed + EXTRA)):
        entry: dict = {"listed": repo in listed}
        try:
            info = api.dataset_info(repo)
            local = snapshot_download(repo, repo_type="dataset", revision=info.sha,
                                      local_dir=DATA / "hf" / repo.split("/", 1)[1])
            files = sorted(p for p in Path(local).rglob("*") if p.is_file() and ".cache" not in p.parts)
            entry.update(revision=info.sha, last_modified=str(info.last_modified),
                         files={p.relative_to(local).as_posix(): p.stat().st_size for p in files})
        except Exception as exc:  # noqa: BLE001 - a gated/401 repo is a finding, not a crash
            entry["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        manifest["datasets"][repo] = entry
        print(repo, entry.get("revision", entry.get("error")))

    site = DATA / "website"
    site.mkdir(parents=True, exist_ok=True)
    html = urllib.request.urlopen(WEBSITE, timeout=30).read().decode("utf-8", "replace")
    (site / "index.html").write_text(html, encoding="utf-8")
    manifest["website"] = {"url": WEBSITE, "links": sorted(set(re.findall(r'href="(https?://[^"]+)"', html)))}
    (DATA / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
