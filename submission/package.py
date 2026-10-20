"""Build small-track Codabench ZIPs for our own solutions with the starter kit's archive code.

The kit's `scripts/build.py` only packages directories under `getting-started/solutions/`,
and `getting-started/` is a vendored submodule we do not edit. This reuses its `archive`
and `directory_entries` helpers, so our ZIPs get the same deterministic layout (fixed
timestamps, no `__pycache__` or dotfiles), with `small.txt` added at the root.

Run:  python submission/package.py e11-text-stagea    # -> dist/solution-e11-text-stagea-small.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUBMISSIONS = Path(__file__).resolve().parent
ROOT = SUBMISSIONS.parent
sys.path.insert(0, str(ROOT / "getting-started" / "scripts"))

from build import archive, directory_entries  # noqa: E402

DIST = ROOT / "dist"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="solution directory under submission/")
    args = parser.parse_args()

    source = SUBMISSIONS / args.name
    if not (source / "solution.py").is_file():
        raise SystemExit(f"{source} has no solution.py")
    entries = directory_entries(source)
    if any(name.casefold() == "small.txt" for name in entries):
        raise SystemExit("remove small.txt from the source directory; the packager adds it")
    entries["small.txt"] = b""

    output = archive(DIST / f"solution-{args.name}-small.zip", entries)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    for name in sorted(entries):
        print(f"  {len(entries[name]):>10,d}  {name}")


if __name__ == "__main__":
    main()
