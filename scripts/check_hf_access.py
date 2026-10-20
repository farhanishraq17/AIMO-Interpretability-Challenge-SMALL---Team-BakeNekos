"""Report which aimo-interp datasets are reachable with the current HF credentials."""

from __future__ import annotations

from huggingface_hub import HfApi

REPOS = [
    "aimo-interp/val-sample",
    "aimo-interp/problems-public",
    "aimo-interp/augmented-sample-math-agg",
]


def main() -> None:
    api = HfApi()
    try:
        who = api.whoami()
        print(f"authenticated as: {who.get('name')}")
    except Exception as exc:  # noqa: BLE001 - anonymous access is a valid state
        print(f"not authenticated ({type(exc).__name__})")

    print()
    for repo in REPOS:
        try:
            info = api.dataset_info(repo)
            files = [s.rfilename for s in (info.siblings or [])]
            print(f"OK    {repo}")
            print(f"        gated={info.gated} private={info.private} files={len(files)}")
            for name in files[:10]:
                print(f"        - {name}")
        except Exception as exc:  # noqa: BLE001 - we are probing availability
            first = str(exc).splitlines()[0][:100]
            print(f"FAIL  {repo}: {type(exc).__name__}: {first}")
        print()

    print("datasets visible for author 'aimo-interp':")
    try:
        for dataset in api.list_datasets(author="aimo-interp"):
            print(f"  {dataset.id}")
    except Exception as exc:  # noqa: BLE001
        print(f"  listing failed: {type(exc).__name__}")


if __name__ == "__main__":
    main()
