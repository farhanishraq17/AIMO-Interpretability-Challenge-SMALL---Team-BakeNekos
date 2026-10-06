"""The read-only account rule: RSD may be read from, never pushed to.

A rule nobody can run is a note. This asserts the guard actually refuses, that it refuses the
`--account` override too (the way a push would most plausibly reach RSD by accident), and that
reading is left alone.

Run:  python datagen/test_read_only_accounts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_launch as k


def push(name: str, account: str | None = None) -> str | None:
    """Run the push path far enough to hit the guard; returns the refusal, or None if it pushed."""
    argv = ["kaggle_launch.py", "push", name] + (["--account", account] if account else [])
    saved = sys.argv
    sys.argv = argv
    try:
        k.main()
    except SystemExit as exc:
        return str(exc)
    finally:
        sys.argv = saved
    return None


def main() -> None:
    assert "RSD" in k.READ_ONLY_ACCOUNTS, "the rule this file tests is gone"

    # An experiment still registered to RSD must be refused, and so must an explicit override.
    on_rsd = [n for n, e in k.EXPERIMENTS.items() if e.account == "RSD"]
    assert on_rsd, "no registry entry names RSD; this test would pass vacuously"
    refusal = push(on_rsd[0])
    assert refusal and "read-only" in refusal, f"RSD push was not refused: {refusal!r}"

    any_name = next(iter(k.EXPERIMENTS))
    refusal = push(any_name, "RSD")
    assert refusal and "read-only" in refusal, f"--account RSD was not refused: {refusal!r}"

    # Reading must still work: the guard belongs to `push` only, so the token is still loaded.
    assert "RSD" in k.load_accounts(), "RSD's token was removed; reads would break"
    assert k.kernel_ref(any_name, "RSD").startswith("rsdaiyan/"), "RSD refs no longer resolve"

    # `scripts/kaggle_followup.py` is a second push path (E37, E49) and must refuse RSD too. Its
    # Kaggle call is replaced first, so a broken guard fails this test instead of pushing.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import kaggle_followup as f

    def no_push(*args):
        raise AssertionError(f"kaggle_followup reached the Kaggle CLI for RSD: {args}")

    f.call, saved = no_push, sys.argv
    sys.argv = ["kaggle_followup.py", "push", "e49-qwen", "--account", "RSD"]
    try:
        f.main()
        refusal = None
    except SystemExit as exc:
        refusal = str(exc)
    finally:
        sys.argv = saved
    assert refusal and "read-only" in refusal, f"kaggle_followup did not refuse RSD: {refusal!r}"

    print(f"read-only accounts: {sorted(k.READ_ONLY_ACCOUNTS)} - push refused on both launchers, read allowed. ok")


if __name__ == "__main__":
    main()
