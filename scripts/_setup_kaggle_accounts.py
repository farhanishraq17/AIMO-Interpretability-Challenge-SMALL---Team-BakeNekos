"""Configure local Kaggle CLI 2.x access tokens from Secrets.env without printing tokens."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ROOT / "Secrets.env"
KAGGLE_DIR = Path.home() / ".kaggle"
ACCOUNTS_DIR = KAGGLE_DIR / "accounts"


def parse_tokens(text: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, value = line.split("==", 1)
        elif "=" in line:
            name, value = line.split("=", 1)
        else:
            continue
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if not value:
            continue
        if name.startswith("KAGGLE_API_TOKEN_"):
            tokens[name.removeprefix("KAGGLE_API_TOKEN_")] = value
        elif name.startswith("KAGGLE_KEY_"):
            tokens[name.removeprefix("KAGGLE_KEY_")] = value
        elif re.fullmatch(r"[A-Z][A-Z0-9_]*", name) and value.startswith("KGAT_"):
            tokens[name] = value
    return tokens


def parse_usernames(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name.startswith("KAGGLE_USERNAME_"):
            found[name.removeprefix("KAGGLE_USERNAME_")] = value.strip().strip('"').strip("'")
    return found


def introspect_token(token: str) -> str | None:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    result = subprocess.run(
        ["kaggle", "config", "view"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    match = re.search(r"username:\s+(\S+)", combined)
    if result.returncode != 0 or not match:
        first_error = next(
            (
                line.strip()
                for line in combined.splitlines()
                if line.strip() and "username:" not in line.lower()
            ),
            f"exit {result.returncode}",
        )
        print(f"  CLI rejected token: {first_error}")
        return None
    return match.group(1)


def write_secrets(resolved: dict[str, tuple[str, str]]) -> None:
    lines = [
        "# Local Kaggle CLI credentials. Do not commit this file.",
        "# These are Kaggle API access tokens (KGAT_...) for CLI 2.x.",
        "# Activate an account:",
        "#   python scripts/use_kaggle_account.py ISHRAQ",
        "# Verify the active account:",
        "#   kaggle config view",
        "",
    ]
    for alias, (username, token) in resolved.items():
        lines.append(f"KAGGLE_USERNAME_{alias}={username}")
        lines.append(f"KAGGLE_API_TOKEN_{alias}={token}")
        lines.append("")
    SECRETS.write_text("\n".join(lines), encoding="utf-8")


def write_account_files(resolved: dict[str, tuple[str, str]]) -> None:
    ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    for alias, (_username, token) in resolved.items():
        path = ACCOUNTS_DIR / f"{alias.lower()}.access_token"
        path.write_text(token + "\n", encoding="utf-8")
        os.chmod(path, 0o600)


def activate_account(alias: str, username: str, token: str) -> None:
    access_token_path = KAGGLE_DIR / "access_token"
    if access_token_path.exists():
        backup = KAGGLE_DIR / "access_token.pre-aimo-interp.bak"
        if not backup.exists():
            backup.write_bytes(access_token_path.read_bytes())
    access_token_path.write_text(token + "\n", encoding="utf-8")
    os.chmod(access_token_path, 0o600)
    (KAGGLE_DIR / "active_account").write_text(alias + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("KAGGLE_API_TOKEN", None)
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    result = subprocess.run(
        ["kaggle", "config", "view"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    print(result.stdout.strip() or result.stderr.strip())
    if username not in (result.stdout + result.stderr):
        print(f"Warning: expected active username {username}", file=sys.stderr)
    list_result = subprocess.run(
        ["kaggle", "competitions", "list", "-s", "aimo", "-p", "1"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    preview = (list_result.stdout or list_result.stderr).strip().splitlines()[:10]
    print("CLI competition search:")
    print("\n".join(preview) if preview else f"exit {list_result.returncode}")


def main() -> int:
    if not SECRETS.exists():
        print("Secrets.env not found", file=sys.stderr)
        return 1
    text = SECRETS.read_text(encoding="utf-8")
    tokens = parse_tokens(text)
    if not tokens:
        print("No Kaggle tokens found in Secrets.env", file=sys.stderr)
        return 1

    resolved: dict[str, tuple[str, str]] = {}
    failed: list[str] = []
    print("Verifying Kaggle API tokens from Secrets.env via `kaggle config view`...")
    for alias, token in tokens.items():
        print(f"\n[{alias}]")
        username = introspect_token(token)
        if username is None:
            failed.append(alias)
            continue
        resolved[alias] = (username, token)
        print(f"  resolved {alias} -> {username}")

    if not resolved:
        print("No accounts could be verified.", file=sys.stderr)
        return 1

    write_secrets(resolved)
    write_account_files(resolved)
    default_alias = "ISHRAQ" if "ISHRAQ" in resolved else next(iter(resolved))
    username, token = resolved[default_alias]
    print(f"\nActivating default CLI account: {default_alias} ({username})")
    activate_account(default_alias, username, token)

    print("\nVerified accounts:")
    for alias, (username, _) in resolved.items():
        print(f"  {alias}: {username}")
    if failed:
        print("Unverified aliases:")
        for alias in failed:
            print(f"  {alias}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
