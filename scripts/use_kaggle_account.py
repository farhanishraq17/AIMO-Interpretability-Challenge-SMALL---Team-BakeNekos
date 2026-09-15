"""Switch the active local Kaggle CLI account from Secrets.env."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ROOT / "Secrets.env"
KAGGLE_DIR = Path.home() / ".kaggle"
ACCOUNTS_DIR = KAGGLE_DIR / "accounts"


def load_accounts() -> dict[str, tuple[str, str]]:
    usernames: dict[str, str] = {}
    tokens: dict[str, str] = {}
    for raw in SECRETS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name.startswith("KAGGLE_USERNAME_"):
            usernames[name.removeprefix("KAGGLE_USERNAME_")] = value
        elif name.startswith("KAGGLE_API_TOKEN_"):
            tokens[name.removeprefix("KAGGLE_API_TOKEN_")] = value
    return {alias: (usernames.get(alias, ""), token) for alias, token in tokens.items()}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/use_kaggle_account.py <ALIAS>")
        print("Example: python scripts/use_kaggle_account.py ISHRAQ")
        return 1
    alias = sys.argv[1].strip().upper()
    accounts = load_accounts()
    if alias not in accounts:
        print(f"Unknown alias {alias}. Available: {', '.join(sorted(accounts))}")
        return 1
    username, token = accounts[alias]
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
    token_copy = ACCOUNTS_DIR / f"{alias.lower()}.access_token"
    token_copy.write_text(token + "\n", encoding="utf-8")
    os.chmod(token_copy, 0o600)
    access_token_path = KAGGLE_DIR / "access_token"
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
    print(f"Active account: {alias}" + (f" ({username})" if username else ""))
    print(result.stdout.strip() or result.stderr.strip())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
