"""Reject staged credentials without displaying secret values or matching source lines."""
import re
import subprocess
from pathlib import Path

SMALL = Path(__file__).resolve().parents[1]
REPOSITORY = SMALL.parent
GIT = ["git", "-c", "safe.directory=" + REPOSITORY.as_posix(), "-C", str(REPOSITORY)]


def main():
    values = []
    for line in (SMALL / "Secrets.env").read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            value = line.split("=", 1)[1].strip().strip("\"'")
            if len(value) >= 24:
                values.append(value.encode())
    paths = subprocess.check_output(GIT + ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]).decode().split("\0")
    paths = [path for path in paths if path]
    pattern = re.compile(rb"KGAT_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    for path in paths:
        if not path.startswith(SMALL.name + "/"):
            raise SystemExit("Staged file outside the authorized Small Models scope.")
        if path.lower().endswith(".env"):
            raise SystemExit("An environment file is staged; refusing to proceed.")
        blob = subprocess.check_output(GIT + ["show", ":" + path])
        if pattern.search(blob) or any(value in blob for value in values):
            raise SystemExit("A staged file contains a possible credential; refusing to proceed.")
    print(f"Credential and scope check passed for {len(paths)} staged files. No secret values displayed.")


if __name__ == "__main__":
    main()
