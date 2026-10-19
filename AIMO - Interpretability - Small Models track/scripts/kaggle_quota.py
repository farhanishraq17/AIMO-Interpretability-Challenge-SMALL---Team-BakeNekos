"""Weekly GPU quota for every Kaggle alias in Secrets.env, read straight from the API.

`kaggle quota` (CLI 2.2.2) and the SDK's `get_accelerator_quota_statistics` both crash parsing the reply
("not enough values to unpack"), so this posts to the same endpoint and reads the JSON itself. Tokens are
sent only as the bearer header and never printed.

Run:  python scripts/kaggle_quota.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_followup import load_accounts  # noqa: E402

URL = "https://api.kaggle.com/v1/kernels.KernelsApiService/GetAcceleratorQuotaStatistics"


def hours(value) -> float:
    """The API reports durations as strings like "82440.5s"."""
    match = re.fullmatch(r"(-?[\d.]+)s", str(value or "0s"))
    return float(match.group(1)) / 3600 if match else 0.0


def main() -> None:
    print(f"{'alias':9s} {'user':22s} {'used':>5s} {'running':>7s} {'allowed':>7s} {'left':>5s}  resets (UTC)")
    for alias, (user, token) in load_accounts().items():
        request = urllib.request.Request(URL, data=b"{}", headers={"Authorization": f"Bearer {token}",
                                                                  "Content-Type": "application/json"})
        try:
            reply = json.load(urllib.request.urlopen(request, timeout=60))
        except urllib.error.HTTPError as exc:
            print(f"{alias:9s} {user:22s} HTTP {exc.code}")
            continue
        gpu = reply.get("gpuQuota", {})
        used, running, allowed = (hours(gpu.get(k)) for k in ("timeUsed", "timeReserved", "totalTimeAllowed"))
        print(f"{alias:9s} {user:22s} {used:5.1f} {running:7.1f} {allowed:7.1f} {allowed - used - running:5.1f}  "
              f"{reply.get('quotaRefreshTime')}")


if __name__ == "__main__":
    main()
