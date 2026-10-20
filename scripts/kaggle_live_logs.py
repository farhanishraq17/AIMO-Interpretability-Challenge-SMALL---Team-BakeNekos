"""Read a bounded snapshot of a running Kaggle session's logs without waiting for output finalization.

Run: python scripts/kaggle_live_logs.py e37-official-olmo --account MUBAMAN
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from kaggle_followup import ROOT, load_accounts


def main():
    from kagglesdk.kaggle_http_client import KaggleHttpClient
    from kagglesdk.common.types.file_download import FileDownload
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("--account", required=True, choices=sorted(load_accounts()))
    args = parser.parse_args()
    username, token = load_accounts()[args.account]
    request = ApiGetKernelSessionLogsStreamRequest()
    request.user_name = username
    request.kernel_slug = "aimo-small-" + args.job
    request.wait_for_logs_url_seconds = 1
    collected, state = [], "snapshot"
    try:
        with KaggleHttpClient(api_token=token) as client:
            original_send = client._session.send
            client._session.send = lambda req, **kwargs: original_send(req, timeout=(10, 12), **kwargs)
            response = client.call("kernels.KernelsApiService", "GetKernelSessionLogsStream", request, FileDownload)
            started = time.monotonic()
            try:
                for line in response.iter_lines(chunk_size=1024, decode_unicode=True):
                    if line:
                        collected.append(line if isinstance(line, str) else line.decode(errors="replace"))
                    if time.monotonic() - started > 15 or len(collected) > 10000:
                        break
            finally:
                response.close()
    except Exception as exc:
        state = type(exc).__name__ + ": " + str(exc)
    text = "\n".join(collected)
    for _, secret in load_accounts().values():
        text = text.replace(secret, "[REDACTED]")
        state = state.replace(secret, "[REDACTED]")
    text = re.sub(r"KGAT_[A-Za-z0-9]+", "[REDACTED]", text)
    dest = ROOT / "runs" / f"{args.job}-{args.account}"
    dest.mkdir(parents=True, exist_ok=True)
    if text:
        (dest / "live-log.txt").write_text(text, encoding="utf-8")
    print(json.dumps({"account": args.account, "job": args.job, "state": state,
                      "lines": len(collected), "tail": text[-6500:]}))


if __name__ == "__main__":
    main()
