import json, subprocess, urllib.request
from pathlib import Path
r = subprocess.run(["sh", "-c", "command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"], capture_output=True, text=True)
result = {"gpu_available": r.returncode == 0, "gpus": r.stdout.strip().splitlines()}
try:
    # A model the kernels download. The dataset probed before 21 Oct went private and answers 401 to everyone,
    # which read as "no internet" on healthy accounts.
    with urllib.request.urlopen("https://huggingface.co/api/models/Qwen/Qwen3.5-4B", timeout=30) as response:
        result["internet_available"] = response.status == 200
except Exception as exc:
    result["internet_available"] = False
    result["network_error"] = type(exc).__name__
Path("/kaggle/working/capability.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
