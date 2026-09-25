"""L0: what hardware does this account actually get for `NvidiaH100`, and what is mounted?"""
import json
import os
import subprocess
import urllib.request


def sh(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd}\n{result.stdout[-4000:]}{result.stderr[-1000:]}", flush=True)
    return result.stdout


report = {"job": JOB}
report["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
sh("nvidia-smi")
sh("df -h /kaggle/working /tmp / 2>/dev/null; free -g; nproc")
report["inputs"] = sh("find /kaggle/input -maxdepth 4 | head -60")
report["pip"] = sh("pip list 2>/dev/null | grep -i -E '^(vllm|torch|transformers|triton|kernels|flash|openai-harmony) '")
try:
    urllib.request.urlopen("https://huggingface.co/api/models/openai/gpt-oss-120b", timeout=20)
    report["internet"] = True
except Exception as exc:  # noqa: BLE001 - we only want to know whether it works
    report["internet"] = repr(exc)
os.makedirs("/kaggle/working", exist_ok=True)
json.dump(report, open("/kaggle/working/probe.json", "w"), indent=1)
print(json.dumps(report, indent=1))
