"""Expose the installed NVIDIA driver to link-time CUDA compilation, then smoke-test it.

Kaggle provides libcuda.so.1 without the libcuda.so linker name on some images.
Only create a local symlink; never replace or install a driver.
Run inside the GPU kernel: python scripts/kaggle_cuda_setup.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


def prepare_cuda_linker(directory: Path = Path("/tmp/aimo-cuda-link")) -> dict:
    listing = subprocess.run(["/sbin/ldconfig", "-p"], capture_output=True, text=True, check=True)
    candidates = re.findall(r"libcuda\.so\.1\s+.*?=>\s+(\S+)", listing.stdout)
    candidates += ["/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64-nvidia/libcuda.so.1",
                   "/usr/local/nvidia/lib64/libcuda.so.1"]
    driver = next((Path(p).resolve() for p in candidates if Path(p).is_file() and "/stubs/" not in p), None)
    if driver is None:
        raise RuntimeError("No installed libcuda.so.1 found; refusing a model run")
    directory.mkdir(parents=True, exist_ok=True)
    linker_name = directory / "libcuda.so"
    if not linker_name.exists():
        linker_name.symlink_to(driver)
    if linker_name.resolve() != driver:
        raise RuntimeError("Existing linker symlink points to a different driver")
    for name in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        os.environ[name] = str(directory) + ":" + str(driver.parent) + ":" + os.environ.get(name, "")
    source = directory / "cuda_link_probe.c"
    binary = directory / "cuda_link_probe"
    source.write_text("extern int cuInit(unsigned int);\nint main(void) { return cuInit(0); }\n")
    subprocess.run(["cc", str(source), "-o", str(binary), "-lcuda"], check=True, timeout=30)
    subprocess.run([str(binary)], check=True, timeout=30)
    return {"driver": str(driver), "linker_name": str(linker_name), "compile_and_cuInit_passed": True}


if __name__ == "__main__":
    result = prepare_cuda_linker()
    Path("/kaggle/working/cuda-link-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
