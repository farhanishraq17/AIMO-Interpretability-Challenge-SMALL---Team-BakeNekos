"""Bound model processes and stop their worker group when a terminal engine error is logged."""
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

FATAL = ("EngineCore failed to start", "Engine core initialization failed",
         "WorkerProc failed to start", "ninja: build stopped: subcommand failed")


def run_guarded(command, cwd: Path, log_path: Path, timeout: float = 7200) -> dict:
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=os.name == "posix")
    messages = queue.Queue()
    def reader():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    reason = None
    with log_path.open("w", encoding="utf-8") as log:
        while True:
            if time.monotonic() - started > timeout:
                reason = "wall_time_limit"
                break
            try:
                line = messages.get(timeout=0.5)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if line is None:
                break
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            if any(marker in line for marker in FATAL):
                reason = "terminal_engine_error"
                break
        if reason:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
        code = process.wait(timeout=10)
        thread.join(timeout=2)
        while not messages.empty():
            line = messages.get_nowait()
            if line:
                log.write(line)
    return {"exit_code": code if not reason else 124, "termination_reason": reason,
            "process_seconds": time.monotonic() - started}
