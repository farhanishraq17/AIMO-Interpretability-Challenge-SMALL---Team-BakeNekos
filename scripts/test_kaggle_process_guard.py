import sys
from kaggle_process_guard import run_guarded


def test_successful_process(tmp_path):
    result = run_guarded([sys.executable, "-c", "print('ready')"], tmp_path, tmp_path / "log.txt", timeout=5)
    assert result["exit_code"] == 0
    assert "ready" in (tmp_path / "log.txt").read_text()


def test_fatal_worker_error_cannot_leave_session_hanging(tmp_path):
    result = run_guarded([sys.executable, "-u", "-c",
                          "import time; print('EngineCore failed to start'); time.sleep(60)"],
                         tmp_path, tmp_path / "log.txt", timeout=5)
    assert result["exit_code"] == 124
    assert result["termination_reason"] == "terminal_engine_error"
    assert result["process_seconds"] < 5


def test_timeout_stops_silent_process(tmp_path):
    result = run_guarded([sys.executable, "-c", "import time; time.sleep(60)"],
                         tmp_path, tmp_path / "log.txt", timeout=0.2)
    assert result["termination_reason"] == "wall_time_limit"
