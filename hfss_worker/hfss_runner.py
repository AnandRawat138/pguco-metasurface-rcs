"""Launches a job's ``run_hfss.py`` in AEDT's Python with a timeout, cancellation and progress mirroring."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

from hfss_worker.protocol import JobState, read_json


def kill_tree(proc: subprocess.Popen) -> None:
    """Terminate the runner and everything it started (AEDT itself is a child process)."""
    try:
        import psutil  # optional
        parent = psutil.Process(proc.pid)
        for ch in parent.children(recursive=True):
            try:
                ch.kill()
            except psutil.Error:
                pass
        parent.kill()
        return
    except ImportError:
        pass
    except Exception:
        pass
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def run_script(job_dir: Path, result_dir: Path, cfg: Mapping[str, Any],
               on_progress: Optional[Callable[[str], None]] = None) -> Tuple[str, str]:
    """Run run_hfss.py; returns (state, message). Never raises for solver problems."""
    py = cfg.get("python_executable") or sys.executable
    timeout_s = float(cfg.get("max_runtime_minutes", 120)) * 60.0
    cmd = [py, str(Path(job_dir) / "run_hfss.py"), "--job", str(job_dir), "--out", str(result_dir), "--cores", str(int(cfg.get("cores", 4)))]
    if not cfg.get("non_graphical", True):
        cmd.append("--graphical")
    if cfg.get("keep_project"):
        cmd.append("--keep-project")
    result_dir.mkdir(parents=True, exist_ok=True)
    out = (result_dir / "stdout.txt").open("w")
    err = (result_dir / "stderr.txt").open("w")
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(cmd, stdout=out, stderr=err, cwd=str(job_dir), **kw)
    t0, last = time.time(), None
    state, msg = None, ""
    try:
        while proc.poll() is None:
            time.sleep(float(cfg.get("progress_poll_s", 2.0)))
            prog = result_dir / "progress.json"
            if prog.is_file():
                try:
                    st = read_json(prog).get("state")
                except Exception:
                    st = None
                if st and st != last and on_progress:
                    on_progress(st)
                last = st or last
            if (Path(job_dir) / "cancel.request").exists():
                kill_tree(proc)
                state, msg = JobState.CANCELLED.value, "cancelled on request"
                break
            if time.time() - t0 > timeout_s:
                kill_tree(proc)
                state, msg = JobState.TIMEOUT.value, f"exceeded max_runtime_minutes={cfg.get('max_runtime_minutes', 120)}"
                break
        proc.wait(timeout=60)
    finally:
        out.close()
        err.close()
    if state is None:
        rc = proc.returncode
        try:
            state = read_json(result_dir / "result.json").get("status")
        except Exception:
            state = None
        if state not in ("COMPLETED", "FAILED", "FAILED_VALIDATION"):
            state = {0: "COMPLETED", 2: "FAILED_VALIDATION"}.get(rc, "FAILED")
            msg = f"runner exited with code {rc} without a readable result.json"
            if state == "COMPLETED":
                state = "FAILED"
    return state, msg
