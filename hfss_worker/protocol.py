"""Job/result protocol shared by the ML controller (PC-1) and the HFSS worker (PC-2).

Pure standard library so it runs unchanged on both machines. It defines the
job states, the files a job or result package may contain, job-id and path
sanitisation, atomic JSON writes, the per-job state history, and HMAC signing
of job manifests.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import os
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

PROTOCOL_VERSION = "1.0"


class JobState(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    SOLVING = "SOLVING"
    EXTRACTING = "EXTRACTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in TERMINAL_STATES

    @property
    def failed(self) -> bool:
        return self in (JobState.FAILED, JobState.FAILED_VALIDATION, JobState.TIMEOUT)


TERMINAL_STATES = {JobState.COMPLETED, JobState.FAILED, JobState.FAILED_VALIDATION, JobState.TIMEOUT, JobState.CANCELLED}

# Allowed transitions; anything else is a protocol error.
TRANSITIONS: Dict[JobState, set] = {
    JobState.PENDING: {JobState.CLAIMED, JobState.CANCELLED},
    JobState.CLAIMED: {JobState.RUNNING, JobState.FAILED_VALIDATION, JobState.FAILED, JobState.CANCELLED},
    JobState.RUNNING: {JobState.SOLVING, JobState.EXTRACTING, JobState.COMPLETED, JobState.FAILED, JobState.FAILED_VALIDATION,
                       JobState.TIMEOUT, JobState.CANCELLED},
    JobState.SOLVING: {JobState.EXTRACTING, JobState.FAILED, JobState.FAILED_VALIDATION, JobState.TIMEOUT, JobState.CANCELLED},
    JobState.EXTRACTING: {JobState.SOLVING, JobState.COMPLETED, JobState.FAILED, JobState.FAILED_VALIDATION, JobState.TIMEOUT, JobState.CANCELLED},
    # retries: a FAILED / TIMEOUT attempt is kept in its own result_attempt_<n> folder and the job is claimed again
    JobState.FAILED: {JobState.CLAIMED},
    JobState.TIMEOUT: {JobState.CLAIMED},
}

# Simulation modes. MOCK results are software tests only and never physics.
SIMULATION_TYPES = ("HFSS", "MOCK", "DRY_RUN")

JOB_FILES = {
    "job.json", "job.sig", "parameters.json", "simulation_config.json", "template.json", "run_hfss.py",
    "metadata.json", "README.txt", "job_status.json", "prediction.json", "cancel.request",
    "geometry/geometry.json", "geometry/geometry.stl", "geometry/geometry.materials.json",
}
RESULT_FILES = {
    "result.json", "sparameters.csv", "rcs.csv", "farfield.csv", "solver_metadata.json", "validation.json",
    "hfss.log", "stdout.txt", "stderr.txt", "progress.json", "job_status.json",
}
RESULT_DIRS = {"plots"}
ALLOWED_PLOT_SUFFIXES = {".png", ".svg", ".csv"}
MAX_RESULT_FILE_BYTES = 200 * 1024 * 1024

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,95}$")


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sanitize_id(s: str) -> str:
    """Turn an arbitrary label into a safe job/candidate id ([A-Za-z0-9_-], max 96 chars)."""
    out = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s)).strip("_-")[:96]
    if not out:
        raise ValueError(f"cannot build an id from {s!r}")
    return out


def check_id(s: str) -> str:
    if not isinstance(s, str) or not _ID_RE.match(s):
        raise ValueError(f"invalid id {s!r}: only letters, digits, '_' and '-' (max 96) are allowed")
    return s


def safe_join(root: Path, rel: str) -> Path:
    """Join ``rel`` under ``root`` refusing absolute paths, drive letters and '..' (path traversal)."""
    if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")) or ":" in rel or "\x00" in rel:
        raise ValueError(f"unsafe path {rel!r}")
    parts = re.split(r"[\\/]+", rel)
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe path {rel!r}")
    root = Path(root).resolve()
    p = (root / Path(*parts)).resolve()
    if root != p and root not in p.parents:
        raise ValueError(f"path {rel!r} escapes {root}")
    return p


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON via a temp file + os.replace so readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


# ------------------------------------------------------------------ signing
def load_key(key: Optional[str] = None, key_file: Optional[str] = None, env: str = "MSO_JOB_KEY") -> Optional[bytes]:
    """Shared HMAC key: explicit value, key file, or environment variable (in that order)."""
    if key:
        return key.encode()
    if key_file and Path(key_file).exists():
        return Path(key_file).read_text(encoding="utf-8").strip().encode()
    v = os.environ.get(env)
    return v.encode() if v else None


def sign_manifest(job: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, canonical(job), hashlib.sha256).hexdigest()


def verify_signature(job: Mapping[str, Any], sig: str, key: bytes) -> bool:
    return hmac.compare_digest(sign_manifest(job, key), sig.strip())


# ------------------------------------------------------------------ job status history
def status_path(job_dir: Path) -> Path:
    return Path(job_dir) / "job_status.json"


def read_status(job_dir: Path) -> Dict[str, Any]:
    p = status_path(job_dir)
    if p.exists():
        return read_json(p)
    return {"state": JobState.PENDING.value, "history": [], "attempts": 0}


def set_state(job_dir: Path, state: JobState, note: str = "", force: bool = False, **extra: Any) -> Dict[str, Any]:
    """Record a state transition (validated against TRANSITIONS unless ``force``)."""
    st = read_status(job_dir)
    cur = JobState(st.get("state", "PENDING"))
    if not force and cur != state and state not in TRANSITIONS.get(cur, set()):
        raise ValueError(f"illegal job state transition {cur.value} -> {state.value}")
    st["state"] = state.value
    st.setdefault("history", []).append({"state": state.value, "time": now_iso(), "note": note, **extra})
    st.update({k: v for k, v in extra.items() if k in ("attempts", "worker", "result_dir", "error")})
    write_json_atomic(status_path(job_dir), st)
    return st


# ------------------------------------------------------------------ package checks
def list_files(root: Path) -> List[str]:
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def unexpected_job_files(job_dir: Path, extra_allowed: Iterable[str] = ()) -> List[str]:
    allowed = set(JOB_FILES) | set(extra_allowed)
    bad = []
    for rel in list_files(job_dir):
        top = rel.split("/")[0]
        if rel in allowed or top.startswith("result_") or rel.startswith("."):
            continue
        bad.append(rel)
    return bad


def unexpected_result_files(result_dir: Path) -> List[str]:
    bad = []
    for rel in list_files(result_dir):
        parts = rel.split("/")
        if len(parts) == 1 and rel in RESULT_FILES:
            continue
        if len(parts) == 2 and parts[0] in RESULT_DIRS and Path(rel).suffix.lower() in ALLOWED_PLOT_SUFFIXES:
            continue
        bad.append(rel)
    return bad
