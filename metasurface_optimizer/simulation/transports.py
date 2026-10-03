"""How job packages travel from PC-1 to PC-2 and results come back.

* :class:`ManualBatchTransport` - no network: write ``batch_NNN/`` (and a zip),
  carry it to PC-2, run ``worker.py --input batch_NNN``, carry
  ``batch_NNN_results/`` back and import it.
* :class:`SharedFolderTransport` - both PCs see one folder with
  ``pending/ running/ completed/ failed/ archive/``; workers claim jobs by
  atomic rename.
* :class:`LocalMockTransport` - manual batches executed immediately by the
  worker in MOCK mode on this machine (pipeline tests; never physics).

A network transport (REST: POST /jobs, GET /jobs/{id}, ...) is a planned
option and not implemented.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hfss_worker.protocol import JobState, now_iso, read_json, read_status, write_json_atomic

BATCH_README = """Batch {batch_id} - {n} job(s) for the HFSS PC ({simulation_type}).

On the HFSS PC:
    python hfss_worker/worker.py --input {batch_id}
This writes {batch_id}_results/ next to this folder. Bring that folder (or its zip) back and import it on PC-1.
"""


class ManualBatchTransport:
    kind = "manual"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _next_batch_id(self) -> str:
        n = 1 + max([int(p.name.split("_")[1]) for p in self.root.glob("batch_[0-9][0-9][0-9]") if p.is_dir()] or [0])
        return f"batch_{n:03d}"

    def submit(self, jobs: List[Any]) -> Dict[str, Any]:
        bid = self._next_batch_id()
        bdir = self.root / bid
        (bdir / "jobs").mkdir(parents=True)
        for j in jobs:
            shutil.copytree(j.job_dir, bdir / "jobs" / j.job_id)
            j.transport = {"kind": self.kind, "batch_id": bid}
            j.state = JobState.PENDING.value
        stype = jobs[0].simulation_type if jobs else "HFSS"
        write_json_atomic(bdir / "batch.json", {"batch_id": bid, "created": now_iso(), "simulation_type": stype,
                                                "jobs": [{"job_id": j.job_id, "candidate_id": j.candidate_id, "kind": j.kind} for j in jobs]})
        (bdir / "README.txt").write_text(BATCH_README.format(batch_id=bid, n=len(jobs), simulation_type=stype))
        zip_path = shutil.make_archive(str(self.root / bid), "zip", root_dir=self.root, base_dir=bid)
        return {"batch_id": bid, "batch_dir": str(bdir), "zip": zip_path}

    def results_root(self, batch_id: str) -> Path:
        return self.root / f"{batch_id}_results"

    def status(self, job: Any) -> str:
        rd = self.result_dir(job)
        if rd is None:
            return job.state
        try:
            return read_json(Path(rd) / "job_status.json")["state"]
        except Exception:
            return job.state

    def result_dir(self, job: Any) -> Optional[str]:
        bid = (job.transport or {}).get("batch_id")
        if not bid:
            return None
        d = self.results_root(bid) / job.job_id
        return str(d) if (d / "result.json").is_file() else None

    def import_folder(self, results_dir: Path, batch_id: str) -> Path:
        """Copy a results folder brought back from the HFSS PC into place (never overwriting an earlier import)."""
        dest = self.results_root(batch_id)
        src = Path(results_dir).resolve()
        if src == dest.resolve():
            return dest
        if dest.exists():
            raise FileExistsError(f"results for {batch_id} were already imported at {dest}")
        shutil.copytree(src, dest)
        return dest

    def cancel(self, job: Any) -> bool:
        return False  # the batch has left this PC; skip the job on the HFSS PC instead


class LocalMockTransport(ManualBatchTransport):
    """Manual batches run at once by the worker in MOCK mode (software test only)."""

    kind = "mock"

    def __init__(self, root: Path, worker_cfg: Optional[Mapping[str, Any]] = None):
        super().__init__(root)
        self.worker_cfg = dict(worker_cfg or {"max_retries": 2, "plots": False})

    def submit(self, jobs: List[Any]) -> Dict[str, Any]:
        info = super().submit(jobs)
        from hfss_worker.worker import run_batch
        run_batch(Path(info["batch_dir"]), self.worker_cfg, "MOCK")
        return info


class SharedFolderTransport:
    kind = "shared_folder"
    DIRS = ("pending", "running", "completed", "failed", "archive")

    def __init__(self, root: Path):
        self.root = Path(root)
        for d in self.DIRS + (".incoming", "workers"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def submit(self, jobs: List[Any]) -> Dict[str, Any]:
        for j in jobs:
            tmp = self.root / ".incoming" / j.job_id
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(j.job_dir, tmp)
            os.rename(tmp, self.root / "pending" / j.job_id)  # appears in pending/ atomically, never half-copied
            j.transport = {"kind": self.kind}
            j.state = JobState.PENDING.value
        return {"queued": [j.job_id for j in jobs], "queue": str(self.root)}

    def locate(self, job_id: str) -> Optional[Path]:
        for d in self.DIRS:
            p = self.root / d / job_id
            if p.is_dir():
                return p
        return None

    def status(self, job: Any) -> str:
        p = self.locate(job.job_id)
        if p is None:
            return job.state
        if p.parent.name == "pending":
            return JobState.PENDING.value
        return read_status(p).get("state", job.state)

    def result_dir(self, job: Any) -> Optional[str]:
        p = self.locate(job.job_id)
        if p is None or p.parent.name not in ("completed", "failed", "archive"):
            return None
        st = read_status(p)
        if not st.get("final"):
            return None
        rd = p / st.get("final_result_dir", "")
        return str(rd) if (rd / "result.json").is_file() else None

    def archive(self, job: Any) -> None:
        p = self.locate(job.job_id)
        if p is not None and p.parent.name in ("completed", "failed"):
            os.rename(p, self.root / "archive" / job.job_id)

    def cancel(self, job: Any) -> bool:
        p = self.locate(job.job_id)
        if p is None:
            return False
        if p.parent.name == "pending":
            dest = self.root / "archive" / job.job_id
            try:
                os.rename(p, dest)
            except OSError:
                return False  # a worker claimed it in the meantime
            from hfss_worker.protocol import set_state
            set_state(dest, JobState.CANCELLED, note="cancelled before it was claimed")
            return True
        if p.parent.name == "running":
            (p / "cancel.request").write_text(now_iso())
            return True
        return False

    def counts(self) -> Dict[str, int]:
        return {d: sum(1 for p in (self.root / d).iterdir() if p.is_dir()) for d in self.DIRS}

    def workers(self) -> List[Dict[str, Any]]:
        out = []
        for p in sorted((self.root / "workers").glob("*.json")):
            try:
                out.append(read_json(p))
            except Exception:
                continue
        return out
