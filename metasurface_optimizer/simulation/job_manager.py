"""SimulationManager: the loop between the optimizer and the high-fidelity solver.

    candidates = optimizer.generate_candidates(k)
    jobs       = manager.create_jobs(candidates)        # validated, packaged, budget-checked
    manager.submit(jobs)                                 # manual batch / shared folder / mock
    results    = manager.collect_results()              # parsed, validated, never overwritten
    optimizer.update(results)                            # failures excluded, predicted vs simulated logged

``step()`` runs one round of that. Budget: every submitted job counts as one
high-fidelity simulation (worker-side retries of the same job do not count
again; a manual retry is a new job and does). The budget is never exceeded.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hfss_worker.protocol import JobState, now_iso, read_json, write_json_atomic
from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.simulation.baseline import compare_responses
from metasurface_optimizer.simulation.candidate_validation import validate_candidate
from metasurface_optimizer.simulation.feedback import candidate_feedback, summarize
from metasurface_optimizer.simulation.result_parser import SimulationResult
from metasurface_optimizer.simulation.solver_adapter import SimulationJob


class BaselineNotReproduced(RuntimeError):
    pass


@dataclass
class SimulationBudget:
    max_simulations: int
    batch_size: int = 4
    max_iterations: Optional[int] = None


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else (float(v) if isinstance(v, np.floating) else str(v))))


class SimulationManager:
    def __init__(self, optimizer, adapter, study_dir: Path, budget: SimulationBudget,
                 baseline_reference: Optional[EMResponse] = None, baseline_thresholds: Optional[Dict[str, float]] = None):
        self.opt = optimizer
        self.adapter = adapter
        self.mode = adapter.simulation_type
        self.study_dir = Path(study_dir)
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.baseline_reference = baseline_reference
        self.baseline_thresholds = baseline_thresholds
        self.jobs: Dict[str, SimulationJob] = {}
        self.results: Dict[str, SimulationResult] = {}
        self.batches: List[Dict[str, Any]] = []
        self.feedback: List[Dict[str, Any]] = []
        self.convergence: List[Dict[str, Any]] = []
        self.rejected: List[Dict[str, Any]] = []
        self.retry_queue: List[str] = []
        self.baseline_check: Optional[Dict[str, Any]] = None
        self.override: Optional[Dict[str, Any]] = None
        self.stop_reason: Optional[str] = None
        self.opt.cfg.stopping.max_hf_evaluations = budget.max_simulations
        if baseline_reference is not None and baseline_reference.fidelity.is_prediction:
            raise ValueError("the baseline reference must be a simulation, measurement or reported data, not PREDICTED")

    # ------------------------------------------------------------ budget
    @property
    def used(self) -> int:
        return sum(1 for j in self.jobs.values() if j.submitted)

    @property
    def remaining(self) -> int:
        return max(self.budget.max_simulations - self.used, 0)

    def budget_report(self) -> Dict[str, Any]:
        states: Dict[str, int] = {}
        for j in self.jobs.values():
            states[j.state] = states.get(j.state, 0) + 1
        return {"mode": self.mode, "max_simulations": self.budget.max_simulations, "used": self.used, "remaining": self.remaining,
                "batch_size": self.budget.batch_size, "max_iterations": self.budget.max_iterations,
                "iterations_done": sum(1 for b in self.batches if b["phase"] == "iterate" and b.get("closed")),
                "completed": sum(1 for j in self.jobs.values() if j.state == "COMPLETED" and j.imported),
                "failed": sum(1 for j in self.jobs.values() if j.imported and j.state != "COMPLETED"),
                "worker_attempts": sum(j.attempts for j in self.jobs.values()), "jobs_by_state": states,
                "dry_run_packages": sum(1 for j in self.jobs.values() if j.simulation_type == "DRY_RUN"),
                "rejected_candidates": len(self.rejected), "stop_reason": self.stop_reason,
                "note": "each submitted job is one simulation; worker retries of the same job are not charged again"}

    # ------------------------------------------------------------ candidates -> jobs
    def _cid(self, phase: str, batch_no: int, n: int) -> str:
        prefix = {"baseline": "b", "initial": "i", "iterate": "c", "retry": "r"}[phase]
        return f"{prefix}{batch_no:03d}_{n:02d}"

    def candidate_dir(self, cid: str) -> Path:
        return self.study_dir / "candidates" / cid

    def _write_predicted(self, cid: str, cand: Dict[str, Any]) -> None:
        d = self.candidate_dir(cid) / "predicted"
        d.mkdir(parents=True, exist_ok=True)
        write_json_atomic(d / "parameters.json", {"values": _jsonable(cand["values"]), "units": {p.name: p.unit for p in self.opt.schema.parameters}})
        if cand.get("mu_J") is None:
            write_json_atomic(d / "prediction.json", {"label": "NO PREDICTION", "reason": f"{cand.get('kind')} point: simulated before any surrogate exists"})
            return
        doc = {"label": "PREDICTED (surrogate; not validated)", "objective_names": list(self.opt.problem.objectives.names),
               **{k: _jsonable(cand.get(k)) for k in ("mu", "sigma", "mu_J", "sigma_J", "kind", "reason", "predicted_effect", "modifications")}}
        write_json_atomic(d / "prediction.json", doc)
        try:
            pred = self.opt.predict(np.asarray(cand["x"], float)[None])
            dec = self.opt.encoder.decode(pred["Y"][0], pred["S"][0])
            rows = []
            for i, f in enumerate(dec["frequency"]):
                row = {"label": "PREDICTED", "frequency_hz": float(f)}
                for name, s in self.opt.encoder.slices.items():
                    if name != "J":
                        row[name] = float(pred["Y"][0][s][i])
                        row[name + "_std"] = float(pred["S"][0][s][i])
                rows.append(row)
            import csv
            with (d / "predicted_response.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        except Exception as e:  # prediction export is informative only
            (d / "predicted_response_error.txt").write_text(str(e))

    def create_jobs(self, candidates: List[Dict[str, Any]], phase: Optional[str] = None, batch_no: Optional[int] = None) -> List[SimulationJob]:
        phase = phase or self.opt.phase
        batch_no = len(self.batches) if batch_no is None else batch_no
        valid = []
        for n, c in enumerate(candidates):
            cid = c.get("candidate_id") or self._cid(phase, batch_no, n)
            chk = validate_candidate(np.asarray(c["x"], float), self.opt.schema, self.opt.problem.validator, self.adapter.template,
                                     {**self.adapter.simulation_config, "simulation_type": self.mode})
            if not chk["valid"]:
                self.rejected.append({"candidate_id": cid, "phase": phase, "values": _jsonable(c.get("values")), **chk})
                self.candidate_dir(cid).mkdir(parents=True, exist_ok=True)
                write_json_atomic(self.candidate_dir(cid) / "rejected.json", {"values": _jsonable(c.get("values")), **chk})
                continue
            valid.append((cid, c, chk))
        if self.mode != "DRY_RUN" and len(valid) > self.remaining:
            for cid, c, _ in valid[self.remaining:]:
                self.rejected.append({"candidate_id": cid, "phase": phase, "values": _jsonable(c.get("values")), "valid": True,
                                      "errors": ["simulation budget exhausted"], "checks": {"budget": "FAIL"}})
            valid = valid[: self.remaining]
        jobs = []
        for cid, c, chk in valid:
            self._write_predicted(cid, c)
            job = self.adapter.create_job(c, phase if phase != "iterate" else "candidate", self.opt.iteration, cid)
            job.tag = c.get("tag", cid)
            if self.mode == "DRY_RUN":
                job.simulation_type = "DRY_RUN"
            self.jobs[job.job_id] = job
            jobs.append(job)
        return jobs

    def submit(self, jobs: List[SimulationJob]) -> Dict[str, Any]:
        if self.mode == "DRY_RUN":
            return {"dry_run": True, "packages": [j.job_dir for j in jobs]}
        if self.used + sum(1 for j in jobs if not j.submitted) > self.budget.max_simulations:
            raise RuntimeError("submitting these jobs would exceed the simulation budget")
        return self.adapter.submit(jobs)

    # ------------------------------------------------------------ one round
    @property
    def open_batch(self) -> Optional[Dict[str, Any]]:
        return next((b for b in self.batches if not b.get("closed")), None)

    def generate_batch(self, k: Optional[int] = None) -> Dict[str, Any]:
        if self.open_batch is not None:
            raise RuntimeError(f"batch {self.open_batch['batch_no']} is still waiting for results")
        if self.stop_reason:
            raise RuntimeError(f"study stopped: {self.stop_reason}")
        if self.mode != "DRY_RUN" and self.remaining <= 0:
            self.stop_reason = "budget exhausted"
            raise RuntimeError("simulation budget exhausted")
        phase = self.opt.phase
        if phase == "iterate":
            self._check_baseline_gate()
            done = sum(1 for b in self.batches if b["phase"] == "iterate" and b.get("closed"))
            if self.budget.max_iterations is not None and done >= self.budget.max_iterations:
                self.stop_reason = "max iterations reached"
                raise RuntimeError(self.stop_reason)
        batch_no = len(self.batches)
        retries = [self.jobs[j] for j in self.retry_queue] if phase == "iterate" else []
        self.retry_queue = [] if retries else self.retry_queue
        cap = self.remaining if self.mode != "DRY_RUN" else 10 ** 6
        if phase == "iterate":
            k = max(min(k or self.budget.batch_size, cap) - len(retries), 0)
            cands = self.opt.generate_candidates(k) if k > 0 else []
        else:
            cands = self.opt.generate_candidates(cap if phase == "initial" else None)
        retry_cands = [{"x": np.asarray(j.x), "values": j.values, "kind": "retry", "tag": f"{j.tag}_retry",
                        "candidate_id": f"{j.candidate_id.split('_r')[0]}_r{1 + sum(1 for x in self.jobs.values() if x.candidate_id.startswith(j.candidate_id.split('_r')[0] + '_r'))}"}
                       for j in retries]
        jobs = self.create_jobs(retry_cands, "retry", batch_no) + self.create_jobs(cands, phase, batch_no)
        if not jobs:
            raise RuntimeError("no valid candidate left to simulate (all rejected); see manager.rejected")
        info = self.submit(jobs)
        batch = {"batch_no": batch_no, "phase": phase, "job_ids": [j.job_id for j in jobs], "created": now_iso(),
                 "transport": _jsonable(info), "closed": False, "iteration": self.opt.iteration}
        if self.mode == "DRY_RUN":
            batch.update(closed=True, closed_at=now_iso(), stop_reason=None, note="DRY RUN: packages generated for inspection; nothing simulated")
        self.batches.append(batch)
        self.save()
        return batch

    def _check_baseline_gate(self) -> None:
        if self.mode != "HFSS" or self.override is not None:
            return
        if self.baseline_reference is None:
            return  # nothing to compare with; reported in the status as "baseline not checked"
        if self.baseline_check is None or not self.baseline_check.get("passed"):
            raise BaselineNotReproduced((self.baseline_check or {}).get("message", "baseline reproduction has not been checked yet"))

    def override_baseline_gate(self, reason: str, user: str = "") -> None:
        if not reason.strip():
            raise ValueError("a reason is required to override the baseline-reproduction gate")
        self.override = {"reason": reason, "user": user, "time": now_iso(), "baseline_check": self.baseline_check}
        self.save()

    # ------------------------------------------------------------ results
    def refresh(self) -> None:
        for j in self.jobs.values():
            if j.submitted and not j.imported:
                j.state = self.adapter.get_status(j)

    def _import(self, job: SimulationJob, res: SimulationResult) -> None:
        dest = self.candidate_dir(job.candidate_id) / "validated" / job.job_id
        if dest.exists():
            raise FileExistsError(f"{dest} already exists; results are never overwritten")
        shutil.copytree(res.result_dir, dest)
        js = Path(res.result_dir).parent / "job_status.json"
        if not js.is_file():
            js = Path(res.result_dir) / "job_status.json"
        if js.is_file():
            st = read_json(js)
            job.attempts = int(st.get("attempts", 0))
            if not (dest / "job_status.json").is_file():
                shutil.copy2(js, dest / "job_status.json")
        res.result_dir = str(dest)
        job.state, job.error, job.result_dir, job.imported = res.state, res.error, str(dest), True
        label = res.response.fidelity.display if res.response is not None else "NOT A RESULT (simulation failed)"
        write_json_atomic(dest.parent.parent / f"validated_{job.job_id}_summary.json",
                          {"job_id": job.job_id, "state": res.state, "label": label, "error": res.error, "attempts": job.attempts})
        self.results[job.job_id] = res

    def collect_results(self) -> List[SimulationResult]:
        """Import every finished job of the open batch; when the whole batch is final, feed it to the optimizer."""
        b = self.open_batch
        if b is None:
            return []
        new = []
        for jid in b["job_ids"]:
            job = self.jobs[jid]
            if job.imported:
                continue
            res = self.adapter.collect_result(job)
            if res is not None:
                self._import(job, res)
                new.append(res)
            else:
                job.state = self.adapter.get_status(job)
        if all(self.jobs[j].imported for j in b["job_ids"]):
            self._close_batch(b)
        self.save()
        return [self.results[j] for j in b["job_ids"] if j in self.results]

    def import_batch(self, results_dir: Path, batch_id: Optional[str] = None) -> List[SimulationResult]:
        """Manual transfer: import a ``batch_NNN_results`` folder brought back from the HFSS PC."""
        b = self.open_batch
        if b is None:
            raise RuntimeError("no batch is waiting for results")
        bid = batch_id or (b.get("transport") or {}).get("batch_id")
        known = set(b["job_ids"])
        found = {p.name for p in Path(results_dir).iterdir() if p.is_dir()}
        unknown = found - known
        if unknown and not (found & known):
            raise ValueError(f"{results_dir} has no result for this batch (found {sorted(found)[:5]})")
        self.adapter.transport.import_folder(Path(results_dir), bid)
        return self.collect_results()

    def _close_batch(self, b: Dict[str, Any]) -> None:
        payload = []
        for jid in b["job_ids"]:
            job, res = self.jobs[jid], self.results[jid]
            if res.ok and job.prediction and job.prediction.get("mu_J") is not None:
                fb = candidate_feedback(self.opt, np.asarray(job.x), res.response, job.prediction)
                self.feedback.append({"job_id": jid, "candidate_id": job.candidate_id, "batch_no": b["batch_no"], "n_hfss": self.used,
                                      "simulation_type": job.simulation_type, **_jsonable(fb)})
                write_json_atomic(self.candidate_dir(job.candidate_id) / "prediction_error.json", _jsonable(fb))
            payload.append({"x": np.asarray(job.x), "response": res.response if res.ok else None, "tag": job.tag,
                            "state": res.state, "error": res.error, "job_id": jid})
        if b["phase"] == "baseline":
            base = next((p for p in payload if p["response"] is not None), None)
            if base is not None and self.baseline_reference is not None:
                self.baseline_check = compare_responses(self.baseline_reference, base["response"], self.baseline_thresholds)
        reason = self.opt.update(payload)
        hf = self.opt.hf_records() if self.opt.phase != "baseline" else []
        best = min((self.opt.composite_of(r.J, r.x) for r in hf), default=None) if self.opt.phase == "iterate" else None
        fb_b = [f for f in self.feedback if f["batch_no"] == b["batch_no"]]
        self.convergence.append({"batch_no": b["batch_no"], "phase": b["phase"], "n_hfss": self.used, "n_valid_results": len(hf),
                                 "J_best": best, "n_failed": sum(1 for p in payload if p["response"] is None),
                                 "batch_composite_mae": float(np.mean([f["composite_abs_error"] for f in fb_b])) if fb_b else None,
                                 "batch_response_rmse_db": float(np.nanmean([f["response_rmse_db"] for f in fb_b if f.get("response_rmse_db") is not None]))
                                 if any(f.get("response_rmse_db") is not None for f in fb_b) else None,
                                 "simulation_type": self.mode})
        b["closed"], b["closed_at"], b["stop_reason"] = True, now_iso(), reason
        if reason:
            self.stop_reason = reason
        elif self.remaining <= 0 and self.mode != "DRY_RUN":
            self.stop_reason = "budget exhausted"

    # ------------------------------------------------------------ control
    def step(self, k: Optional[int] = None) -> Dict[str, Any]:
        """Collect the open batch if any, else generate and submit the next one."""
        if self.open_batch is not None:
            got = self.collect_results()
            return {"collected": len(got), "batch_closed": self.open_batch is None, "stop_reason": self.stop_reason}
        b = self.generate_batch(k)
        return {"submitted": b["job_ids"], "batch_no": b["batch_no"], "phase": b["phase"]}

    def run(self, max_batches: int = 1000, poll_s: float = 0.0, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Drive the loop until a stopping rule fires (mock or shared-folder transports)."""
        t0 = time.time()
        for _ in range(max_batches):
            if self.stop_reason:
                break
            if self.open_batch is None:
                try:
                    self.generate_batch()
                except BaselineNotReproduced:
                    raise
                except RuntimeError as e:
                    self.stop_reason = self.stop_reason or str(e)
                    break
            while self.open_batch is not None:
                self.collect_results()
                if self.open_batch is None:
                    break
                if timeout_s is not None and time.time() - t0 > timeout_s:
                    raise TimeoutError("results did not arrive in time")
                time.sleep(poll_s or 1.0)
        self.save()
        return self.summary()

    def cancel(self, job_id: str) -> bool:
        job = self.jobs[job_id]
        ok = bool(self.adapter.transport and self.adapter.transport.cancel(job))
        if ok:
            job.state = JobState.CANCELLED.value
        self.save()
        return ok

    def retry(self, job_id: str) -> None:
        """Queue a failed candidate for a fresh job in the next batch (charged to the budget as a new simulation)."""
        job = self.jobs[job_id]
        if job.state not in ("FAILED", "FAILED_VALIDATION", "TIMEOUT", "CANCELLED"):
            raise ValueError(f"job {job_id} is {job.state}; only failed jobs can be retried")
        if self.remaining <= len(self.retry_queue):
            raise RuntimeError("no simulation budget left for a retry")
        self.retry_queue.append(job_id)
        self.save()

    # ------------------------------------------------------------ reporting
    def summary(self) -> Dict[str, Any]:
        return {"budget": self.budget_report(), "stop_reason": self.stop_reason, "phase": self.opt.phase,
                "baseline_check": self.baseline_check, "override": self.override, "feedback_summary": summarize(self.feedback),
                "convergence": self.convergence, "n_failed_excluded": len(getattr(self.opt, "failed_evaluations", [])),
                "fidelity": {"HFSS": Fidelity.SIMULATED_HF.display, "MOCK": Fidelity.MOCK.display, "DRY_RUN": "no simulation"}[self.mode]}

    def jobs_table(self) -> List[Dict[str, Any]]:
        out = []
        for j in self.jobs.values():
            d = {k: v for k, v in j.to_dict().items() if k not in ("x",)}
            d["label"] = {"COMPLETED": Fidelity.SIMULATED_HF.display if j.simulation_type == "HFSS" else Fidelity.MOCK.display}.get(j.state, j.state)
            out.append(d)
        return out

    def save(self) -> None:
        write_json_atomic(self.study_dir / "simulation_state.json", _jsonable({
            "mode": self.mode, "budget": asdict(self.budget), "budget_report": self.budget_report(), "jobs": self.jobs_table(),
            "batches": self.batches, "rejected": self.rejected, "feedback": self.feedback, "convergence": self.convergence,
            "baseline_check": self.baseline_check, "override": self.override, "stop_reason": self.stop_reason,
            "template": {"template_id": self.adapter.template.template_id, "status": self.adapter.template.status()}}))
        write_json_atomic(self.study_dir / "simulation_budget_report.json", self.budget_report())

    def plot_convergence(self, path: Path) -> Optional[Path]:
        rows = [r for r in self.convergence if r["J_best"] is not None]
        if not rows:
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lab = {"HFSS": "HFSS (validated)", "MOCK": "MOCK (software test, not physics)"}.get(self.mode, self.mode)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        axes[0].plot([r["n_hfss"] for r in rows], [r["J_best"] for r in rows], marker="o", lw=2, color="#2a6fdb")
        axes[0].set_xlabel("simulations used (N_HFSS)")
        axes[0].set_ylabel("best composite objective J_best")
        axes[0].set_title(f"Best {lab} design", fontsize=9)
        er = [r for r in self.convergence if r["batch_composite_mae"] is not None]
        if er:
            axes[1].plot([r["n_hfss"] for r in er], [r["batch_composite_mae"] for r in er], marker="o", lw=2, color="#d9480f")
        axes[1].set_xlabel("simulations used (N_HFSS)")
        axes[1].set_ylabel("|J_sim - J_PREDICTED| (batch mean)")
        axes[1].set_title("Surrogate prediction error (out of sample)", fontsize=9)
        for ax in axes:
            ax.grid(alpha=0.3)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
