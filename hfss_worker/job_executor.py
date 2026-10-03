"""Runs one job package to a final state, with validation, retries and timeouts.

Every attempt writes its own ``result_attempt_<n>/`` folder, so a failed
attempt is never overwritten; ``job_status.json`` keeps the full state history.
FAILED and TIMEOUT attempts are retried up to ``max_retries`` times;
FAILED_VALIDATION is not retried (it is a configuration problem, not bad luck).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from hfss_worker.hfss_runner import run_script
from hfss_worker.protocol import JobState, TRANSITIONS, now_iso, read_json, read_status, set_state, write_json_atomic
from hfss_worker.result_extractor import finalize, write_failure_result
from hfss_worker.validator import suite_passed, validate_job_package, validate_result_package


def _safe_progress(job_dir: Path) -> Callable[[str], None]:
    def cb(state: str) -> None:
        try:
            s = JobState(state)
        except ValueError:
            return
        cur = JobState(read_status(job_dir)["state"])
        if s in (JobState.SOLVING, JobState.EXTRACTING) and s in TRANSITIONS.get(cur, set()):
            set_state(job_dir, s)
    return cb


def execute_job(job_dir: Path, cfg: Mapping[str, Any], mode: str, worker_id: str = "", log: Callable[[str], None] = print) -> Dict[str, Any]:
    job_dir = Path(job_dir)
    mode = mode.upper()
    max_retries = int(cfg.get("max_retries", 2))
    try:
        job = read_json(job_dir / "job.json")
    except Exception as e:
        job = {"job_id": job_dir.name, "candidate_id": None, "simulation_type": mode}
        errs = [f"job.json unreadable: {e}"]
    else:
        errs = validate_job_package(job_dir, cfg, mode)
    st = read_status(job_dir)
    if st.get("final"):
        return {"job_id": job["job_id"], "state": st["state"], "attempts": st.get("attempts", 0), "result_dir": st.get("result_dir"), "skipped": "already final"}
    attempts = int(st.get("attempts", 0))
    if errs:
        attempts += 1
        rd = job_dir / f"result_attempt_{attempts}"
        set_state(job_dir, JobState.CLAIMED, force=True, attempts=attempts, worker=worker_id)
        write_failure_result(rd, job, JobState.FAILED_VALIDATION.value, "job package rejected: " + "; ".join(errs))
        set_state(job_dir, JobState.FAILED_VALIDATION, note="; ".join(errs)[:2000], result_dir=rd.name)
        _finish(job_dir, rd)
        log(f"{job['job_id']}: FAILED_VALIDATION - {errs[0]}")
        return {"job_id": job["job_id"], "state": "FAILED_VALIDATION", "attempts": attempts, "result_dir": str(rd), "errors": errs}
    sim_cfg = read_json(job_dir / "simulation_config.json")
    while True:
        attempts += 1
        rd = job_dir / f"result_attempt_{attempts}"
        if (job_dir / "cancel.request").exists():
            set_state(job_dir, JobState.CANCELLED, force=True, note="cancelled before start")
            write_failure_result(rd, job, "CANCELLED", "cancelled on request")
            state = "CANCELLED"
            break
        set_state(job_dir, JobState.CLAIMED, attempts=attempts, worker=worker_id, note=f"attempt {attempts}")
        set_state(job_dir, JobState.RUNNING)
        log(f"{job['job_id']}: attempt {attempts} running ({mode})")
        if mode == "HFSS":
            state, msg = run_script(job_dir, rd, cfg, _safe_progress(job_dir))
        else:
            from metasurface_optimizer.simulation.mock_solver import run_mock_job  # PC-1 / test machines only
            state, msg = run_mock_job(job_dir, rd, cfg, on_progress=_safe_progress(job_dir), attempt=attempts)
        if state in ("TIMEOUT", "CANCELLED") or not (rd / "result.json").is_file():
            write_failure_result(rd, job, state if state != "COMPLETED" else "FAILED", msg or "no result.json written")
            if state == "COMPLETED":
                state = "FAILED"
        rerrs = validate_result_package(rd, job, sim_cfg)
        if rerrs and state == "COMPLETED":
            state, msg = "FAILED_VALIDATION", "result package rejected: " + "; ".join(rerrs)
            write_failure_result(rd, job, state, msg)
        if state == "COMPLETED":
            finalize(rd, plots=bool(cfg.get("plots", True)))
        if not msg:
            try:
                msg = (read_json(rd / "result.json").get("error") or "")[:2000]
            except Exception:
                msg = ""
        set_state(job_dir, JobState(state), note=msg, result_dir=rd.name, force=state == "CANCELLED")
        log(f"{job['job_id']}: attempt {attempts} -> {state}{(' - ' + msg.splitlines()[0]) if msg else ''}")
        if state in ("FAILED", "TIMEOUT") and attempts <= max_retries:
            log(f"{job['job_id']}: retrying ({attempts}/{max_retries} retries used)")
            continue
        break
    if job.get("kind") == "verify_template" and mode == "HFSS":
        record_verification(job_dir, rd, cfg)
    _finish(job_dir, rd)
    return {"job_id": job["job_id"], "state": state, "attempts": attempts, "result_dir": str(rd)}


def _finish(job_dir: Path, rd: Path) -> None:
    st = read_status(job_dir)
    st["final"] = True
    st["final_result_dir"] = rd.name
    st["finished"] = now_iso()
    write_json_atomic(job_dir / "job_status.json", st)


def record_verification(job_dir: Path, rd: Path, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Store the outcome of a template verification run; candidate jobs are gated on it."""
    tmpl = read_json(job_dir / "template.json")
    job = read_json(job_dir / "job.json")
    try:
        rows = read_json(rd / "validation.json")["rows"]
    except Exception:
        rows = []
    res = read_json(rd / "result.json")
    try:
        sha = read_json(rd / "solver_metadata.json").get("template_sha256") or tmpl.get("sha256")
    except Exception:
        sha = tmpl.get("sha256")
    passed = res.get("status") == "COMPLETED" and suite_passed(rows)
    path = Path(cfg.get("validation_record", "worker_state/validation_suite.json"))
    rec = read_json(path) if path.is_file() else {"templates": {}}
    entry = {"template_id": tmpl["template_id"], "sha256": sha, "aedt_version": job.get("aedt_version"),
             "versions": res.get("versions"), "passed": passed, "rows": rows, "date": now_iso(), "job_id": job["job_id"]}
    rec.setdefault("templates", {})[tmpl["template_id"]] = entry
    rec.setdefault("history", []).append({k: entry[k] for k in ("template_id", "passed", "date", "job_id")})
    write_json_atomic(path, rec)
    return entry
