"""MOCK solver: exercises the whole HFSS job pipeline without HFSS.

It reads a job package exactly as ``run_hfss.py`` would and writes a result
package in the same format, computed with the synthetic analytic oracle. It
is deterministic (same job -> same numbers) and can emulate failures,
timeouts and invalid output so retries and validation can be tested.

Every file it writes says ``simulation_type: MOCK``; the parsed responses are
labeled ``MOCK_SOLVER``. MOCK results are a software test, never physics,
and must never be used for a scientific result.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from hfss_worker.protocol import now_iso, read_json, sha256_file, write_json_atomic
from hfss_worker.runner.runner_core import VALIDATION_STEPS
from metasurface_optimizer.solvers.analytic_oracle import AnalyticCheckerboardOracle

WARNING = "MOCK solver (synthetic analytic oracle): software test only, NOT physics, never use for scientific results"


def _oracle_conditions(sim_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    fr, ex = sim_cfg["frequency"], sim_cfg.get("excitation", {})
    cond = {"f_start": fr["start_hz"], "f_stop": fr["stop_hz"], "n_freq": int(fr["points"]),
            "theta_inc": list(ex.get("theta_deg", [0.0])), "polarizations": [p.upper() for p in ex.get("polarizations", ["TE"])]}
    cond.update((sim_cfg.get("mock") or {}).get("oracle_conditions", {}))
    return cond


def run_mock_job(job_dir: Path, result_dir: Path, cfg: Mapping[str, Any], on_progress: Optional[Callable[[str], None]] = None,
                 attempt: int = 1) -> Tuple[str, str]:
    job_dir, result_dir = Path(job_dir), Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    job = read_json(job_dir / "job.json")
    params = read_json(job_dir / "parameters.json")["parameters"]
    sim_cfg = read_json(job_dir / "simulation_config.json")
    knobs = sim_cfg.get("mock") or {}
    log = (result_dir / "hfss.log").open("a")
    log.write(f"{now_iso()} {WARNING}\n")

    def finish(state: str, error: Optional[str], files: Dict[str, str]) -> Tuple[str, str]:
        rows = [{"id": i, "name": n, "check": d, "status": "SKIP", "detail": "MOCK: no HFSS involved"} for i, n, d in VALIDATION_STEPS]
        if state == "COMPLETED":
            rows[10].update(status="PASS", detail="MOCK output finite and on the requested grid")
        write_json_atomic(result_dir / "validation.json", {"job_id": job["job_id"], "state": state, "rows": rows, "warning": WARNING})
        write_json_atomic(result_dir / "solver_metadata.json", {"simulation_type": "MOCK", "solver": "analytic oracle (mock)", "warning": WARNING,
                                                                  "attempt": attempt, "knobs": knobs, "runtime_s": round(time.time() - t0, 3)})
        write_json_atomic(result_dir / "result.json", {
            "protocol_version": job.get("protocol_version"), "job_id": job["job_id"], "candidate_id": job["candidate_id"], "kind": job.get("kind"),
            "status": state, "error": error, "simulation_type": "MOCK", "fidelity": "MOCK_SOLVER" if state == "COMPLETED" else None,
            "solver": "MOCK", "versions": {"mock": True}, "runtime_s": round(time.time() - t0, 3), "files": files,
            "frequency_unit": "Hz", "angle_unit": "deg", "rcs_unit": "m2", "warning": WARNING})
        log.write(f"{now_iso()} {state}{(': ' + error) if error else ''}\n")
        log.close()
        return state, error or ""

    if knobs.get("delay_s"):
        time.sleep(float(knobs["delay_s"]))
    if attempt <= int(knobs.get("fail_attempts", 0)):
        return finish("FAILED", f"MOCK emulated solver failure on attempt {attempt}", {})
    if knobs.get("timeout"):
        log.close()
        return "TIMEOUT", "MOCK emulated timeout"
    if on_progress:
        on_progress("SOLVING")
    values = {p["name"]: float(p.get("schema_value", p["value"])) for p in params}
    resp = AnalyticCheckerboardOracle(_oracle_conditions(sim_cfg)).evaluate(values)
    if on_progress:
        on_progress("EXTRACTING")
    files: Dict[str, str] = {}
    outputs = sim_cfg.get("outputs", ["sparameters"])
    if "sparameters" in outputs and resp.sparams is not None:
        sp = resp.sparams
        keys = sorted(sp.data)
        with (result_dir / "sparameters.csv").open("w", newline="") as f:
            w = csv.writer(f)
            head = ["theta_inc", "phi_inc", "frequency"]
            for (i, j, pol) in keys:
                k = f"S{i}{j}_{pol}"
                head += [k + "_re", k + "_im", k + "_mag", k + "_phase_deg"]
            w.writerow(head)
            for n, fz in enumerate(sp.frequency):
                row = [0.0, 0.0, float(fz)]
                for key in keys:
                    v = complex(sp.data[key][n])
                    if knobs.get("invalid_output"):
                        v = complex(float("nan"), 0.0)
                    row += [v.real, v.imag, abs(v), math.degrees(math.atan2(v.imag, v.real))]
                w.writerow(row)
        files["sparameters.csv"] = sha256_file(result_dir / "sparameters.csv")
    if "rcs" in outputs and resp.rcs is not None:
        r = resp.rcs
        with (result_dir / "rcs.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pol", "theta_inc", "phi_inc", "frequency", "theta_s", "phi_s", "rcs_m2", "rcs_dbsm", "ref_m2"])
            for p, pol in enumerate(r.polarizations):
                for i, th in enumerate(r.theta_inc):
                    for k, fz in enumerate(r.frequency):
                        ref = float(r.reference[p, i, k])
                        blk = r.bistatic[p, i, k]
                        for a, ts in enumerate(r.theta_s):
                            for b, ps in enumerate(r.phi_s):
                                s = float(blk[a, b])
                                w.writerow([pol, float(th), float(r.phi_inc), float(fz), float(ts), float(ps), s,
                                            10 * np.log10(s) if s > 0 else float("-inf"), ref])
        files["rcs.csv"] = sha256_file(result_dir / "rcs.csv")
    return finish("COMPLETED", None, files)
