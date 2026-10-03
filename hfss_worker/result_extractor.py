"""Completes a result package on the worker: failure records, optional plots, file hashes."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hfss_worker.protocol import now_iso, read_json, sha256_file, write_json_atomic


def write_failure_result(result_dir: Path, job: Mapping[str, Any], state: str, error: str,
                         validation_rows: Optional[list] = None) -> None:
    """Result package for a job that failed before or outside run_hfss.py (validation, timeout, crash)."""
    result_dir.mkdir(parents=True, exist_ok=True)
    rp = result_dir / "result.json"
    prev: Dict[str, Any] = {}
    if rp.is_file():
        try:
            prev = read_json(rp)
        except Exception:
            prev = {}
    res = {**prev, "protocol_version": job.get("protocol_version"), "job_id": job.get("job_id"), "candidate_id": job.get("candidate_id"),
           "kind": job.get("kind"), "status": state, "error": error, "simulation_type": job.get("simulation_type"),
           "fidelity": None, "solver": job.get("solver"), "finished": now_iso()}
    res.setdefault("files", {})
    write_json_atomic(rp, res)
    if validation_rows is not None:
        write_json_atomic(result_dir / "validation.json", {"job_id": job.get("job_id"), "state": state, "rows": validation_rows})
    log = result_dir / "hfss.log"
    with log.open("a") as f:
        f.write(f"{now_iso()} {state}: {error}\n")


def make_plots(result_dir: Path) -> Dict[str, str]:
    """Quick-look plots (needs matplotlib; silently skipped otherwise). Returns {rel_path: sha256}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    out: Dict[str, str] = {}
    pdir = result_dir / "plots"
    sp = result_dir / "sparameters.csv"
    if sp.is_file():
        with sp.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            import math
            keys = sorted({k[:-4] for k in rows[0] if k.endswith("_mag")})
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for (th, ph) in sorted({(r["theta_inc"], r["phi_inc"]) for r in rows}, key=lambda t: (float(t[0]), float(t[1]))):
                sub = [r for r in rows if (r["theta_inc"], r["phi_inc"]) == (th, ph)]
                fx = [float(r["frequency"]) / 1e9 for r in sub]
                for k in keys:
                    ax.plot(fx, [20 * math.log10(max(float(r[k + "_mag"]), 1e-12)) for r in sub], label=f"{k} θ={float(th):g}°")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("|S| (dB)")
            ax.set_title("HFSS result (VALIDATED SIMULATION)", fontsize=9)
            ax.legend(fontsize=6)
            fig.tight_layout()
            pdir.mkdir(exist_ok=True)
            fig.savefig(pdir / "sparameters.png", dpi=110)
            plt.close(fig)
    rc = result_dir / "rcs.csv"
    if rc.is_file():
        with rc.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for key in sorted({(r["pol"], r["theta_inc"]) for r in rows}):
                sub = [r for r in rows if (r["pol"], r["theta_inc"]) == key]
                ax.plot([float(r["frequency"]) / 1e9 for r in sub], [float(r["rcs_dbsm"]) for r in sub], label=f"{key[0]} θ={float(key[1]):g}°")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("RCS (dBsm)")
            ax.legend(fontsize=6)
            fig.tight_layout()
            pdir.mkdir(exist_ok=True)
            fig.savefig(pdir / "rcs.png", dpi=110)
            plt.close(fig)
    for p in sorted(pdir.glob("*.png")) if pdir.is_dir() else []:
        out[f"plots/{p.name}"] = sha256_file(p)
    return out


def finalize(result_dir: Path, plots: bool = True) -> None:
    rp = result_dir / "result.json"
    res = read_json(rp)
    if plots and res.get("status") == "COMPLETED":
        res.setdefault("files", {}).update(make_plots(result_dir))
        write_json_atomic(rp, res)
