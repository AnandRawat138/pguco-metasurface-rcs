"""Turns a validated result package into an :class:`EMResponse` with the right fidelity label.

HFSS results become ``VALIDATED_SIMULATION``; MOCK results become
``MOCK_SOLVER`` (software test only). Failed packages never become responses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hfss_worker.protocol import read_json
from hfss_worker.validator import validate_result_package
from metasurface_optimizer.core.provenance import Fidelity, Provenance
from metasurface_optimizer.data.loaders.target_loader import load_rcs
from metasurface_optimizer.electromagnetics.rcs import RCSData
from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.electromagnetics.sparameter import from_columns

C0 = 299_792_458.0
FIDELITY_OF = {"HFSS": Fidelity.SIMULATED_HF, "MOCK": Fidelity.MOCK}


@dataclass
class SimulationResult:
    job_id: str
    candidate_id: str
    state: str
    response: Optional[EMResponse] = None
    error: str = ""
    result_dir: Optional[str] = None
    attempts: int = 0
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.state == "COMPLETED" and self.response is not None


def _pec_plate_reference(rcs: RCSData, area_m2: float) -> np.ndarray:
    """Physical-optics specular RCS of a flat PEC plate: 4 pi A^2 cos^2(theta) / lambda^2, shape (P, I, F)."""
    lam = C0 / rcs.frequency
    ref = 4 * np.pi * area_m2 ** 2 * np.cos(np.radians(rcs.theta_inc))[:, None] ** 2 / lam[None, :] ** 2
    return np.broadcast_to(ref, (len(rcs.polarizations),) + ref.shape).copy()


def parse_sparameters(path: Path):
    df = pd.read_csv(path)
    inc = sorted({(float(t), float(p)) for t, p in zip(df["theta_inc"], df["phi_inc"])}, key=lambda tp: (abs(tp[0]), tp[1]))
    out = {}
    for t, p in inc:
        sub = df[(df["theta_inc"] == t) & (df["phi_inc"] == p)].sort_values("frequency")
        cols = {c: sub[c].to_numpy() for c in sub.columns if c.endswith(("_re", "_im"))}
        out[(t, p)] = from_columns(sub["frequency"].to_numpy(), cols, "hz")
    return out


def parse_result_package(result_dir: Path, job: Dict[str, Any], sim_cfg: Optional[Dict[str, Any]] = None,
                         template: Optional[Dict[str, Any]] = None) -> SimulationResult:
    result_dir = Path(result_dir)
    res = read_json(result_dir / "result.json")
    base = dict(job_id=job["job_id"], candidate_id=job["candidate_id"], result_dir=str(result_dir))
    if res.get("status") != "COMPLETED":
        return SimulationResult(state=res.get("status", "FAILED"), error=res.get("error") or "", **base)
    errs = validate_result_package(result_dir, job, sim_cfg)
    if errs:
        return SimulationResult(state="FAILED_VALIDATION", error="; ".join(errs), **base)
    stype = res.get("simulation_type")
    if stype not in FIDELITY_OF:
        return SimulationResult(state="FAILED_VALIDATION", error=f"unknown simulation_type {stype!r}", **base)
    fid = FIDELITY_OF[stype]
    sp = None
    by_inc: Dict[Any, Any] = {}
    if (result_dir / "sparameters.csv").is_file():
        by_inc = parse_sparameters(result_dir / "sparameters.csv")
        sp = by_inc[next(iter(by_inc))]  # normal incidence when present, else the smallest angle
        sp.port_definition = f"{stype} result, incidence theta={next(iter(by_inc))[0]:g} deg"
    rcs = None
    if (result_dir / "rcs.csv").is_file():
        rcs = load_rcs(result_dir / "rcs.csv", freq_unit="hz")
        ref = (template or {}).get("rcs_reference") or {}
        if rcs.reference is None and ref.get("kind") == "pec_plate" and ref.get("area_m2"):
            rcs.reference = _pec_plate_reference(rcs, float(ref["area_m2"]))
            rcs.reference_label = f"PEC plate {ref['area_m2']} m^2 (physical-optics formula, not simulated)"
    if sp is None and rcs is None:
        return SimulationResult(state="FAILED_VALIDATION", error="result has neither S-parameters nor RCS", **base)
    meta = read_json(result_dir / "solver_metadata.json") if (result_dir / "solver_metadata.json").is_file() else {}
    prov = Provenance(fid, f"{stype} job {job['job_id']}",
                      "MOCK solver: software test only, not physics" if fid is Fidelity.MOCK else "HFSS full-wave simulation via verified template",
                      {"job_id": job["job_id"], "candidate_id": job["candidate_id"], "template_id": job.get("template_id"),
                       "template_sha256": meta.get("template_sha256") or job.get("template_sha256"), "versions": res.get("versions"),
                       "runtime_s": res.get("runtime_s")})
    conds = {"incidences_with_sparameters": [list(k) for k in by_inc], "simulation_type": stype}
    if (result_dir / "farfield.csv").is_file():
        conds["farfield_csv"] = str(result_dir / "farfield.csv")
    resp = EMResponse(prov, sp, rcs, None, conds)
    return SimulationResult(state="COMPLETED", response=resp, info={"runtime_s": res.get("runtime_s"), "versions": res.get("versions")}, **base)


def load_validation_rows(result_dir: Path) -> List[Dict[str, Any]]:
    p = Path(result_dir) / "validation.json"
    return json.loads(p.read_text()).get("rows", []) if p.is_file() else []
