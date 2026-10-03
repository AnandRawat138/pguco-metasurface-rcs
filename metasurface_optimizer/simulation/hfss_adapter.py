"""HFSS adapter (PC-1 side): candidates -> job packages; result packages -> responses.

It writes data, never PyAEDT calls: the only PyAEDT code is the fixed,
self-contained ``run_hfss.py`` (see ``hfss_worker.runner``) that every job
carries and the worker verifies byte-for-byte before running it.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from hfss_worker.package import write_job_package
from hfss_worker.protocol import now_iso, read_json, sanitize_id
from hfss_worker.runner.script import RUNNER_VERSION, build_run_hfss_script
from hfss_worker.validator import validate_result_package
from metasurface_optimizer.geometry.parameter_schema import ParameterSchema
from metasurface_optimizer.simulation.result_parser import SimulationResult, parse_result_package
from metasurface_optimizer.simulation.solver_adapter import SimulationJob, SolverAdapter
from metasurface_optimizer.simulation.templates import KINDS, TemplateSpec

FRAMEWORK_VERSION = "0.2.0"
UNIT_ALIASES = {"degree": "deg", "degrees": "deg", "1": "", "none": "", "-": "", "micron": "um"}


def aedt_unit(schema_unit: str) -> str:
    u = str(schema_unit or "").strip()
    return UNIT_ALIASES.get(u.lower(), u)


def make_simulation_config(template: TemplateSpec, f_start_hz: float, f_stop_hz: float, n_points: int,
                           polarizations=("TE", "TM"), theta_deg=(0.0,), phi_deg=(0.0,), outputs: Optional[List[str]] = None,
                           passivity_tol: float = 0.05, mock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Solver-independent simulation configuration (frequency, excitation, outputs, checks)."""
    outs = outputs or (["sparameters"] if template.simulation_type == "periodic_fss" else ["rcs"])
    cfg = {"frequency": {"start_hz": float(f_start_hz), "stop_hz": float(f_stop_hz), "points": int(n_points)},
           "excitation": {"type": KINDS[template.simulation_type], "polarizations": [str(p).upper() for p in polarizations],
                          "theta_deg": [float(t) for t in theta_deg], "phi_deg": [float(p) for p in phi_deg]},
           "outputs": list(outs), "checks": {"passivity_tol": float(passivity_tol)}}
    if mock:
        cfg["mock"] = mock
    return cfg


class HFSSAdapter(SolverAdapter):
    solver_name = "HFSS"

    def __init__(self, template: TemplateSpec, schema: ParameterSchema, simulation_config: Dict[str, Any], jobs_root: Path,
                 transport=None, simulation_type: str = "HFSS", aedt_version: str = "2024.2", key: Optional[bytes] = None,
                 generator=None, include_stl: bool = False, study: str = "study"):
        super().__init__(transport, simulation_type)
        if simulation_type not in ("HFSS", "MOCK", "DRY_RUN"):
            raise ValueError("simulation_type must be HFSS, MOCK or DRY_RUN")
        self.template = template
        self.schema = schema
        self.simulation_config = simulation_config
        self.jobs_root = Path(jobs_root)
        self.aedt_version = aedt_version
        self.key = key
        self.generator = generator
        self.include_stl = include_stl
        self.study = sanitize_id(study)

    # -------------------------------------------------------------- script
    @staticmethod
    def generate_pyaedt_script() -> str:
        """The self-contained run_hfss.py (identical for every job; job data lives in JSON)."""
        return build_run_hfss_script()

    # -------------------------------------------------------------- packages
    def parameters_for(self, x: np.ndarray) -> List[Dict[str, Any]]:
        out = []
        for spec in self.schema.parameters:
            m = self.template.parameter_mapping.get(spec.name)
            if m is None:
                continue  # fixed parameter the template keeps at its own value
            v = float(x[self.schema.index(spec.name)])
            scale = float(m.get("scale", 1.0))
            out.append({"name": spec.name, "variable": m["variable"], "value": v * scale,
                        "unit": m.get("unit", aedt_unit(spec.unit)), "schema_value": v, "schema_unit": spec.unit})
        return out

    def generate_job_package(self, candidate: Mapping[str, Any], kind: str, iteration: int, candidate_id: str,
                             parent_design_id: str = "x0", extra_metadata: Optional[Dict[str, Any]] = None) -> Path:
        x = np.asarray(candidate["x"], float)
        values = self.schema.to_dict_values(x)
        job_id = sanitize_id(f"{self.study}_{candidate_id}")
        geometry, stl = None, None
        if self.generator is not None:
            geometry = {"template": self.schema.template, "length_unit": self.schema.length_unit,
                        "template_values": self.generator.template_values(values), "schema_values": values,
                        "note": "Description of the generated cell. HFSS uses the verified template + parameters.json, not this file."}
            if self.include_stl:
                from metasurface_optimizer.geometry.stl_exporter import export_stl
                tmp = Path(tempfile.mkdtemp()) / "geometry.stl"
                export_stl(self.generator.build(x), tmp, unit=self.schema.length_unit)
                stl = tmp
        pred = None
        if candidate.get("mu_J") is not None:
            pred = {"label": "PREDICTED (surrogate; not a simulation, never returned as a result)",
                    "mu_J": list(candidate["mu_J"]), "sigma_J": list(candidate["sigma_J"]), "mu": candidate.get("mu"),
                    "sigma": candidate.get("sigma"), "kind": candidate.get("kind"), "reason": candidate.get("reason")}
        meta = {"framework": "metasurface_optimizer", "framework_version": FRAMEWORK_VERSION, "runner_version": RUNNER_VERSION,
                "candidate_id": candidate_id, "job_kind": kind, "iteration": iteration, "parameters": values,
                "parameter_units": {p.name: p.unit for p in self.schema.parameters}, "schema": self.schema.name,
                "schema_fingerprint": self.schema.fingerprint(), "template_id": self.template.template_id,
                "template_sha256": self.template.sha256, "solver": "HFSS", "simulation_type": self.simulation_type,
                "simulation_config": self.simulation_config, "created": now_iso(), **(extra_metadata or {})}
        d = write_job_package(self.jobs_root, job_id=job_id, candidate_id=candidate_id, parent_design_id=parent_design_id, kind=kind,
                              simulation_type="HFSS" if self.simulation_type == "DRY_RUN" else self.simulation_type,
                              aedt_version=self.aedt_version, framework_version=FRAMEWORK_VERSION,
                              parameters=self.parameters_for(x), simulation_config=self.simulation_config,
                              template=self.template.worker_json(), geometry=geometry, stl_path=stl, metadata=meta,
                              prediction=pred, key=self.key)
        return d

    def create_job(self, candidate: Dict[str, Any], kind: str, iteration: int, candidate_id: str) -> SimulationJob:
        d = self.generate_job_package(candidate, "candidate" if kind in ("initial", "candidate") else kind, iteration, candidate_id)
        x = np.asarray(candidate["x"], float)
        pred = None
        if candidate.get("mu_J") is not None:
            pred = {k: candidate.get(k) for k in ("mu", "sigma", "mu_J", "sigma_J", "kind", "reason")}
        return SimulationJob(job_id=d.name, candidate_id=candidate_id, kind=kind, iteration=iteration, job_dir=str(d),
                             simulation_type=self.simulation_type, x=x.tolist(), values=self.schema.to_dict_values(x),
                             tag=candidate.get("tag", candidate_id), prediction=pred, created=now_iso())

    # -------------------------------------------------------------- results
    def parse_results(self, result_dir: Path, job: SimulationJob) -> SimulationResult:
        jd = Path(job.job_dir)
        r = parse_result_package(result_dir, read_json(jd / "job.json"), read_json(jd / "simulation_config.json"),
                                 read_json(jd / "template.json"))
        return r

    def validate_result(self, result_dir: Path, job: SimulationJob) -> List[str]:
        jd = Path(job.job_dir)
        return validate_result_package(Path(result_dir), read_json(jd / "job.json"), read_json(jd / "simulation_config.json"))

    # spec names
    parse_results_package = parse_results
    validate_results = validate_result
