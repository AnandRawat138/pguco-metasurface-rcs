"""Writes job packages (standard library only, so both PCs can build them).

Layout of ``<root>/<job_id>/``::

    job.json                 identity, versions, kind, simulation type, manifest of file hashes
    job.sig                  HMAC-SHA256 of job.json (only when a shared key is configured)
    parameters.json          [{name, variable, value, unit}] - explicit units, e.g. 10 mm
    simulation_config.json   solver-independent: frequency, excitation, outputs, checks
    template.json            the verified template registration the job must use
    run_hfss.py              the trusted, self-contained PyAEDT runner (identical for every job)
    geometry/geometry.json   generated geometry description (+ geometry.stl when requested)
    metadata.json            reproducibility record (framework, schema, template hash, ...)
    prediction.json          the surrogate's PREDICTED response for this candidate (never sent back as a result)
    README.txt
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hfss_worker.protocol import (PROTOCOL_VERSION, check_id, canonical, now_iso, sha256_file, sign_manifest,
                                  write_json_atomic)
from hfss_worker.runner.script import build_run_hfss_script

README = """Job {job_id}  ({kind}, {simulation_type})
Candidate: {candidate_id}   parent design: {parent_design_id}
Template:  {template_id}  ({aedt_project})
Created:   {created_at} by metasurface_optimizer {framework_version}

What this is
  A request to simulate ONE design in HFSS {aedt_version}. Parameters are in parameters.json
  (explicit units); frequency, excitation and outputs in simulation_config.json.

How it runs
  Normally the worker runs it:   python hfss_worker/worker.py --input <batch folder>
  By hand (AEDT Python):         python run_hfss.py --out result_manual

Do not edit any file: the worker checks every file's hash{signed}.
Results are VALIDATED SIMULATION only if the run ends COMPLETED; MOCK jobs are software tests, not physics.
"""


def write_job_package(root: Path, *, job_id: str, candidate_id: str, parent_design_id: str, kind: str,
                      simulation_type: str, aedt_version: str, framework_version: str,
                      parameters: List[Mapping[str, Any]], simulation_config: Mapping[str, Any],
                      template: Mapping[str, Any], geometry: Optional[Mapping[str, Any]] = None,
                      stl_path: Optional[Path] = None, metadata: Optional[Mapping[str, Any]] = None,
                      prediction: Optional[Mapping[str, Any]] = None, key: Optional[bytes] = None,
                      solver: str = "HFSS") -> Path:
    check_id(job_id)
    check_id(candidate_id)
    d = Path(root) / job_id
    if d.exists():
        raise FileExistsError(f"job {job_id} already exists at {d}; job folders are never overwritten")
    (d / "geometry").mkdir(parents=True)
    write_json_atomic(d / "parameters.json", {"parameters": [dict(p) for p in parameters]})
    write_json_atomic(d / "simulation_config.json", dict(simulation_config))
    write_json_atomic(d / "template.json", dict(template))
    (d / "run_hfss.py").write_text(build_run_hfss_script(), encoding="utf-8")
    if geometry is not None:
        write_json_atomic(d / "geometry" / "geometry.json", dict(geometry))
    if stl_path is not None:
        shutil.copy2(stl_path, d / "geometry" / "geometry.stl")
        side = Path(stl_path).with_suffix(".materials.json")
        if side.is_file():
            shutil.copy2(side, d / "geometry" / "geometry.materials.json")
    if metadata is not None:
        write_json_atomic(d / "metadata.json", dict(metadata))
    if prediction is not None:
        write_json_atomic(d / "prediction.json", dict(prediction))
    created = now_iso()
    info = dict(job_id=job_id, kind=kind, simulation_type=simulation_type, candidate_id=candidate_id, parent_design_id=parent_design_id,
                template_id=template.get("template_id"), aedt_project=template.get("aedt_project"), created_at=created,
                framework_version=framework_version, aedt_version=aedt_version, signed=" and its signature" if key else "")
    (d / "README.txt").write_text(README.format(**info), encoding="utf-8")
    files: Dict[str, str] = {}
    for p in sorted(d.rglob("*")):
        if p.is_file():
            files[p.relative_to(d).as_posix()] = sha256_file(p)
    job = {"job_id": job_id, "framework_version": framework_version, "created_at": created, "solver": solver,
           "aedt_version": aedt_version, "status": "PENDING", "candidate_id": candidate_id, "parent_design_id": parent_design_id,
           "kind": kind, "simulation_type": simulation_type, "template_id": template.get("template_id"),
           "template_sha256": template.get("sha256"), "protocol_version": PROTOCOL_VERSION, "files": files}
    write_json_atomic(d / "job.json", job)
    if key:
        (d / "job.sig").write_text(sign_manifest(job, key), encoding="utf-8")
    return d


def manifest_digest(job: Mapping[str, Any]) -> str:
    import hashlib
    return hashlib.sha256(canonical(job)).hexdigest()
