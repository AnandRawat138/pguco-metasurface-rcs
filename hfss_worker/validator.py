"""Checks a job package before it may run, and a result package before it may be returned.

A job runs only if: its files are on the allow-list and match the manifest
hashes; its ``run_hfss.py`` is byte-identical to the worker's own trusted copy
(so no job can ship arbitrary code); its HMAC signature verifies when a key is
configured; its parameters are plain numbers with known units; and, for
candidate jobs on a real HFSS worker, its template has passed the 12-step
validation suite on this machine with the same file hash and AEDT version.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, List, Mapping, Optional

from hfss_worker.protocol import (PROTOCOL_VERSION, check_id, load_key, read_json, safe_join, sha256_file,
                                  unexpected_job_files, unexpected_result_files, verify_signature)
from hfss_worker.runner.script import trusted_script_sha256

VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
ALLOWED_UNITS = {"mm", "um", "nm", "cm", "m", "mil", "in", "deg", "rad", "GHz", "MHz", "Hz", ""}
REQUIRED_JOB_FIELDS = ("job_id", "framework_version", "created_at", "solver", "aedt_version", "status", "candidate_id",
                       "parent_design_id", "simulation_type", "kind", "files", "protocol_version")
OPTIONAL_SKIP_ROWS = {"05", "06", "08"}  # may be SKIP (feature unavailable / nothing registered); all others must PASS


def _num_ok(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_job_package(job_dir: Path, cfg: Mapping[str, Any], mode: str) -> List[str]:
    job_dir = Path(job_dir)
    errs: List[str] = []
    try:
        job = read_json(job_dir / "job.json")
    except Exception as e:
        return [f"job.json unreadable: {e}"]
    for k in REQUIRED_JOB_FIELDS:
        if k not in job:
            errs.append(f"job.json missing field {k!r}")
    if errs:
        return errs
    try:
        check_id(job["job_id"])
        check_id(job["candidate_id"])
    except ValueError as e:
        errs.append(str(e))
    if job["job_id"] != job_dir.name:
        errs.append(f"folder name {job_dir.name!r} does not match job_id {job['job_id']!r}")
    if str(job["protocol_version"]).split(".")[0] != PROTOCOL_VERSION.split(".")[0]:
        errs.append(f"protocol {job['protocol_version']} is not compatible with worker protocol {PROTOCOL_VERSION}")
    if job["simulation_type"] != mode.upper():
        errs.append(f"job is a {job['simulation_type']} job but this worker runs in {mode.upper()} mode")
    if job["kind"] not in ("candidate", "verify_template", "baseline"):
        errs.append(f"unknown job kind {job['kind']!r}")
    bad = unexpected_job_files(job_dir)
    if bad:
        errs.append("files not allowed in a job package: " + ", ".join(bad[:10]))
    files = job.get("files", {})
    for rel, h in files.items():
        try:
            p = safe_join(job_dir, rel)
        except ValueError as e:
            errs.append(str(e))
            continue
        if not p.is_file():
            errs.append(f"manifest file missing: {rel}")
        elif sha256_file(p) != h:
            errs.append(f"{rel} was modified after the job was generated (hash mismatch)")
    for need in ("parameters.json", "simulation_config.json", "template.json", "run_hfss.py"):
        if need not in files:
            errs.append(f"{need} is not in the job manifest")
    if (job_dir / "run_hfss.py").is_file() and sha256_file(job_dir / "run_hfss.py") != trusted_script_sha256():
        errs.append("run_hfss.py differs from this worker's trusted runner (different framework version or edited); refusing to run it")
    key = load_key(cfg.get("job_key"), cfg.get("job_key_file"))
    sig = job_dir / "job.sig"
    if key is not None:
        if not sig.is_file():
            errs.append("job is not signed but this worker has a job key configured")
        elif not verify_signature(job, sig.read_text(), key):
            errs.append("job signature does not verify (wrong key or edited job.json)")
    elif cfg.get("require_signature"):
        errs.append("require_signature is set but no job key is configured on this worker")
    if errs:
        return errs
    errs += _check_parameters(read_json(job_dir / "parameters.json"))
    errs += _check_sim_config(read_json(job_dir / "simulation_config.json"))
    tmpl = read_json(job_dir / "template.json")
    errs += _check_template(tmpl, cfg, mode)
    if not errs and mode.upper() == "HFSS" and job["kind"] != "verify_template" and cfg.get("require_validation_suite", True):
        errs += check_template_verified(tmpl, job, cfg)
    return errs


def _check_parameters(p: Mapping[str, Any]) -> List[str]:
    errs = []
    plist = p.get("parameters")
    if not isinstance(plist, list) or not plist:
        return ["parameters.json has no parameter list"]
    seen = set()
    for q in plist:
        v = q.get("variable", "")
        if not VAR_RE.match(str(v)):
            errs.append(f"invalid AEDT variable name {v!r}")
        if v in seen:
            errs.append(f"variable {v!r} set twice")
        seen.add(v)
        if not _num_ok(q.get("value")):
            errs.append(f"{v}: value {q.get('value')!r} is not a finite number")
        if q.get("unit", "") not in ALLOWED_UNITS:
            errs.append(f"{v}: unit {q.get('unit')!r} not allowed")
    return errs


def _check_sim_config(c: Mapping[str, Any]) -> List[str]:
    errs = []
    fr = c.get("frequency", {})
    if not all(_num_ok(fr.get(k)) for k in ("start_hz", "stop_hz", "points")):
        return ["simulation_config.frequency needs numeric start_hz, stop_hz, points"]
    if not 0 < fr["start_hz"] < fr["stop_hz"]:
        errs.append("frequency start must be > 0 and < stop")
    if not 1 <= int(fr["points"]) <= 20001:
        errs.append("frequency points must be 1..20001")
    ex = c.get("excitation", {})
    if ex.get("type") not in ("floquet", "plane_wave"):
        errs.append(f"excitation.type {ex.get('type')!r} must be floquet or plane_wave")
    for p in ex.get("polarizations", []):
        if str(p).upper() not in ("TE", "TM"):
            errs.append(f"polarization {p!r} must be TE or TM")
    for k in ("theta_deg", "phi_deg"):
        for a in ex.get(k, []):
            if not _num_ok(a) or not -360 <= a <= 360:
                errs.append(f"{k} value {a!r} invalid")
    for o in c.get("outputs", []):
        if o not in ("sparameters", "rcs", "farfield"):
            errs.append(f"unknown output {o!r}")
    return errs


def _check_template(t: Mapping[str, Any], cfg: Mapping[str, Any], mode: str) -> List[str]:
    errs = []
    for k in ("template_id", "aedt_project", "design", "setup", "sweep"):
        if not t.get(k):
            errs.append(f"template.json missing {k!r}")
    if errs:
        return errs
    try:
        check_id(t["template_id"])
    except ValueError as e:
        errs.append(str(e))
    if mode.upper() != "HFSS":
        return errs
    proj = str(t["aedt_project"])
    if not proj.lower().endswith((".aedt", ".aedtz")):
        errs.append("template project must be a .aedt/.aedtz file")
    roots = [Path(r) for r in cfg.get("allowed_template_roots") or []]
    if roots:
        try:
            rp = Path(proj).resolve()
            if not any(rp == r.resolve() or r.resolve() in rp.parents for r in roots):
                errs.append(f"template {proj} is outside allowed_template_roots")
        except OSError as e:
            errs.append(f"template path invalid: {e}")
    for name in ("design", "setup", "sweep"):
        if any(ch in str(t[name]) for ch in "\"'\n\r;"):
            errs.append(f"template {name} contains forbidden characters")
    return errs


def check_template_verified(tmpl: Mapping[str, Any], job: Mapping[str, Any], cfg: Mapping[str, Any]) -> List[str]:
    rec_path = Path(cfg.get("validation_record", "worker_state/validation_suite.json"))
    if not rec_path.is_file():
        return [f"worker not operational: no validation-suite record at {rec_path}; run a verify_template job (or tests/hfss) first"]
    rec = read_json(rec_path).get("templates", {}).get(tmpl["template_id"])
    if not rec:
        return [f"template {tmpl['template_id']} has not passed the validation suite on this worker"]
    errs = []
    if tmpl.get("sha256") and rec.get("sha256") != tmpl["sha256"]:
        errs.append("template file changed since it passed the validation suite; re-verify it")
    if rec.get("aedt_version") and job.get("aedt_version") and rec["aedt_version"] != job["aedt_version"]:
        errs.append(f"template verified with AEDT {rec['aedt_version']}, job targets {job['aedt_version']}")
    if not rec.get("passed"):
        errs.append("template validation suite did not pass")
    return errs


def suite_passed(rows: List[Mapping[str, Any]]) -> bool:
    if len(rows) != 12:
        return False
    return all(r["status"] == "PASS" or (r["status"] == "SKIP" and r["id"] in OPTIONAL_SKIP_ROWS) for r in rows)


def validate_result_package(result_dir: Path, job: Mapping[str, Any], sim_cfg: Optional[Mapping[str, Any]] = None) -> List[str]:
    result_dir = Path(result_dir)
    errs: List[str] = []
    rp = result_dir / "result.json"
    if not rp.is_file():
        return ["result.json missing"]
    try:
        res = read_json(rp)
    except Exception as e:
        return [f"result.json unreadable: {e}"]
    if res.get("job_id") != job.get("job_id"):
        errs.append(f"result job_id {res.get('job_id')!r} != {job.get('job_id')!r}")
    if res.get("candidate_id") != job.get("candidate_id"):
        errs.append("result candidate_id does not match the job")
    if res.get("simulation_type") != job.get("simulation_type"):
        errs.append(f"result simulation_type {res.get('simulation_type')!r} != job {job.get('simulation_type')!r}")
    bad = unexpected_result_files(result_dir)
    if bad:
        errs.append("unexpected files in result package: " + ", ".join(bad[:10]))
    for rel, h in (res.get("files") or {}).items():
        try:
            p = safe_join(result_dir, rel)
        except ValueError as e:
            errs.append(str(e))
            continue
        if not p.is_file() or sha256_file(p) != h:
            errs.append(f"{rel} missing or modified")
    if res.get("status") == "COMPLETED":
        outs = (sim_cfg or {}).get("outputs", [])
        need = {"sparameters": "sparameters.csv", "rcs": "rcs.csv", "farfield": "farfield.csv"}
        for o in outs:
            f = result_dir / need[o]
            if not f.is_file():
                errs.append(f"{need[o]} missing for requested output {o}")
                continue
            errs += _check_csv(f, o)
    return errs


def _check_csv(path: Path, kind: str) -> List[str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [f"{path.name} is empty"]
    cols = set(rows[0].keys())
    req = {"sparameters": {"frequency", "theta_inc", "phi_inc"}, "rcs": {"pol", "theta_inc", "frequency", "rcs_m2"},
           "farfield": {"frequency", "theta", "phi", "value"}}[kind]
    if not req <= cols:
        return [f"{path.name} lacks columns {sorted(req - cols)}"]
    errs = []
    for r in rows[:100000]:
        for k, v in r.items():
            if k == "pol" or k == "quantity":
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                errs.append(f"{path.name}: non-numeric {k}={v!r}")
                break
            if math.isnan(x):
                errs.append(f"{path.name}: NaN in {k}")
                break
        if len(errs) > 5:
            break
    return errs
