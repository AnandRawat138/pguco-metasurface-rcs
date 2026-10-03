"""Two-PC HFSS automation: protocol, generated runner (against a FAKE PyAEDT), worker, transports, manager.

No test here runs HFSS. The fake PyAEDT (tests/fakes/fake_aedt.py) is a closed-form resonator that only
exercises control flow and file formats; the real-HFSS suite is tests/hfss/ (runs on the HFSS PC).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from hfss_worker.job_executor import execute_job
from hfss_worker.package import write_job_package
from hfss_worker.protocol import JobState, read_json, read_status, safe_join, sanitize_id, set_state, sha256_file, sign_manifest, verify_signature
from hfss_worker.runner import runner_core
from hfss_worker.runner.script import build_run_hfss_script, trusted_script_sha256
from hfss_worker.validator import validate_job_package
from hfss_worker.worker import claim_next, load_config, run_batch, run_queue, verify_template
from tests.fakes.fake_aedt import fake_project, fss_template_dict, make_module

ROOT = Path(__file__).resolve().parents[1]
FAKE_PKG = ROOT / "tests" / "fakes" / "fake_pyaedt"
SIM_CFG = {"frequency": {"start_hz": 10e9, "stop_hz": 20e9, "points": 21},
           "excitation": {"type": "floquet", "polarizations": ["TE", "TM"], "theta_deg": [0.0, 30.0], "phi_deg": [0.0]},
           "outputs": ["sparameters"], "checks": {"passivity_tol": 0.05}}
PARAMS = [{"name": "period", "variable": "p", "value": 10.0, "unit": "mm"},
          {"name": "length_a", "variable": "L", "value": 7.5, "unit": "mm"},
          {"name": "width_a", "variable": "W", "value": 1.2, "unit": "mm"}]


def _job(tmp_path, *, sim_type="HFSS", kind="candidate", params=None, cfg=None, proj_overrides=None, tmpl_overrides=None, key=None, job_id="job_0001"):
    proj = fake_project(tmp_path / "unit_cell.aedt", **(proj_overrides or {}))
    tmpl = fss_template_dict(proj, sha256_file(proj))
    tmpl.update(tmpl_overrides or {})
    return write_job_package(tmp_path / "jobs", job_id=job_id, candidate_id="c000_00", parent_design_id="x0", kind=kind, simulation_type=sim_type,
                             aedt_version="2024.2", framework_version="test", parameters=params or PARAMS, simulation_config=cfg or SIM_CFG,
                             template=tmpl, key=key)


# ------------------------------------------------------------------ protocol
def test_ids_and_paths():
    assert sanitize_id("iter 3/cand #1") == "iter_3_cand_1"
    for bad in ("../x", "/etc/passwd", "a/../../b", "C:\\x", "a\\..\\b"):
        with pytest.raises(ValueError):
            safe_join(Path("/tmp/root"), bad)
    assert safe_join(Path("/tmp/root"), "plots/a.png") == Path("/tmp/root/plots/a.png").resolve()


def test_state_transitions(tmp_path):
    set_state(tmp_path, JobState.CLAIMED)
    set_state(tmp_path, JobState.RUNNING)
    with pytest.raises(ValueError):
        set_state(tmp_path, JobState.PENDING)
    set_state(tmp_path, JobState.FAILED)
    set_state(tmp_path, JobState.CLAIMED)  # retry
    assert [h["state"] for h in read_status(tmp_path)["history"]] == ["CLAIMED", "RUNNING", "FAILED", "CLAIMED"]


def test_signature():
    job = {"job_id": "a", "files": {"x": "1"}}
    sig = sign_manifest(job, b"k")
    assert verify_signature(job, sig, b"k")
    assert not verify_signature({**job, "files": {"x": "2"}}, sig, b"k")


# ------------------------------------------------------------------ generated script
def test_script_is_self_contained_and_stable(tmp_path):
    s = build_run_hfss_script()
    compile(s, "run_hfss.py", "exec")
    assert s == build_run_hfss_script()
    assert "INLINE-SKIP" not in s and "from hfss_worker" not in s
    assert "import pyaedt" not in s.split("def import_pyaedt")[0]  # PyAEDT is imported lazily, at run time


def test_package_checks(tmp_path):
    jd = _job(tmp_path, sim_type="MOCK")
    assert validate_job_package(jd, {}, "MOCK") == []
    assert any("MOCK" in e for e in validate_job_package(jd, {}, "HFSS"))
    (jd / "run_hfss.py").write_text((jd / "run_hfss.py").read_text() + "\nimport os; os.system('echo pwned')\n")
    errs = validate_job_package(jd, {}, "MOCK")
    assert any("trusted runner" in e for e in errs) and any("modified" in e for e in errs)


def test_package_rejects_extra_files_bad_params_and_signature(tmp_path):
    jd = _job(tmp_path, sim_type="MOCK", params=[{"name": "x", "variable": "L;rm", "value": float("inf"), "unit": "parsec"}])
    (jd / "payload.exe").write_bytes(b"MZ")
    errs = " ".join(validate_job_package(jd, {}, "MOCK"))
    assert "payload.exe" in errs
    (jd / "payload.exe").unlink()
    errs = " ".join(validate_job_package(jd, {}, "MOCK"))
    assert "invalid AEDT variable" in errs and "not a finite number" in errs and "not allowed" in errs
    jd2 = _job(tmp_path / "s", sim_type="MOCK", key=b"secret")
    assert validate_job_package(jd2, {"job_key": "secret"}, "MOCK") == []
    assert any("signature" in e for e in validate_job_package(jd2, {"job_key": "other"}, "MOCK"))


def test_hfss_candidate_jobs_need_a_verified_template(tmp_path):
    jd = _job(tmp_path)
    errs = validate_job_package(jd, {"validation_record": str(tmp_path / "none.json")}, "HFSS")
    assert errs and "not operational" in errs[0]


# ------------------------------------------------------------------ runner vs fake PyAEDT
@pytest.mark.parametrize("legacy", [False, True])
def test_runner_completes_with_both_api_generations(tmp_path, legacy):
    jd = _job(tmp_path)
    state, err = runner_core.run(str(jd), str(tmp_path / "out"), pyaedt_module=make_module(legacy))
    assert state == "COMPLETED", err
    rows = read_json(tmp_path / "out" / "validation.json")["rows"]
    assert [r["id"] for r in rows] == [f"{i:02d}" for i in range(1, 13)]
    assert all(r["status"] == "PASS" for r in rows), rows
    with open(tmp_path / "out" / "sparameters.csv") as f:
        data = list(csv.DictReader(f))
    assert len(data) == 2 * 21  # two incidence angles x 21 points
    assert {float(r["theta_inc"]) for r in data} == {0.0, 30.0}
    r0 = data[0]
    assert abs(float(r0["S11_TE_mag"]) ** 2 + float(r0["S21_TE_mag"]) ** 2 - 1) < 1e-9
    res = read_json(tmp_path / "out" / "result.json")
    assert res["fidelity"] == "VALIDATED_SIMULATION" and res["files"]["sparameters.csv"] == sha256_file(tmp_path / "out" / "sparameters.csv")
    meta = read_json(tmp_path / "out" / "solver_metadata.json")
    assert meta["versions"]["aedt_version"] == "2024.2" and len(meta["solves"]) == 2
    assert {v["variable"] for v in meta["variables"]} == {"p", "L", "W"}
    assert not (tmp_path / "out_work").exists()


@pytest.mark.parametrize("override,row,text", [
    ({"proj_overrides": {"variables": {"p": "10mm", "W": "1mm", "scan_theta": "0deg", "scan_phi": "0deg"}}}, "04", "missing"),
    ({"proj_overrides": {"invalid_geometry": True}}, "05", "validation failed"),
    ({"proj_overrides": {"materials": ["vacuum"]}}, "06", "missing"),
    ({"proj_overrides": {"excitations": ["FloquetPort1"]}}, "07", "missing"),
    ({"proj_overrides": {"boundaries": ["Master1"]}}, "08", "missing"),
    ({"proj_overrides": {"sweep_ignores_update": True}}, "09", "reads back"),
    ({"proj_overrides": {"nonpassive": True}}, "11", "non-passive"),
    ({"proj_overrides": {"short_output": True}}, "11", "values for"),
    ({"tmpl_overrides": {"excitation": {"names": ["FloquetPort1"], "polarization": {"mode": "floquet_modes"}}}}, "07", "angle"),
    ({"proj_overrides": {"aedt_version": "2023.2"}}, "02", "targets"),
])
def test_runner_fails_validation_instead_of_simulating_wrongly(tmp_path, override, row, text):
    jd = _job(tmp_path, **override)
    state, err = runner_core.run(str(jd), str(tmp_path / "out"), pyaedt_module=make_module(False))
    assert state == "FAILED_VALIDATION", err
    rows = {r["id"]: r for r in read_json(tmp_path / "out" / "validation.json")["rows"]}
    assert rows[row]["status"] == "FAIL" and text in rows[row]["detail"], rows[row]
    assert read_json(tmp_path / "out" / "result.json")["fidelity"] is None
    assert not (tmp_path / "out" / "sparameters.csv").exists()


def test_runner_detects_changed_template_and_failed_solve(tmp_path):
    jd = _job(tmp_path)
    Path(read_json(jd / "template.json")["aedt_project"]).write_text("{}")
    state, err = runner_core.run(str(jd), str(tmp_path / "o1"), pyaedt_module=make_module(False))
    assert state == "FAILED_VALIDATION" and "changed since registration" in err
    jd2 = _job(tmp_path / "b", proj_overrides={"solve_fails": True})
    state, err = runner_core.run(str(jd2), str(tmp_path / "o2"), pyaedt_module=make_module(False))
    assert state == "FAILED" and "solve failed" in err


# ------------------------------------------------------------------ worker (subprocess, fake PyAEDT on the path)
@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(FAKE_PKG), str(ROOT), os.environ.get("PYTHONPATH", "")]))


def test_worker_hfss_mode_gate_verification_and_run(tmp_path, fake_env):
    cfg = {"python_executable": sys.executable, "validation_record": str(tmp_path / "state" / "validation_suite.json"),
           "max_retries": 0, "progress_poll_s": 0.05, "plots": False}
    jd = _job(tmp_path)
    r = execute_job(jd, cfg, "HFSS")
    assert r["state"] == "FAILED_VALIDATION" and "not operational" in r["errors"][0]
    # verification on the HFSS PC: the 12-step suite with the baseline parameters
    tmpl = read_json(jd / "template.json")
    tmpl.update(baseline_parameters=PARAMS, verification_config=SIM_CFG)
    (tmp_path / "tmpl.json").write_text(json.dumps(tmpl))
    out = verify_template(tmp_path / "tmpl.json", cfg)
    assert out["result"]["state"] == "COMPLETED", out
    assert all(row["status"] == "PASS" for row in out["rows"])
    rec = read_json(tmp_path / "state" / "validation_suite.json")["templates"]["fake_fss"]
    assert rec["passed"] and rec["sha256"] == tmpl["sha256"]
    jd2 = _job(tmp_path / "second", job_id="job_0002")
    shutil.copy2(tmp_path / "unit_cell.aedt", tmp_path / "second" / "unit_cell.aedt")
    r2 = execute_job(jd2, cfg, "HFSS")
    assert r2["state"] == "COMPLETED", r2
    hist = [h["state"] for h in read_status(jd2)["history"]]
    assert hist[:2] == ["CLAIMED", "RUNNING"] and hist[-1] == "COMPLETED"


def test_worker_timeout_kills_runner(tmp_path, fake_env):
    jd = _job(tmp_path, proj_overrides={})
    cfg = {"python_executable": sys.executable, "max_runtime_minutes": 1e-9, "max_retries": 1, "progress_poll_s": 0.01,
           "require_validation_suite": False, "plots": False}
    r = execute_job(jd, cfg, "HFSS")
    assert r["state"] == "TIMEOUT" and r["attempts"] == 2
    assert (jd / "result_attempt_1" / "result.json").is_file() and (jd / "result_attempt_2" / "result.json").is_file()


# ------------------------------------------------------------------ MOCK worker: retries, invalid output, batches, queue
def _mock_job(tmp_path, job_id, mock=None):
    cfg = dict(SIM_CFG, excitation=dict(SIM_CFG["excitation"], theta_deg=[0.0]), mock=mock or {})
    params = [{"name": n, "variable": n, "value": v, "unit": "mm"} for n, v in
              {"period": 10.0, "length_a": 8.5, "width_a": 8.5, "length_b": 5.0, "width_b": 5.0, "substrate_thickness": 3.0}.items()]
    params.append({"name": "eps_r", "variable": "eps_r", "value": 4.4, "unit": ""})
    return _job(tmp_path, sim_type="MOCK", params=params, cfg=cfg, job_id=job_id)


def test_mock_retries_keep_every_attempt(tmp_path):
    jd = _mock_job(tmp_path, "j1", {"fail_attempts": 1})
    r = execute_job(jd, {"max_retries": 2, "plots": False}, "MOCK")
    assert r["state"] == "COMPLETED" and r["attempts"] == 2
    assert read_json(jd / "result_attempt_1" / "result.json")["status"] == "FAILED"
    assert read_json(jd / "result_attempt_2" / "result.json")["fidelity"] == "MOCK_SOLVER"
    jd2 = _mock_job(tmp_path / "b", "j2", {"fail_attempts": 9})
    r2 = execute_job(jd2, {"max_retries": 2, "plots": False}, "MOCK")
    assert r2["state"] == "FAILED" and r2["attempts"] == 3
    jd3 = _mock_job(tmp_path / "c", "j3", {"invalid_output": True})
    r3 = execute_job(jd3, {"max_retries": 2, "plots": False}, "MOCK")
    assert r3["state"] == "FAILED_VALIDATION" and r3["attempts"] == 1


def test_manual_batch_and_shared_queue(tmp_path):
    batch = tmp_path / "batch_001"
    for i in range(3):
        jd = _mock_job(tmp_path / f"src{i}", f"job_{i}", {"fail_attempts": 9} if i == 2 else {})
        shutil.copytree(jd, batch / "jobs" / jd.name)
    out = run_batch(batch, {"max_retries": 0, "plots": False}, "MOCK")
    summ = read_json(out / "batch_results.json")
    assert [j["state"] for j in summ["jobs"]] == ["COMPLETED", "COMPLETED", "FAILED"]
    assert (out / "job_0" / "sparameters.csv").is_file() and (out / "job_2" / "job_status.json").is_file()
    q = tmp_path / "queue"
    for d in ("pending", "running", "completed", "failed", "archive"):
        (q / d).mkdir(parents=True)
    for i in range(2):
        shutil.copytree(batch / "jobs" / f"job_{i}", q / "pending" / f"job_{i}")
        shutil.rmtree(q / "pending" / f"job_{i}" / "result_attempt_1")
        (q / "pending" / f"job_{i}" / "job_status.json").unlink()
    a = claim_next(q)
    assert a is not None and a.parent.name == "running"
    os.rename(a, q / "pending" / a.name)
    n = run_queue(q, {"max_retries": 0, "plots": False}, "MOCK", once=True)
    assert n == 2 and sorted(p.name for p in (q / "completed").iterdir()) == ["job_0", "job_1"]
    assert read_json(q / "workers" / f"{load_config(None).get('worker_id') or __import__('platform').node()}.json")["state"] == "stopped"


def test_claim_is_atomic(tmp_path):
    q = tmp_path / "q"
    (q / "pending" / "job_a").mkdir(parents=True)
    (q / "running").mkdir()
    first, second = claim_next(q), claim_next(q)
    assert first is not None and second is None
