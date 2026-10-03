"""Optimizer <-> simulation loop: MOCK end-to-end (5 candidates), budget, rejection, fidelity guards,
and an HFSS-mode run through the shared-folder queue against the FAKE PyAEDT (control flow only)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from hfss_worker.protocol import read_json, read_status
from hfss_worker.worker import run_queue, verify_template
from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.experiment import ExperimentBuilder, load_config
from metasurface_optimizer.optimization.active_learning import CounterfactualOptimizer
from metasurface_optimizer.simulation.exports import export_final_outputs
from metasurface_optimizer.simulation.hfss_adapter import HFSSAdapter, make_simulation_config
from metasurface_optimizer.simulation.job_manager import BaselineNotReproduced, SimulationBudget, SimulationManager
from metasurface_optimizer.simulation.templates import TemplateRegistry, TemplateSpec
from metasurface_optimizer.simulation.transports import LocalMockTransport, ManualBatchTransport, SharedFolderTransport
from tests.fakes.fake_aedt import fake_project

ROOT = Path(__file__).resolve().parents[1]
FAKE_PKG = ROOT / "tests" / "fakes" / "fake_pyaedt"
MOCK_ORACLE = {"oracle_conditions": {"theta_s_step": 9.0, "phi_s_step": 45.0, "tile_cells": 4, "n_tiles": 6}}


def _demo(tmp_path, n_init=4, budget=10, batch=5, mock_knobs=None, transport="mock"):
    cfg = load_config(ROOT / "configs/experiments/demo_checkerboard.yaml")
    cfg["optimizer"]["n_init_lhs"] = n_init
    b = ExperimentBuilder(cfg, ROOT)
    opt = CounterfactualOptimizer(b.problem(), b.pguco_config(10), b.prior_feature_fn())
    t = TemplateSpec.from_dict(json.loads((ROOT / "configs/hfss/example_finite_rcs_template.json").read_text()))
    sc = make_simulation_config(t, 6e9, 14e9, 17, ("TE", "TM"), (0.0, 20.0, 40.0), outputs=["sparameters", "rcs"],
                                mock={**MOCK_ORACLE, **(mock_knobs or {})})
    tr = LocalMockTransport(tmp_path / "transfer", {"max_retries": 2, "plots": False}) if transport == "mock" else ManualBatchTransport(tmp_path / "transfer")
    ad = HFSSAdapter(t, b.schema, sc, tmp_path / "jobs", tr, "MOCK", generator=b.generator, study="demo")
    return b, opt, SimulationManager(opt, ad, tmp_path / "study", SimulationBudget(max_simulations=budget, batch_size=batch))


def test_mock_end_to_end_five_candidates(tmp_path):
    b, opt, m = _demo(tmp_path, n_init=4, budget=10, batch=5)
    s = m.run()
    # 1 baseline + 4 initial + 5 PGUCO candidates, every one through package -> worker -> result -> import -> update
    assert [bt["phase"] for bt in m.batches] == ["baseline", "initial", "iterate"]
    assert len(m.batches[2]["job_ids"]) == 5
    assert s["budget"]["used"] == 10 and s["budget"]["remaining"] == 0 and s["budget"]["completed"] == 10
    assert opt.n_hf == 10 and all(r.fidelity is Fidelity.MOCK for r in opt.hf_records())
    assert len(m.feedback) == 5 and s["feedback_summary"]["n"] == 5
    for f in m.feedback:
        assert len(f["abs_error"]) == len(opt.problem.objectives.names) and f["response_rmse_db"] is not None
    c = m.batches[2]["job_ids"][0]
    cdir = m.candidate_dir(m.jobs[c].candidate_id)
    assert read_json(cdir / "predicted" / "prediction.json")["label"].startswith("PREDICTED")
    assert (cdir / "predicted" / "predicted_response.csv").is_file()
    assert read_json(cdir / "validated" / c / "result.json")["simulation_type"] == "MOCK"
    with pytest.raises(RuntimeError):
        m.generate_batch()  # budget is never exceeded
    out = export_final_outputs(b, m, tmp_path / "final")
    assert out["missing"] == []
    assert "hfss_validated_response.csv" not in out["written"] and "MOCK_response_NOT_PHYSICS_response.csv" in out["written"]
    assert (tmp_path / "final" / "README_MOCK.txt").is_file() and (tmp_path / "final" / "optimization_history.png").is_file()


def test_failures_are_excluded_and_retried_by_the_worker(tmp_path):
    b, opt, m = _demo(tmp_path, n_init=3, budget=4, batch=4, mock_knobs={"fail_attempts": 1})
    m.run()
    jobs = list(m.jobs.values())
    assert all(j.state == "COMPLETED" and j.attempts == 2 for j in jobs)  # each succeeded on the worker's retry
    assert m.budget_report()["used"] == 4  # retries of the same job are not charged again
    b2, opt2, m2 = _demo(tmp_path / "x", n_init=3, budget=4, batch=4, mock_knobs={"fail_attempts": 9})
    with pytest.raises(RuntimeError, match="baseline simulation failed"):
        m2.run()


def test_invalid_candidates_never_reach_the_solver(tmp_path):
    b, opt, m = _demo(tmp_path, budget=3)
    x_bad = opt.problem.x0.copy()
    x_bad[b.schema.index("length_a")] = 11.9  # patch larger than the period -> overlap
    jobs = m.create_jobs([{"x": x_bad, "values": b.schema.to_dict_values(x_bad), "kind": "baseline", "tag": "bad"}], "baseline")
    assert jobs == [] and m.rejected and m.rejected[0]["checks"]["geometry"] == "FAIL"
    assert m.used == 0
    xs = [opt.problem.x0.copy() for _ in range(5)]
    jobs = m.create_jobs([{"x": x, "values": b.schema.to_dict_values(x), "kind": "initial"} for x in xs], "initial")
    assert len(jobs) == 3 and any("budget" in r["errors"][0] for r in m.rejected)


def test_manual_transfer_round_trip(tmp_path):
    from hfss_worker.worker import run_batch
    b, opt, m = _demo(tmp_path, budget=1, transport="manual")
    bt = m.generate_batch()
    assert m.collect_results() == []  # nothing back yet
    bdir = Path(bt["transport"]["batch_dir"])
    assert (tmp_path / "transfer" / "batch_001.zip").is_file()
    # ... carried to the HFSS PC, run there (MOCK here), results carried back to some other folder
    out = run_batch(bdir, {"max_retries": 0, "plots": False}, "MOCK", output=tmp_path / "usb" / "batch_001_results")
    res = m.import_batch(out)
    assert len(res) == 1 and res[0].ok and opt.phase == "initial"
    with pytest.raises(FileExistsError):
        m.adapter.transport.import_folder(out, "batch_001")  # an import is never overwritten


def test_mock_cannot_mix_with_other_fidelities(tmp_path):
    b, opt, m = _demo(tmp_path, budget=1)
    real = b.solver().evaluate(b.schema.to_dict_values(opt.problem.x0))  # synthetic oracle, a different fidelity
    opt.set_baseline(real)
    with pytest.raises(ValueError, match="MOCK results cannot be mixed"):
        m.run()
    assert all(r.fidelity is not Fidelity.MOCK for r in opt.dataset.records)


# ------------------------------------------------------------------ HFSS mode, shared folder, FAKE PyAEDT
@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(FAKE_PKG), str(ROOT), os.environ.get("PYTHONPATH", "")]))


def _fss_setup(tmp_path):
    cfg = load_config(ROOT / "configs/experiments/demo_checkerboard.yaml")
    cfg["schema"] = "configs/schemas/fss_patch_example.yaml"
    cfg["objectives"] = [{"name": "stopband", "kind": "sparam_level", "sparam": [2, 1], "band": [14.0e9, 16.0e9], "weight": 1.0},
                         {"name": "match", "kind": "sparam_level", "sparam": [1, 1], "band": [10.0e9, 12.0e9], "weight": 0.5}]
    cfg["normalization"] = "minmax"
    cfg["optimizer"]["n_init_lhs"] = 4
    cfg["physics"]["reciprocal"] = True
    b = ExperimentBuilder(cfg, ROOT)
    opt = CounterfactualOptimizer(b.problem(), b.pguco_config(3), b.prior_feature_fn())
    proj = fake_project(tmp_path / "pc2" / "fss_unit_cell.aedt")
    spec = {"template_id": "fake_fss", "name": "FAKE unit cell (tests only)", "simulation_type": "periodic_fss", "aedt_project": str(proj),
            "design": "HFSSDesign1", "solution_type": "Modal", "setup": "Setup1", "sweep": "Sweep1",
            "parameter_mapping": {"period": {"variable": "p", "unit": "mm"}, "patch_length": {"variable": "L", "unit": "mm"},
                                  "patch_width": {"variable": "W", "unit": "mm"}},
            "excitation": {"names": ["FloquetPort1", "FloquetPort2"], "angle_variables": {"theta": "scan_theta"}, "polarization": {"mode": "floquet_modes"}},
            "boundaries": ["Master1", "Slave1"], "materials": ["Rogers RO4003 (tm)"],
            "outputs": {"sparameters": {"S11_TE": "S(FloquetPort1:1,FloquetPort1:1)", "S21_TE": "S(FloquetPort2:1,FloquetPort1:1)"}}}
    reg = TemplateRegistry(tmp_path / "templates")
    t = reg.register(TemplateSpec.from_dict(spec), b.schema)
    sc = make_simulation_config(t, 10e9, 20e9, 41, ("TE",), (0.0,), outputs=["sparameters"])
    return b, opt, reg, t, sc


def test_hfss_mode_through_shared_folder_with_fake_pyaedt(tmp_path, fake_env):
    b, opt, reg, t, sc = _fss_setup(tmp_path)
    wcfg = {"python_executable": sys.executable, "validation_record": str(tmp_path / "pc2" / "state" / "validation_suite.json"),
            "max_retries": 0, "progress_poll_s": 0.02, "plots": False}
    q = SharedFolderTransport(tmp_path / "simulation_queue")
    ad = HFSSAdapter(t, b.schema, sc, tmp_path / "jobs", q, "HFSS", study="fss")
    m = SimulationManager(opt, ad, tmp_path / "study", SimulationBudget(max_simulations=8, batch_size=3))
    with pytest.raises(RuntimeError, match="no valid candidate"):
        m.generate_batch()  # unverified template: nothing may be simulated
    # verify on "PC-2" and import the verification into the registry on "PC-1"
    tj = tmp_path / "tmpl.json"
    tj.write_text(json.dumps({**t.worker_json(), "baseline_parameters": ad.parameters_for(opt.problem.x0), "verification_config": sc}))
    v = verify_template(tj, wcfg)
    assert v["result"]["state"] == "COMPLETED"
    t = reg.record_verification("fake_fss", Path(v["job_dir"]) / read_status(Path(v["job_dir"]))["final_result_dir"])
    assert t.verified and t.status() == "VERIFIED"
    ad.template = t
    m.rejected.clear()
    while m.stop_reason is None:
        if m.open_batch is None:
            try:
                m.generate_batch()
            except RuntimeError:
                break
        run_queue(q.root, wcfg, "HFSS", once=True)
        m.collect_results()
    assert m.used == 8 and all(j.state == "COMPLETED" for j in m.jobs.values())
    assert all(r.fidelity is Fidelity.SIMULATED_HF for r in opt.hf_records())
    assert q.counts()["completed"] == 8 and q.counts()["pending"] == 0
    rec = opt.hf_records()[-1]
    assert rec.response.provenance.extra["template_id"] == "fake_fss" and rec.response.provenance.extra["versions"]["aedt_version"] == "2024.2"


def test_baseline_reproduction_gate(tmp_path, fake_env):
    from metasurface_optimizer.simulation.baseline import compare_responses
    b, opt, reg, t, sc = _fss_setup(tmp_path)
    from hfss_worker.protocol import sha256_file
    t.sha256 = sha256_file(Path(t.aedt_project))
    t.verification = {"passed": True, "sha256": t.sha256}  # verification itself is covered by the test above
    wcfg = {"python_executable": sys.executable, "require_validation_suite": False, "max_retries": 0, "progress_poll_s": 0.02, "plots": False}
    q = SharedFolderTransport(tmp_path / "q")
    ad = HFSSAdapter(t, b.schema, sc, tmp_path / "jobs", q, "HFSS", study="fss")
    # reference = the "paper" response of the existing design, here deliberately shifted (wrong template)
    from metasurface_optimizer.core.provenance import Provenance
    from metasurface_optimizer.electromagnetics.response import EMResponse
    from metasurface_optimizer.electromagnetics.sparameter import SParameterSet
    f = np.linspace(10e9, 20e9, 41)
    x = 6 * (f / 12e9 - 12e9 / f)
    ref = EMResponse(Provenance(Fidelity.REPORTED, "paper Fig. 3"), SParameterSet(f, {(1, 1, "TE"): 1 / (1 + 1j * x), (2, 1, "TE"): 1j * x / (1 + 1j * x)}))
    m = SimulationManager(opt, ad, tmp_path / "study", SimulationBudget(max_simulations=8, batch_size=3), baseline_reference=ref)
    for _ in range(2):
        m.generate_batch()
        run_queue(q.root, wcfg, "HFSS", once=True)
        m.collect_results()
    assert m.baseline_check is not None and not m.baseline_check["passed"]
    assert m.baseline_check["curves"]["S11_TE"]["resonance_shift_rel"] > 0.03
    with pytest.raises(BaselineNotReproduced):
        m.generate_batch()
    with pytest.raises(ValueError):
        m.override_baseline_gate("  ")
    m.override_baseline_gate("template intentionally differs: testing", "tester")
    assert m.generate_batch()["phase"] == "iterate"
    same = compare_responses(ref, ref)
    assert same["passed"] and not same["physically_meaningful"]
