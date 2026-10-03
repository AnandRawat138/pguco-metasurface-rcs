"""Song et al. (2016) metasurface: approximate model, objectives, and the HFSS batch round trip against the FAKE PyAEDT."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from metasurface_optimizer.experiment import ExperimentBuilder, load_config
from metasurface_optimizer.solvers.ring_disc_array import (FIG2_F_GHZ, FIG2_PHASE_DEG, FIG2_R1_MM, PAPER, RingDiscArrayModel, RingDiscCell,
                                                            layout_r1)
from tests.fakes.fake_aedt import fake_project

ROOT = Path(__file__).resolve().parents[1]
FAKE_PKG = ROOT / "tests" / "fakes" / "fake_pyaedt"


def test_cell_model_reproduces_fig2_and_is_passive():
    c = RingDiscCell()
    assert c.source == "fig2_calibrated_circuit"
    g = c.gamma(FIG2_F_GHZ * 1e9, FIG2_R1_MM)
    err = np.degrees(np.angle(g * np.exp(-1j * np.radians(FIG2_PHASE_DEG.T))))
    assert np.abs(err).max() < 1e-6
    G = c.gamma(np.linspace(6e9, 13e9, 30), np.linspace(1.5, 4.5, 25), theta_deg=30.0, pol="TM")
    assert np.all(np.abs(G) <= 1 + 1e-9)
    assert RingDiscCell(k2=0.9).source == "uncalibrated_circuit"


def test_layout_rule_matches_paper_structure():
    c = RingDiscCell()
    R1, sub = layout_r1(c, PAPER, 30, seed=1)
    assert R1.shape == (30, 30) and R1.min() >= 1.5 - 1e-9 and R1.max() <= 4.5 + 1e-9
    assert [int((sub == s).sum()) for s in range(4)] == [225] * 4  # 4 x 225 cells as in the paper
    assert sub[0, 0] == 0 and sub[0, 1] == 1 and sub[1, 0] == 2 and sub[1, 1] == 3
    R1b, _ = layout_r1(c, PAPER, 30, seed=1)
    np.testing.assert_array_equal(R1, R1b)  # fixed seed -> deterministic layout
    R1n, _ = layout_r1(c, {**PAPER, "r1_min": 2.0, "r1_max": 4.0, "phase_span": 0.7}, 30, seed=1)
    assert R1n.min() >= 2.0 - 1e-9 and R1n.max() <= 4.0 + 1e-9


def test_table_source_replaces_model(tmp_path):
    p = tmp_path / "t.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r1_mm", "f_ghz", "re", "im"])
        for r in (1.5, 3.0, 4.5):
            for fg in (6.0, 13.0):
                w.writerow([r, fg, np.cos(r), np.sin(r)])
    c = RingDiscCell.from_table(p)
    assert c.source == "hfss_table"
    np.testing.assert_allclose(c.gamma([9e9], [3.0])[0, 0], np.cos(3.0) + 1j * np.sin(3.0), atol=1e-12)


def test_model_rcs_and_objectives():
    m = RingDiscArrayModel({"theta_inc": [0.0], "n_freq": 21})
    f = np.linspace(6, 13, 21)
    band = (f >= 7) & (f <= 12)
    paper = m.evaluate(PAPER)
    uni = m.evaluate({**PAPER, "r1_min": 4.0, "r1_max": 4.0})
    red = lambda r: 10 * np.log10(r.rcs.monostatic / r.rcs.reference)[0, 0, band]
    assert red(paper).mean() < -10 and red(uni).mean() > -3  # random-phase layout reduces RCS, a uniform array does not
    assert paper.provenance.fidelity.value == "SYNTHETIC_ANALYTIC_ORACLE"
    b = ExperimentBuilder(load_config(ROOT / "configs/experiments/song2016_rcs.yaml"))
    b.cfg["high_fidelity"]["conditions"].update({"n_freq": 21})
    b.cfg["high_fidelity"]["bistatic"] = False
    b.cfg["high_fidelity"]["conditions"]["theta_inc"] = [0.0]
    b.cfg["objectives"] = [o for o in b.cfg["objectives"] if o["name"] != "angular"]
    prob = b.problem()
    J = prob.evaluate_response(b.solver().evaluate(b.schema.to_dict_values(prob.x0)), prob.x0)
    assert np.isfinite(J).all() and J[0] > J[1]  # worst frequency is worse than the band mean


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(FAKE_PKG), str(ROOT), os.environ.get("PYTHONPATH", "")]))


def test_hfss_batches_round_trip_with_fake_pyaedt(tmp_path, fake_env):
    from hfss_worker.worker import run_batch, verify_template
    from metasurface_optimizer.simulation.song2016 import collect_array_rcs, collect_cell_table, make_batches

    lay = tmp_path / "layouts"
    lay.mkdir()
    m = RingDiscArrayModel()
    for name, v in (("baseline_paper", PAPER), ("uniform_R1_4mm", {**PAPER, "r1_min": 4.0, "r1_max": 4.0})):
        R1, sub, _ = m.layout(v)
        with (lay / f"{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "col", "x_mm", "y_mm", "subarray", "r1_mm"])
            for i in range(30):
                for j in range(30):
                    w.writerow([i, j, 0, 0, sub[i, j] + 1, R1[i, j]])
    pc2 = tmp_path / "pc2"
    fake_project(pc2 / "song2016_unit_cell.aedt", design="UnitCell", variables={"p": "10mm", "R1": "3mm", "k2": "0.92", "k3": "0.8", "L": "6mm"},
                 excitations=["FloquetPort1"], boundaries=["LatticePair1", "LatticePair2", "Ground"], materials=["vacuum", "FR4_epoxy"])
    arr_vars = {f"r1_{i:02d}_{j:02d}": "3mm" for i in range(30) for j in range(30)}
    fake_project(pc2 / "song2016_array.aedt", design="Array", variables={**arr_vars, "theta_inc": "0deg"}, excitations=["IncPWave_TE"],
                 boundaries=["Rad1"], materials=["vacuum", "FR4_epoxy"])
    fake_project(pc2 / "song2016_plate.aedt", design="Plate", variables={"plate_L": "300mm", "theta_inc": "0deg"}, excitations=["IncPWave_TE"],
                 boundaries=["Rad1"], materials=["vacuum"])
    out = tmp_path / "hfss_jobs"
    info = make_batches(lay, out, str(pc2).replace("\\", "/"))
    wcfg = {"python_executable": sys.executable, "validation_record": str(pc2 / "state" / "validation_suite.json"), "max_retries": 0,
            "progress_poll_s": 0.02, "plots": False}
    for k in ("unit_cell", "array", "plate"):
        v = verify_template(Path(info["registrations"][k]), wcfg)
        assert v["result"]["state"] == "COMPLETED", (k, v["rows"])
    res = {k: run_batch(Path(info["batches"][k]["batch_dir"]), wcfg, "HFSS") for k in ("unit_cell", "array", "plate")}
    rep = collect_cell_table(res["unit_cell"], tmp_path / "cell_table.csv")
    assert rep["n_r1"] == 13
    cell = RingDiscCell.from_table(tmp_path / "cell_table.csv")
    assert cell.source == "hfss_table" and cell.table["G"].shape[0] == 13
    both = tmp_path / "array_and_plate"
    both.mkdir()
    import shutil
    for k in ("array", "plate"):
        shutil.copytree(res[k], both / k)
    summ = collect_array_rcs(both, tmp_path / "array_rcs.csv")
    assert set(summ) == {"baseline_paper@0deg", "uniform_R1_4mm@0deg"}
    assert json.loads((tmp_path / "array_rcs.summary.json").read_text())
