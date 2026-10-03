"""HFSS validation suite (runs on the HFSS PC, PC-2).

The worker refuses candidate jobs for a template until these 12 checks pass on
that PC. All 12 rows come from ONE verification run of the template's baseline
design (solved once, for real, in HFSS), so the suite takes as long as one solve.

Run on the HFSS PC (in the Python that has PyAEDT):

    set HFSS_TEMPLATE_JSON=D:\\hfss_templates\\my_template.json     (registration + baseline_parameters + verification_config)
    set HFSS_WORKER_CONFIG=hfss_worker\\config.yaml                   (optional)
    python -m pytest tests/hfss -v

Without HFSS_TEMPLATE_JSON every test is skipped. HFSS_SUITE_USE_FAKE=1 runs
the suite against the FAKE PyAEDT in tests/fakes (checks the suite's own code;
says nothing about HFSS).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_SKIP = {"05", "06", "08"}


def _fake_template(tmp: Path) -> Path:
    sys.path.insert(0, str(ROOT))
    from hfss_worker.protocol import sha256_file
    from tests.fakes.fake_aedt import fake_project, fss_template_dict

    proj = fake_project(tmp / "fake_unit_cell.aedt")
    t = fss_template_dict(proj, sha256_file(proj))
    t["baseline_parameters"] = [{"name": "period", "variable": "p", "value": 10.0, "unit": "mm"},
                                {"name": "length_a", "variable": "L", "value": 8.0, "unit": "mm"},
                                {"name": "width_a", "variable": "W", "value": 1.0, "unit": "mm"}]
    t["verification_config"] = {"frequency": {"start_hz": 10e9, "stop_hz": 20e9, "points": 21},
                                "excitation": {"type": "floquet", "polarizations": ["TE", "TM"], "theta_deg": [0.0], "phi_deg": [0.0]},
                                "outputs": ["sparameters"], "checks": {"passivity_tol": 0.05}}
    p = tmp / "fake_template.json"
    p.write_text(json.dumps(t))
    return p


@pytest.fixture(scope="session")
def verification(tmp_path_factory):
    tj = os.environ.get("HFSS_TEMPLATE_JSON")
    fake = os.environ.get("HFSS_SUITE_USE_FAKE") == "1"
    if not tj and not fake:
        pytest.skip("HFSS validation suite: set HFSS_TEMPLATE_JSON on the HFSS PC (see tests/hfss/conftest.py)")
    sys.path.insert(0, str(ROOT))
    from hfss_worker.worker import load_config, verify_template

    cfg = load_config(os.environ.get("HFSS_WORKER_CONFIG"))
    if fake:
        tmp = tmp_path_factory.mktemp("fake_suite")
        tj = str(_fake_template(tmp))
        os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "tests" / "fakes" / "fake_pyaedt"), str(ROOT), os.environ.get("PYTHONPATH", "")])
        cfg = {**cfg, "python_executable": sys.executable, "validation_record": str(tmp / "validation_suite.json"), "progress_poll_s": 0.05}
    out = verify_template(Path(tj), cfg)
    rows = {r["id"]: r for r in out["rows"]}
    if len(rows) != 12:
        pytest.fail(f"verification produced {len(rows)} rows; state {out['result']['state']}; see {out['job_dir']}")
    return {"rows": rows, "out": out}


def check_row(verification, sid: str) -> None:
    r = verification["rows"][sid]
    if r["status"] == "SKIP" and sid in OPTIONAL_SKIP:
        pytest.skip(f"{r['name']}: {r['detail']}")
    assert r["status"] == "PASS", f"{r['id']} {r['name']}: {r['status']} - {r['detail']} (job folder {verification['out']['job_dir']})"
