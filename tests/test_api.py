"""API test: the human-in-the-loop flow end to end on the synthetic oracle (demo mode)."""
import importlib

import pytest

pytest.importorskip("fastapi")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MSOPT_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("MSOPT_OUTPUTS", str(tmp_path / "out"))
    from fastapi.testclient import TestClient

    import metasurface_optimizer.api.app as appmod

    appmod = importlib.reload(appmod)
    return TestClient(appmod.app), appmod


def _ok(r):
    assert r.status_code == 200, r.text[:500]
    return r.json()


def test_human_in_the_loop_flow(client):
    c, appmod = client
    sid = _ok(c.post("/api/sessions", data={"config_path": "configs/experiments/demo_checkerboard.yaml"}))["id"]
    assert _ok(c.get(f"/api/sessions/{sid}/geometry"))["validation"]["valid"]
    # Proposing before any data is impossible; the baseline is required first.
    assert c.get(f"/api/sessions/{sid}/baseline_analysis").status_code == 400
    base = _ok(c.post(f"/api/sessions/{sid}/baseline", data={"use_oracle": "true"}))
    assert base["response"]["fidelity"] == "SYNTHETIC_ANALYTIC_ORACLE"
    n0 = len(_ok(c.post(f"/api/sessions/{sid}/initial_design"))["designs"])
    assert _ok(c.post(f"/api/sessions/{sid}/evaluate_pending_with_oracle"))["evaluated"] == n0
    _ok(c.post(f"/api/sessions/{sid}/finish_initial"))
    prop = _ok(c.post(f"/api/sessions/{sid}/propose", data={"k": "2"}))
    assert prop["label"].startswith("PREDICTED") and len(prop["candidates"]) == 2
    _ok(c.post(f"/api/sessions/{sid}/evaluate_pending_with_oracle"))
    _ok(c.post(f"/api/sessions/{sid}/complete_batch"))
    st = _ok(c.get(f"/api/sessions/{sid}/state"))
    assert st["n_hf"] == n0 + 2
    assert st["best_validated"]["composite"] <= st["baseline"]["composite"] + 1e-9
    # session survives a process restart (pickled to the workspace)
    appmod.SESSIONS.clear()
    assert _ok(c.get(f"/api/sessions/{sid}/state"))["n_hf"] == n0 + 2
    assert _ok(c.get(f"/api/sessions/{sid}/compare"))["predicted"]["label"] == "PREDICTED"
    files = _ok(c.post(f"/api/sessions/{sid}/export"))["files"]
    assert any(f.endswith(".stl") for f in files) and any(f.endswith("experiment.json") or f.endswith("report.json") for f in files)
    assert c.get("/api/files", params={"path": "../../etc/passwd"}).status_code == 404


def test_high_fidelity_simulation_tab_api(client, tmp_path):
    """MOCK loop and a manual-transfer round trip through the API (no HFSS involved)."""
    import io
    import json
    import shutil
    import zipfile
    from pathlib import Path

    from hfss_worker.worker import run_batch

    c, appmod = client
    sid = _ok(c.post("/api/sessions", data={"config_path": "configs/experiments/demo_checkerboard.yaml"}))["id"]
    ex = _ok(c.get("/api/sim/templates/examples"))["example_finite_rcs_template"]
    t = _ok(c.post("/api/sim/templates", data={"template_json": json.dumps(ex), "sid": sid}))
    assert t["status"] == "UNVERIFIED"
    assert "def main" in c.get("/api/sim/script").text
    common = {"template_id": t["template_id"], "max_simulations": "6", "batch_size": "3", "f_start_ghz": "6", "f_stop_ghz": "14",
              "points": "9", "polarizations": "TE,TM", "theta_deg": "0,20,40", "outputs": "sparameters,rcs"}
    r = c.post(f"/api/sessions/{sid}/sim/setup", data={**common, "mode": "HFSS"})
    assert r.status_code == 400 and "verify" in r.json()["detail"]  # unverified templates cannot run HFSS jobs
    st = _ok(c.post(f"/api/sessions/{sid}/sim/setup", data={**common, "mode": "MOCK", "transport": "mock"}))
    assert st["configured"] and st["mode"] == "MOCK" and st["summary"]["budget"]["remaining"] == 6
    for _ in range(2):
        _ok(c.post(f"/api/sessions/{sid}/sim/next_batch"))
    st = _ok(c.get(f"/api/sessions/{sid}/sim/status"))
    assert st["phase"] == "iterate" and st["summary"]["budget"]["used"] == 6
    assert all("MOCK" in j["label"] for j in st["jobs"])
    jid = st["jobs"][0]["job_id"]
    assert "MOCK" in c.get(f"/api/sessions/{sid}/sim/jobs/{jid}/logs").text
    assert c.get(f"/api/sessions/{sid}/sim/jobs/{jid}/download").headers["content-type"] == "application/zip"
    assert c.post(f"/api/sessions/{sid}/sim/next_batch").status_code == 400  # budget used up

    # manual transfer: package -> (HFSS PC) -> results zip -> import
    sid2 = _ok(c.post("/api/sessions", data={"config_path": "configs/experiments/demo_checkerboard.yaml"}))["id"]
    _ok(c.post(f"/api/sessions/{sid2}/sim/setup", data={**common, "mode": "DRY_RUN"}))
    b = _ok(c.post(f"/api/sessions/{sid2}/sim/next_batch"))["batch"]
    assert b["closed"] and "DRY RUN" in b["note"]
    v = _ok(c.post(f"/api/sim/templates/{t['template_id']}/verification_job",
                   data={"sid": sid2, "f_start_ghz": "6", "f_stop_ghz": "14", "points": "9", "theta_deg": "0"}))
    assert Path(v["zip"]).is_file() and "worker.py --input" in v["instruction"]
    assert c.get(v["download"]).headers["content-type"] == "application/zip"
    assert c.get("/api/sim/verification_download", params={"path": "../../etc/passwd"}).status_code == 404

    # MOCK through the manual-transfer path: download the batch, run the worker elsewhere, upload the results zip
    sid3 = _ok(c.post("/api/sessions", data={"config_path": "configs/experiments/demo_checkerboard.yaml"}))["id"]
    _ok(c.post(f"/api/sessions/{sid3}/sim/setup", data={**common, "mode": "MOCK", "transport": "manual"}))
    b = _ok(c.post(f"/api/sessions/{sid3}/sim/next_batch"))["batch"]
    z = c.get(f"/api/sessions/{sid3}/sim/batches/{b['transport']['batch_id']}/download")
    pc2 = tmp_path / "pc2"
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(pc2)
    out = run_batch(pc2 / b["transport"]["batch_id"], {"max_retries": 0, "plots": False}, "MOCK")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in out.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out.parent).as_posix())
    imp = _ok(c.post(f"/api/sessions/{sid3}/sim/import", files={"results_zip": ("r.zip", buf.getvalue(), "application/zip")}))
    assert [r["state"] for r in imp["imported"]] == ["COMPLETED"] and imp["open_batch"] is None
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../../evil.txt", "x")
    assert c.post(f"/api/sessions/{sid3}/sim/import", files={"results_zip": ("e.zip", bad.getvalue(), "application/zip")}).status_code >= 400
    shutil.rmtree(pc2)
