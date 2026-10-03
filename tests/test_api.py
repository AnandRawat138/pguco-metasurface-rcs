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
