"""FastAPI backend for the research dashboard.

Run:  uvicorn metasurface_optimizer.api.app:app --port 8000
(serves the built React UI from ``ui/dist`` when present).

Sessions hold one optimisation study (schema, problem, optimiser state, data)
and are pickled to the workspace so a human-in-the-loop study can span days.
Every response that contains model output is labeled PREDICTED; every uploaded
result keeps the fidelity the user declared.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cloudpickle
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from metasurface_optimizer.core.provenance import Fidelity, Provenance
from metasurface_optimizer.electromagnetics.rcs import to_dbsm
from metasurface_optimizer.electromagnetics.response_features import response_summary
from metasurface_optimizer.experiment import ExperimentBuilder, coerce_numbers, load_config, write_artifacts
from metasurface_optimizer.geometry.geometry_generator import available_templates
from metasurface_optimizer.optimization.active_learning import CounterfactualOptimizer
from metasurface_optimizer.optimization.pareto import pareto_mask
from metasurface_optimizer.solvers.base import BudgetedEvaluator
from metasurface_optimizer.tracking.experiment import SQLiteRepository, dumps

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("MSOPT_WORKSPACE", ROOT / "workspace"))
OUTPUTS = Path(os.environ.get("MSOPT_OUTPUTS", ROOT / "outputs"))
WORKSPACE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Metasurface counterfactual optimizer", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _finite(o: Any) -> Any:
    """Browsers cannot parse NaN/Infinity; non-finite numbers become null (e.g. -inf dBsm of a perfect null)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def J(obj: Any) -> JSONResponse:
    return JSONResponse(_finite(json.loads(dumps(obj))))


class Session:
    def __init__(self, cfg: Dict[str, Any]):
        self.id = uuid.uuid4().hex[:10]
        self.cfg = cfg
        self.builder = ExperimentBuilder(cfg, ROOT)
        self.opt: Optional[CounterfactualOptimizer] = None
        self.status: Dict[str, Any] = {"state": "created", "log": []}
        self.pending_plan: List[Dict[str, Any]] = []
        self.created = time.time()

    def log(self, msg: str) -> None:
        self.status["log"].append({"t": time.time(), "msg": msg})

    def ensure_opt(self) -> CounterfactualOptimizer:
        if self.opt is None:
            b = self.builder
            seed = int(self.cfg.get("seed", 0))
            self.opt = CounterfactualOptimizer(b.problem(), b.pguco_config(seed), b.prior_feature_fn())
        return self.opt

    @property
    def dir(self) -> Path:
        d = WORKSPACE / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self) -> None:
        with (self.dir / "session.pkl").open("wb") as f:
            cloudpickle.dump(self, f)  # closures (prior feature maps, torch input fns) need cloudpickle


SESSIONS: Dict[str, Session] = {}
LOCK = threading.Lock()


def get_session(sid: str) -> Session:
    if sid not in SESSIONS:
        p = WORKSPACE / sid / "session.pkl"
        if p.exists():
            SESSIONS[sid] = pickle.loads(p.read_bytes())
        else:
            raise HTTPException(404, f"session {sid} not found")
    return SESSIONS[sid]


# ------------------------------------------------------------------ meta
@app.get("/api/health")
def health():
    return {"ok": True, "workspace": str(WORKSPACE)}


@app.get("/api/templates")
def templates():
    return available_templates()


@app.get("/api/configs")
def configs():
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs" / "experiments").glob("*.yaml"))


@app.get("/api/sessions")
def list_sessions():
    out = []
    for p in sorted(WORKSPACE.glob("*/session.pkl")):
        try:
            s = get_session(p.parent.name)
            out.append({"id": s.id, "name": s.cfg.get("name"), "state": s.status["state"], "created": s.created})
        except Exception:
            continue
    return out


@app.post("/api/sessions")
def create_session(config_path: Optional[str] = Form(None), config_json: Optional[str] = Form(None)):
    if config_json:
        cfg = coerce_numbers(json.loads(config_json))
    else:
        cfg = load_config(ROOT / (config_path or "configs/experiments/demo_checkerboard.yaml"))
    s = Session(cfg)
    SESSIONS[s.id] = s
    s.save()
    return J({"id": s.id, "config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "schema": s.builder.schema.to_dict()})


@app.get("/api/sessions/{sid}/config")
def get_config(sid: str):
    s = get_session(sid)
    return J({"config": {k: v for k, v in s.cfg.items() if not k.startswith("_")}, "schema": s.builder.schema.to_dict()})


@app.put("/api/sessions/{sid}/config")
def update_config(sid: str, config_json: str = Form(...)):
    """Edit objectives / physics / optimiser / schema bounds before the study starts."""
    s = get_session(sid)
    if s.opt is not None and s.opt.dataset.records:
        raise HTTPException(409, "Study already has data; create a new session to change the configuration")
    new = coerce_numbers(json.loads(config_json))
    cfg = {**s.cfg, **new}
    sch = new.get("schema_overrides")
    s.cfg = cfg
    s.builder = ExperimentBuilder(cfg, ROOT)
    if sch:
        s.builder.schema = s.builder.schema.with_overrides(**sch)
    s.save()
    return J({"ok": True, "schema": s.builder.schema.to_dict()})


# ------------------------------------------------------------------ geometry
@app.get("/api/sessions/{sid}/geometry")
def geometry(sid: str, which: str = "baseline", values_json: Optional[str] = None):
    s = get_session(sid)
    b = s.builder
    if values_json:
        x = b.schema.to_vector({**{p.name: p.default for p in b.schema.parameters}, **json.loads(values_json)})
    elif which == "best" and s.opt is not None and s.opt.dataset.records:
        x = s.opt.best_record().x
    else:
        x = b.x0()
    geom = b.generator.build(x)
    polys = [[list(c) for c in p.exterior.coords] for p in geom.top_metal().geoms]
    val = s.ensure_opt().problem.validator.validate(x).to_dict()
    return J({"values": b.schema.to_dict_values(x), "period": [geom.period_x, geom.period_y], "polygons": polys,
              "layers": [{"name": l.name, "z0": l.z0, "thickness": l.thickness, "material": l.material} for l in geom.layers], "validation": val})


@app.post("/api/geometry/image")
async def parse_image(file: UploadFile = File(...), period_mm: Optional[float] = Form(None), mm_per_pixel: Optional[float] = Form(None)):
    from metasurface_optimizer.geometry.image_parser import analyze_mask, load_mask

    p = WORKSPACE / "uploads" / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(await file.read())
    if p.suffix.lower() == ".npy":
        mask = np.load(p).astype(bool)
    else:
        mask = load_mask(p)
    return J(analyze_mask(mask, mm_per_pixel, period_mm).to_dict())


@app.post("/api/geometry/stl")
async def parse_stl(file: UploadFile = File(...)):
    from metasurface_optimizer.geometry.stl_loader import load_stl

    p = WORKSPACE / "uploads" / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(await file.read())
    return J(load_stl(p, preserve_dir=WORKSPACE / "uploads" / "original_stl").to_dict())


# ------------------------------------------------------------------ baseline
def _resp_payload(resp, band=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"fidelity": resp.fidelity.value, "label": resp.fidelity.display}
    if resp.sparams is not None:
        sp = resp.sparams
        out["frequency"] = sp.frequency.tolist()
        out["sparams"] = {f"S{i}{j}_{p}": {"db": sp.db(i, j, p).tolist(), "phase": sp.phase(i, j, p).tolist(), "re": sp.real(i, j, p).tolist(),
                                            "im": sp.imag(i, j, p).tolist()} for (i, j, p) in sp.data}
    if resp.rcs is not None:
        r = resp.rcs
        out["rcs"] = {"frequency": r.frequency.tolist(), "theta_inc": r.theta_inc.tolist(), "polarizations": r.polarizations}
        try:
            out["rcs"]["specular_dbsm"] = to_dbsm(r.specular()).tolist()
        except ValueError:
            pass
        if r.monostatic is not None:
            out["rcs"]["monostatic_dbsm"] = to_dbsm(r.monostatic).tolist()
        if r.reference is not None:
            out["rcs"]["reference_dbsm"] = to_dbsm(r.reference).tolist()
        if r.bistatic is not None:
            k = len(r.frequency) // 2 if band is None else int(np.argmin(np.abs(r.frequency - 0.5 * (band[0] + band[1]))))
            out["rcs"]["bistatic_center"] = {"frequency": float(r.frequency[k]), "theta_s": r.theta_s.tolist(), "phi_s": r.phi_s.tolist(),
                                             "dbsm": to_dbsm(r.bistatic[0, 0, k]).tolist()}
    return out


@app.post("/api/sessions/{sid}/baseline")
async def set_baseline(sid: str, fidelity: str = Form("VALIDATED_SIMULATION"), use_oracle: bool = Form(False),
                       parameters_json: Optional[str] = Form(None), sparams: Optional[UploadFile] = File(None),
                       rcs: Optional[UploadFile] = File(None), freq_unit: str = Form("ghz")):
    from metasurface_optimizer.data.loaders.target_loader import load_response

    s = get_session(sid)
    if parameters_json:
        vals = json.loads(parameters_json)
        s.cfg.setdefault("baseline", {})["values"] = vals
        s.builder = ExperimentBuilder(s.cfg, ROOT)
        s.opt = None
    opt = s.ensure_opt()
    if use_oracle:
        resp = s.builder.solver().evaluate(s.builder.schema.to_dict_values(opt.problem.x0))
    else:
        paths = {}
        for key, up in (("sparams", sparams), ("rcs", rcs)):
            if up is not None:
                p = s.dir / f"baseline_{key}_{up.filename}"
                p.write_bytes(await up.read())
                paths[key] = p
        if not paths:
            raise HTTPException(400, "Upload S-parameters and/or RCS, or use the synthetic oracle")
        resp = load_response(Fidelity(fidelity), paths.get("sparams"), paths.get("rcs"), origin="user upload", freq_unit=freq_unit)
    J0 = opt.set_baseline(resp)
    s.status["state"] = "baseline_set"
    s.log(f"baseline set ({resp.fidelity.display})")
    s.save()
    return J({"J0": J0.tolist(), "objective_names": opt.problem.objectives.names, "response": _resp_payload(resp, opt.band)})


@app.get("/api/sessions/{sid}/baseline_analysis")
def baseline_analysis(sid: str):
    s = get_session(sid)
    opt = s.ensure_opt()
    if not opt.dataset.records:
        raise HTTPException(400, "Set the baseline first")
    base = opt.dataset.records[0]
    out = {"summary": response_summary(base.response, opt.band), "J0": base.J.tolist(), "objective_names": opt.problem.objectives.names,
           "response": _resp_payload(base.response, opt.band), "hf_sensitivity": opt.hf_sensitivity.to_dict() if opt.hf_sensitivity else None,
           "surrogate_sensitivity": opt.surrogate_sensitivity.to_dict() if opt.surrogate_sensitivity else None}
    return J(out)


# ------------------------------------------------------------------ loop
@app.post("/api/sessions/{sid}/initial_design")
def initial_design(sid: str):
    s = get_session(sid)
    opt = s.ensure_opt()
    plan = opt.initial_design()
    s.pending_plan = [{"x": d["x"], "tag": d["tag"], "values": s.builder.schema.to_dict_values(d["x"])} for d in plan]
    s.status["state"] = "awaiting_initial_results"
    s.save()
    return J({"designs": [{"index": i, "tag": d["tag"], "values": d["values"]} for i, d in enumerate(s.pending_plan)],
              "instruction": "Simulate these designs (HFSS/CST) and upload the results, or evaluate them with the synthetic oracle in demo mode."})


@app.post("/api/sessions/{sid}/results")
async def upload_result(sid: str, index: int = Form(...), fidelity: str = Form("VALIDATED_SIMULATION"),
                        sparams: Optional[UploadFile] = File(None), rcs: Optional[UploadFile] = File(None), freq_unit: str = Form("ghz")):
    """Mode B: ingest a high-fidelity result for pending design ``index``."""
    from metasurface_optimizer.data.loaders.target_loader import load_response

    s = get_session(sid)
    opt = s.ensure_opt()
    if index >= len(s.pending_plan):
        raise HTTPException(400, "No pending design with that index")
    paths = {}
    for key, up in (("sparams", sparams), ("rcs", rcs)):
        if up is not None:
            p = s.dir / f"result_{index}_{key}_{up.filename}"
            p.write_bytes(await up.read())
            paths[key] = p
    resp = load_response(Fidelity(fidelity), paths.get("sparams"), paths.get("rcs"), origin="user upload", freq_unit=freq_unit)
    d = s.pending_plan[index]
    rec = opt.observe(d["x"], resp, d["tag"])
    d["done"] = True
    s.log(f"result uploaded for {d['tag']}")
    s.save()
    return J({"record_id": rec.record_id, "J": rec.J.tolist(), "remaining": sum(1 for p in s.pending_plan if not p.get("done"))})


@app.post("/api/sessions/{sid}/evaluate_pending_with_oracle")
def evaluate_pending(sid: str):
    """DEMO ONLY: evaluate pending designs with the synthetic analytic oracle (labeled as such)."""
    s = get_session(sid)
    opt = s.ensure_opt()
    solver = s.builder.solver()
    n = 0
    for d in s.pending_plan:
        if not d.get("done"):
            opt.observe(d["x"], solver.evaluate(d["values"]), d["tag"])
            d["done"] = True
            n += 1
    s.save()
    return {"evaluated": n, "fidelity": Fidelity.SYNTHETIC_ORACLE.value}


@app.post("/api/sessions/{sid}/finish_initial")
def finish_initial(sid: str):
    s = get_session(sid)
    opt = s.ensure_opt()
    opt.finish_initial_design()
    s.status["state"] = "ready"
    s.save()
    return J({"hf_sensitivity": opt.hf_sensitivity.to_dict() if opt.hf_sensitivity else None, "best": opt.best_composite_hist})


@app.post("/api/sessions/{sid}/propose")
def propose(sid: str, k: int = Form(4)):
    s = get_session(sid)
    opt = s.ensure_opt()
    if opt.pending:
        opt.update_after_batch() if all(d.get("done") for d in s.pending_plan) else None
    props = opt.propose(k)
    s.pending_plan = [{"x": p["x"], "tag": f"iter{opt.iteration}_{p['kind']}", "values": p["values"]} for p in props]
    s.status["state"] = "awaiting_results"
    s.save()
    return J({"candidates": props, "label": "PREDICTED - requires high-fidelity validation", "pool_stats": opt.counterfactual.stats})


@app.post("/api/sessions/{sid}/complete_batch")
def complete_batch(sid: str):
    s = get_session(sid)
    opt = s.ensure_opt()
    if not opt.pending:
        raise HTTPException(400, "No proposed batch to complete; propose candidates first")
    if not all(d.get("done") for d in s.pending_plan):
        raise HTTPException(400, "Upload (or, in demo mode, oracle-evaluate) every proposed design before completing the batch")
    reason = opt.update_after_batch()
    s.status["state"] = "stopped" if reason else "ready"
    s.status["stop_reason"] = reason
    s.save()
    return {"stop_reason": reason, "history": opt.history[-1] if opt.history else None}


def _run_auto(s: Session) -> None:
    try:
        s.status["state"] = "running"
        b = s.builder
        opt = s.ensure_opt()
        ev = BudgetedEvaluator(b.solver(), b.budget())
        base = opt.dataset.records[0].response if opt.dataset.records else None
        if opt.dataset.records:  # restart cleanly from the stored baseline
            s.opt = None
            opt = s.ensure_opt()
        res = opt.run(ev, base)
        s.status["state"] = "finished"
        s.status["result_summary"] = {k: res[k] for k in ("best_validated", "n_hf", "stop_reason", "composite_improvement")}
        s.log("automatic run finished")
    except Exception as exc:  # pragma: no cover
        s.status["state"] = "error"
        s.status["error"] = f"{exc}\n{traceback.format_exc()}"
    s.save()


@app.post("/api/sessions/{sid}/run_auto")
def run_auto(sid: str):
    """Mode B with the configured automatic HF source (synthetic oracle in the demo)."""
    s = get_session(sid)
    if s.status["state"] == "running":
        raise HTTPException(409, "already running")
    threading.Thread(target=_run_auto, args=(s,), daemon=True).start()
    return {"started": True}


@app.post("/api/sessions/{sid}/predict_only")
def predict_only(sid: str, k: int = Form(10)):
    s = get_session(sid)
    opt = s.ensure_opt()
    prev = opt.pending
    out = opt.predict_only(k)
    opt.pending = prev  # Mode A ranks candidates only; it must not replace a batch awaiting validation
    s.save()
    return J(out)


@app.get("/api/sessions/{sid}/state")
def state(sid: str):
    s = get_session(sid)
    opt = s.opt
    out: Dict[str, Any] = {"id": s.id, "name": s.cfg.get("name"), "status": s.status, "budget": s.builder.budget(),
                           "pending": [{"index": i, "tag": d["tag"], "values": d["values"], "done": d.get("done", False)} for i, d in enumerate(s.pending_plan)]}
    if opt is None or not opt.dataset.records:
        return J(out)
    recs = opt.hf_records()
    comps = [opt.composite_of(r.J, r.x) for r in recs]
    names = opt.problem.objectives.names
    Fn = opt.problem.normalized(np.vstack([r.J for r in recs]))
    best = int(np.argmin(comps))
    out.update({
        "objective_names": names, "n_hf": opt.n_hf, "baseline": {"J": recs[0].J.tolist(), "composite": comps[0], "label": recs[0].fidelity.display},
        "best_validated": {"J": recs[best].J.tolist(), "composite": comps[best], "values": opt.schema.to_dict_values(recs[best].x), "label": recs[best].fidelity.display},
        "history": opt.history, "records": [{"tag": r.tag, "values": opt.schema.to_dict_values(r.x), "J": r.J.tolist(), "J_norm": Fn[i].tolist(),
                                             "composite": comps[i], "fidelity": r.fidelity.value, "pareto": bool(m)} for i, (r, m) in enumerate(zip(recs, pareto_mask(Fn)))],
        "best_predicted": min(opt.last_proposals, key=lambda p: p["mu"]) if opt.last_proposals else None,
        "proposals": opt.last_proposals, "proposals_label": "PREDICTED - requires high-fidelity validation",
        "hf_sensitivity": opt.hf_sensitivity.to_dict() if opt.hf_sensitivity else None,
        "surrogate_sensitivity": opt.surrogate_sensitivity.to_dict() if opt.surrogate_sensitivity else None,
        "prediction_log": opt.prediction_log, "trust_region_radius": float(opt.radius.mean()),
    })
    if getattr(opt, "last_pool", None) is not None:
        out["pool_pred_norm"] = opt.problem.normalized(opt.last_pool["mu_J"]).tolist()[:400]
        out["pool_sigma_median"] = float(np.median(opt.last_pool["sigma"]))
    return J(out)


@app.get("/api/sessions/{sid}/compare")
def compare(sid: str):
    """Baseline vs best validated responses (both with their labels) + predicted response of the best design."""
    s = get_session(sid)
    opt = s.ensure_opt()
    recs = opt.hf_records()
    if not recs:
        raise HTTPException(400, "no data")
    best = opt.best_record()
    out = {"baseline": _resp_payload(recs[0].response, opt.band), "optimized": _resp_payload(best.response, opt.band),
           "objective_names": opt.problem.objectives.names, "J0": recs[0].J.tolist(), "J_best": best.J.tolist()}
    if opt.surrogate is not None:
        p = opt.predict(best.x[None])
        d = opt.encoder.decode(p["Y"][0], p["S"][0])
        out["predicted"] = {"label": "PREDICTED", "frequency": d["frequency"].tolist(), "rcs_reduction_db": {k: v.tolist() for k, v in d["rcs_reduction_db"].items()},
                            "rcs_reduction_db_std": {k: v.tolist() for k, v in d["rcs_reduction_db_std"].items()}}
    return J(out)


@app.post("/api/sessions/{sid}/export")
def export(sid: str):
    s = get_session(sid)
    opt = s.ensure_opt()
    if opt.surrogate is None:
        raise HTTPException(400, "Run at least one proposal step before exporting")
    res = opt.results()
    from metasurface_optimizer.experiment import pguco_convergence

    res["convergence"] = pguco_convergence(opt)
    res["solver"] = s.builder.solver().describe() if s.builder.cfg["high_fidelity"]["kind"] == "synthetic_oracle" else {"name": "user uploads"}
    out_dir = OUTPUTS / f"{s.cfg.get('name', 'session')}_{s.id}_{time.strftime('%Y%m%d-%H%M%S')}"
    info = write_artifacts(s.builder, opt, res, out_dir)
    files = sorted(str(p.relative_to(OUTPUTS)) for p in out_dir.rglob("*") if p.is_file())
    return J({**info, "files": files})


@app.get("/api/files")
def files(path: str):
    p = (OUTPUTS / path).resolve()
    if not str(p).startswith(str(OUTPUTS.resolve())) or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p)


# ------------------------------------------------------------------ paper evaluation (engineering feature)
@app.post("/api/paper/extract")
async def paper_extract(file: UploadFile = File(...)):
    from metasurface_optimizer.ingestion.pdf_extractor import extract_paper

    d = WORKSPACE / "papers" / uuid.uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    p = d / file.filename
    p.write_bytes(await file.read())
    ex = extract_paper(p, d / "figures")
    return J({**ex.to_dict(), "paper_id": d.name,
              "note": "Evaluation feature only: the paper's own optimised design must not be used as training data or as an optimisation target."})


@app.get("/api/paper/figure")
def paper_figure(path: str):
    p = Path(path).resolve()
    if not str(p).startswith(str((WORKSPACE / "papers").resolve())):
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/paper/digitize")
def paper_digitize(figure_path: str = Form(...), calibration_json: str = Form(...), color_json: Optional[str] = Form(None),
                   tol: float = Form(0.15), n_points: Optional[int] = Form(None)):
    from metasurface_optimizer.ingestion.plot_digitizer import AxisCalibration, detect_axes, extract_curve, load_rgb

    p = Path(figure_path).resolve()
    if not str(p).startswith(str((WORKSPACE / "papers").resolve())):
        raise HTTPException(404)
    img = load_rgb(p)
    cal = AxisCalibration(**{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(calibration_json).items()})
    c = extract_curve(img, cal, tuple(json.loads(color_json)) if color_json else None, tol, None, n_points)
    return J({**c.to_dict(), "axes_proposal": detect_axes(img)})


@app.post("/api/paper/reconstruct")
def paper_reconstruct(sid: str = Form(...), extracted_json: str = Form(...), mapping_json: str = Form(...), user_values_json: Optional[str] = Form(None)):
    from metasurface_optimizer.core.provenance import ExtractedValue
    from metasurface_optimizer.ingestion.reconstruction import reconstruct

    s = get_session(sid)
    vals = [ExtractedValue(**{k: v for k, v in d.items() if k in ExtractedValue.__dataclass_fields__}) for d in json.loads(extracted_json)]
    rec = reconstruct(s.builder.schema, vals, json.loads(mapping_json), json.loads(user_values_json) if user_values_json else None)
    s.cfg.setdefault("baseline", {})["geometry_confidence"] = rec.confidence.value
    s.save()
    return J(rec.to_dict())


# ------------------------------------------------------------------ experiments / benchmarks
@app.get("/api/experiments")
def experiments():
    out = []
    for db in OUTPUTS.rglob("experiments.sqlite"):
        out += SQLiteRepository(db).list()
    return sorted(out, key=lambda r: r["timestamp"], reverse=True)


@app.get("/api/experiments/{eid}")
def experiment(eid: str):
    for db in OUTPUTS.rglob("experiments.sqlite"):
        r = SQLiteRepository(db).get(eid)
        if r:
            return r
    raise HTTPException(404)


@app.get("/api/benchmarks")
def benchmarks():
    import csv

    out = []
    for f in sorted(OUTPUTS.rglob("comparison_table.csv")):
        rows = list(csv.DictReader(f.open()))
        runs = f.parent / "benchmark_runs.json"
        curves = json.loads(runs.read_text())["curves"] if runs.exists() else {}
        out.append({"kind": "comparison", "path": str(f.relative_to(OUTPUTS)), "rows": rows, "curves": curves})
    for f in sorted(OUTPUTS.rglob("ablation_table.csv")):
        out.append({"kind": "ablation", "path": str(f.relative_to(OUTPUTS)), "rows": list(csv.DictReader(f.open()))})
    return out


from metasurface_optimizer.api.simulation_routes import register as _register_simulation  # noqa: E402

_register_simulation(app, get_session, J, WORKSPACE)

_UI = ROOT / "ui" / "dist"
if _UI.exists():
    app.mount("/", StaticFiles(directory=_UI, html=True), name="ui")
