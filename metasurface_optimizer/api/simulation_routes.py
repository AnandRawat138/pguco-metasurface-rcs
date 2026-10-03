"""API for the High-Fidelity Simulation tab (two-PC HFSS automation).

Registered from ``app.py`` via :func:`register`. A study's simulation manager
lives on its Session (``session.sim``) and is pickled with it.
"""
from __future__ import annotations

import io
import json
import shutil
import tempfile
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from hfss_worker.protocol import read_json, read_status, safe_join
from metasurface_optimizer.simulation.hfss_adapter import HFSSAdapter, make_simulation_config
from metasurface_optimizer.simulation.job_manager import BaselineNotReproduced, SimulationBudget, SimulationManager
from metasurface_optimizer.simulation.templates import TemplateRegistry, TemplateSpec
from metasurface_optimizer.simulation.transports import LocalMockTransport, ManualBatchTransport, SharedFolderTransport

MAX_UPLOAD = 2 * 1024 ** 3


def _safe_unzip(data: bytes, dest: Path) -> Path:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise HTTPException(400, "not a zip archive")
    with z:
        total = 0
        for info in z.infolist():
            if info.is_dir():
                continue
            total += info.file_size
            if total > MAX_UPLOAD:
                raise HTTPException(400, "archive too large")
            try:
                target = safe_join(dest, info.filename)
            except ValueError as e:
                raise HTTPException(400, f"refused archive entry: {e}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
    return dest


def _find_results_root(root: Path) -> Path:
    """The folder whose sub-folders are job results (accepts a zip of batch_NNN_results/ or of its contents)."""
    if (root / "batch_results.json").is_file():
        return root
    for p in root.rglob("batch_results.json"):
        return p.parent
    subs = [p for p in root.iterdir() if p.is_dir()]
    if len(subs) == 1:
        return subs[0]
    return root


def _zip_dir(d: Path) -> Path:
    out = Path(tempfile.mkdtemp()) / f"{d.name}.zip"
    shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=d.parent, base_dir=d.name)
    return out


def register(app, get_session, J, workspace: Path) -> None:
    reg_root = workspace / "hfss_templates"

    def registry() -> TemplateRegistry:
        return TemplateRegistry(reg_root)

    def manager(sid: str) -> SimulationManager:
        s = get_session(sid)
        m = getattr(s, "sim", None)
        if m is None:
            raise HTTPException(400, "simulation is not set up for this study; choose a template and mode first")
        return m

    # ------------------------------------------------------------ templates
    @app.get("/api/sim/templates")
    def sim_templates():
        return J({"templates": registry().list()})

    @app.post("/api/sim/templates")
    def sim_register(template_json: str = Form(...), sid: Optional[str] = Form(None), replace: bool = Form(False)):
        try:
            spec = TemplateSpec.from_dict(json.loads(template_json))
            schema = get_session(sid).builder.schema if sid else None
            t = registry().register(spec, schema, replace=replace)
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(400, str(e))
        return J(t.to_dict())

    @app.get("/api/sim/templates/examples")
    def sim_template_examples():
        root = Path(__file__).resolve().parents[2] / "configs" / "hfss"
        return J({p.stem: json.loads(p.read_text()) for p in sorted(root.glob("example_*.json"))})

    @app.post("/api/sim/templates/{tid}/verification_job")
    def sim_verification_job(tid: str, sid: str = Form(...), f_start_ghz: float = Form(...), f_stop_ghz: float = Form(...),
                             points: int = Form(101), polarizations: str = Form("TE,TM"), theta_deg: str = Form("0")):
        """Package the 12-step verification (baseline design x0) for the HFSS PC."""
        s = get_session(sid)
        t = registry().get(tid)
        sc = make_simulation_config(t, f_start_ghz * 1e9, f_stop_ghz * 1e9, points, polarizations.split(","),
                                    [float(v) for v in theta_deg.split(",")])
        root = reg_root / "verification" / f"{tid}_{time.strftime('%Y%m%d_%H%M%S')}"
        ad = HFSSAdapter(t, s.builder.schema, sc, root / "jobs", ManualBatchTransport(root / "transfer"), "HFSS", study=f"verify_{tid}")
        x0 = s.builder.x0()
        d = ad.generate_job_package({"x": x0, "values": s.builder.schema.to_dict_values(x0)}, "verify_template", 0, "baseline")
        from metasurface_optimizer.simulation.solver_adapter import SimulationJob
        job = SimulationJob(d.name, "baseline", "verify_template", 0, str(d), "HFSS")
        info = ad.transport.submit([job])
        info["download"] = f"/api/sim/verification_download?path={Path(info['zip']).relative_to(reg_root).as_posix()}"
        return J({**info, "instruction": f"Copy {info['zip']} to the HFSS PC, unzip, run: python hfss_worker/worker.py --input {info['batch_id']} ; "
                                         "then import the results folder here."})

    @app.get("/api/sim/verification_download")
    def sim_verification_download(path: str):
        try:
            p = safe_join(reg_root, path)
        except ValueError:
            raise HTTPException(404, "not found")
        if p.suffix != ".zip" or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(p, filename=p.name)

    @app.post("/api/sim/templates/{tid}/import_verification")
    async def sim_import_verification(tid: str, results_zip: UploadFile = File(...)):
        tmp = Path(tempfile.mkdtemp())
        _safe_unzip(await results_zip.read(), tmp)
        root = _find_results_root(tmp)
        cands = [p for p in root.rglob("result.json") if read_json(p).get("kind") == "verify_template"]
        if not cands:
            raise HTTPException(400, "no verification result in the archive")
        try:
            t = registry().record_verification(tid, cands[0].parent)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return J(t.to_dict())

    @app.get("/api/sim/script", response_class=PlainTextResponse)
    def sim_script():
        return HFSSAdapter.generate_pyaedt_script()

    # ------------------------------------------------------------ study setup
    @app.post("/api/sessions/{sid}/sim/setup")
    def sim_setup(sid: str, template_id: str = Form(...), mode: str = Form("MOCK"), transport: str = Form("manual"),
                  queue_path: Optional[str] = Form(None), max_simulations: int = Form(...), batch_size: int = Form(4),
                  max_iterations: Optional[int] = Form(None), f_start_ghz: float = Form(...), f_stop_ghz: float = Form(...),
                  points: int = Form(101), polarizations: str = Form("TE,TM"), theta_deg: str = Form("0"),
                  outputs: str = Form("sparameters"), aedt_version: str = Form("2024.2")):
        s = get_session(sid)
        mode = mode.upper()
        if mode not in ("HFSS", "MOCK", "DRY_RUN"):
            raise HTTPException(400, "mode must be HFSS, MOCK or DRY_RUN")
        opt = s.ensure_opt()
        if opt.dataset.records and getattr(s, "sim", None) is None:
            raise HTTPException(400, "this study already holds results from another source; start a new study for simulation-driven optimization")
        t = registry().get(template_id)
        if mode == "HFSS" and not t.verified:
            raise HTTPException(400, f"template {template_id} is {t.status()}; verify it on the HFSS PC first")
        mock = None
        if mode == "MOCK":
            oc = dict((s.cfg.get("high_fidelity") or {}).get("conditions") or {})
            mock = {"oracle_conditions": {k: v for k, v in oc.items() if k not in ("f_start", "f_stop", "n_freq", "theta_inc", "polarizations")}}
            mock["oracle_conditions"].setdefault("theta_s_step", 6.0)
            mock["oracle_conditions"].setdefault("phi_s_step", 30.0)
        sc = make_simulation_config(t, f_start_ghz * 1e9, f_stop_ghz * 1e9, points, polarizations.split(","),
                                    [float(v) for v in theta_deg.split(",")], outputs=outputs.split(","), mock=mock)
        root = s.dir / "simulation"
        if transport == "mock":
            if mode != "MOCK":
                raise HTTPException(400, "the local mock transport only runs MOCK jobs")
            tr = LocalMockTransport(root / "transfer", {"max_retries": 2, "plots": False})
        elif transport == "shared_folder":
            if not queue_path:
                raise HTTPException(400, "shared-folder mode needs the queue folder path")
            tr = SharedFolderTransport(Path(queue_path))
        else:
            tr = ManualBatchTransport(root / "transfer")
        ad = HFSSAdapter(t, s.builder.schema, sc, root / "jobs", tr, mode, aedt_version=aedt_version, generator=s.builder.generator,
                         study=s.id)
        s.sim = SimulationManager(opt, ad, root / "study", SimulationBudget(max_simulations, batch_size, max_iterations))
        s.status["state"] = "simulation_ready"
        s.save()
        return sim_status(sid)

    @app.get("/api/sessions/{sid}/sim/status")
    def sim_status(sid: str):
        s = get_session(sid)
        m = getattr(s, "sim", None)
        tmpl = registry().list()
        if m is None:
            return {"configured": False, "templates": tmpl}
        m.refresh()
        tr = m.adapter.transport
        versions = next((r.info.get("versions") for r in reversed(list(m.results.values())) if r.info.get("versions")), None)
        out = {"configured": True, "mode": m.mode, "transport": getattr(tr, "kind", None), "template": m.adapter.template.to_dict(),
               "simulation_config": m.adapter.simulation_config, "summary": m.summary(), "jobs": m.jobs_table(), "batches": m.batches,
               "rejected": m.rejected, "phase": m.opt.phase, "open_batch": m.open_batch, "versions": versions,
               "templates": tmpl, "running": bool(getattr(s, "_sim_thread_busy", False)), "error": getattr(s, "sim_error", None)}
        if isinstance(tr, SharedFolderTransport):
            out["queue"] = {"counts": tr.counts(), "workers": tr.workers(), "path": str(tr.root)}
        return J(out)

    # ------------------------------------------------------------ loop
    @app.post("/api/sessions/{sid}/sim/next_batch")
    def sim_next_batch(sid: str, k: Optional[int] = Form(None)):
        s, m = get_session(sid), manager(sid)
        try:
            b = m.generate_batch(k)
        except BaselineNotReproduced as e:
            raise HTTPException(409, str(e))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        if isinstance(m.adapter.transport, LocalMockTransport):
            m.collect_results()
        s.save()
        return J({"batch": b, "open_batch": m.open_batch})

    @app.post("/api/sessions/{sid}/sim/collect")
    def sim_collect(sid: str):
        s, m = get_session(sid), manager(sid)
        got = m.collect_results()
        s.save()
        return J({"collected": [{"job_id": r.job_id, "state": r.state, "error": r.error} for r in got], "open_batch": m.open_batch})

    @app.post("/api/sessions/{sid}/sim/import")
    async def sim_import(sid: str, results_zip: UploadFile = File(...)):
        s, m = get_session(sid), manager(sid)
        tmp = Path(tempfile.mkdtemp())
        _safe_unzip(await results_zip.read(), tmp)
        try:
            got = m.import_batch(_find_results_root(tmp))
        except (ValueError, RuntimeError, FileExistsError) as e:
            raise HTTPException(400, str(e))
        s.save()
        return J({"imported": [{"job_id": r.job_id, "state": r.state, "error": r.error} for r in got], "open_batch": m.open_batch})

    @app.post("/api/sessions/{sid}/sim/run")
    def sim_run(sid: str):
        """Drive the loop to the end (MOCK, or shared folder with a running worker)."""
        s, m = get_session(sid), manager(sid)
        if getattr(s, "_sim_thread_busy", False):
            raise HTTPException(409, "already running")
        if type(m.adapter.transport) is ManualBatchTransport:
            raise HTTPException(400, "manual transfer needs you to carry each batch; use 'Next batch' and 'Import results'")

        def work():
            s._sim_thread_busy = True
            s.sim_error = None
            try:
                m.run(poll_s=5.0)
            except Exception as e:  # pragma: no cover - surfaced in the status
                s.sim_error = f"{e}\n{traceback.format_exc()}"
            finally:
                s._sim_thread_busy = False
                s.save()
        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    @app.post("/api/sessions/{sid}/sim/override_baseline")
    def sim_override(sid: str, reason: str = Form(...)):
        s, m = get_session(sid), manager(sid)
        try:
            m.override_baseline_gate(reason, "dashboard user")
        except ValueError as e:
            raise HTTPException(400, str(e))
        s.save()
        return J({"override": m.override})

    @app.post("/api/sessions/{sid}/sim/jobs/{job_id}/cancel")
    def sim_cancel(sid: str, job_id: str):
        s, m = get_session(sid), manager(sid)
        ok = m.cancel(job_id)
        s.save()
        return {"cancelled": ok, "note": None if ok else "not cancellable from here (manual batches: skip it on the HFSS PC)"}

    @app.post("/api/sessions/{sid}/sim/jobs/{job_id}/retry")
    def sim_retry(sid: str, job_id: str):
        s, m = get_session(sid), manager(sid)
        try:
            m.retry(job_id)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(400, str(e))
        s.save()
        return {"queued_for_next_batch": job_id}

    @app.post("/api/sessions/{sid}/sim/jobs/{job_id}/archive")
    def sim_archive(sid: str, job_id: str):
        m = manager(sid)
        tr = m.adapter.transport
        if not isinstance(tr, SharedFolderTransport):
            raise HTTPException(400, "archiving applies to the shared-folder queue")
        tr.archive(m.jobs[job_id])
        return {"archived": job_id}

    @app.get("/api/sessions/{sid}/sim/jobs/{job_id}/download")
    def sim_download_job(sid: str, job_id: str, what: str = "package"):
        m = manager(sid)
        job = m.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        d = Path(job.job_dir if what == "package" else (job.result_dir or ""))
        if not d.is_dir():
            raise HTTPException(404, "no result yet" if what != "package" else "package missing")
        return FileResponse(_zip_dir(d), filename=f"{job_id}_{what}.zip")

    @app.get("/api/sessions/{sid}/sim/jobs/{job_id}/logs", response_class=PlainTextResponse)
    def sim_logs(sid: str, job_id: str):
        m = manager(sid)
        job = m.jobs[job_id]
        parts = []
        if job.result_dir and (Path(job.result_dir) / "hfss.log").is_file():
            parts.append((Path(job.result_dir) / "hfss.log").read_text(errors="replace"))
        if job.result_dir and (Path(job.result_dir) / "job_status.json").is_file():
            parts.append(json.dumps(read_json(Path(job.result_dir) / "job_status.json"), indent=2))
        tr = m.adapter.transport
        if not parts and isinstance(tr, SharedFolderTransport):
            p = tr.locate(job_id)
            if p is not None:
                parts.append(json.dumps(read_status(p), indent=2))
        return "\n\n".join(parts) or "no log yet"

    @app.get("/api/sessions/{sid}/sim/batches/{batch_id}/download")
    def sim_download_batch(sid: str, batch_id: str):
        m = manager(sid)
        tr = m.adapter.transport
        z = Path(tr.root) / f"{batch_id}.zip" if hasattr(tr, "root") else None
        if z is None or not z.is_file():
            raise HTTPException(404, "no such batch archive")
        return FileResponse(z, filename=z.name)

    @app.post("/api/sessions/{sid}/sim/export")
    def sim_export(sid: str):
        from metasurface_optimizer.simulation.exports import export_final_outputs
        s, m = get_session(sid), manager(sid)
        if m.opt.phase != "iterate":
            raise HTTPException(400, "nothing to export before the initial design has been simulated")
        out = export_final_outputs(s.builder, m, s.dir / "simulation" / "final_outputs")
        s.save()
        return J(out)
