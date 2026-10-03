#!/usr/bin/env python
"""HFSS worker (PC-2): runs job packages produced by metasurface_optimizer on PC-1.

Modes
  Manual transfer (no network needed):
      python hfss_worker/worker.py --input batch_001            -> writes batch_001_results/
  Shared folder (PC-1 and PC-2 both see the folder):
      python hfss_worker/worker.py --queue \\\\server\\simulation_queue   (runs until stopped; --once to drain and exit)
  Template verification (the 12-step validation suite; required before candidate jobs run):
      python hfss_worker/worker.py --verify-template my_template.json [--baseline baseline_parameters.json]
  Environment check:
      python hfss_worker/worker.py --check-env
  MOCK (software test only, never physics; needs the framework installed):
      python hfss_worker/worker.py --input batch_001 --mock
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # allow "python hfss_worker/worker.py"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfss_worker import WORKER_VERSION  # noqa: E402
from hfss_worker.job_executor import execute_job  # noqa: E402
from hfss_worker.protocol import (JobState, now_iso, read_json, read_status, safe_join, unexpected_result_files,  # noqa: E402
                                  write_json_atomic)

QUEUE_DIRS = ("pending", "running", "completed", "failed", "archive")


# ------------------------------------------------------------------ config
def _parse_scalar(v: str) -> Any:
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_parse_scalar(x) for x in inner.split(",")] if inner else []
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def load_config(path: Optional[str]) -> Dict[str, Any]:
    p = Path(path) if path else Path(__file__).resolve().parent / "config.yaml"
    if not p.is_file():
        return {}
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        out: Dict[str, Any] = {}
        for line in text.splitlines():
            line = line.split(" #")[0].rstrip() if not line.lstrip().startswith("#") else ""
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = _parse_scalar(v)
        return out


def worker_id(cfg: Dict[str, Any]) -> str:
    return cfg.get("worker_id") or platform.node() or "worker"


# ------------------------------------------------------------------ env check
def check_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    py = cfg.get("python_executable") or sys.executable
    probe = ("import json,sys,platform\nout={'python':sys.version.split()[0],'os':platform.platform()}\n"
             "try:\n import ansys.aedt.core as m; out['pyaedt']=m.__version__; out['ns']='ansys.aedt.core'\n"
             "except ImportError:\n try:\n  import pyaedt as m; out['pyaedt']=m.__version__; out['ns']='pyaedt'\n except ImportError: out['pyaedt']=None\n"
             "print(json.dumps(out))")
    try:
        r = subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=120)
        info = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-500:]}
    except Exception as e:
        info = {"error": str(e)}
    roots = {k: v for k, v in os.environ.items() if k.startswith("ANSYSEM_ROOT")}
    info.update({"worker_version": WORKER_VERSION, "python_executable": py, "aedt_env": roots,
                 "aedt_version_configured": cfg.get("aedt_version")})
    vr = Path(cfg.get("validation_record", "worker_state/validation_suite.json"))
    info["verified_templates"] = {k: v.get("passed") for k, v in (read_json(vr).get("templates", {}) if vr.is_file() else {}).items()}
    return info


# ------------------------------------------------------------------ manual batch
def _copy_result(job_dir: Path, dest: Path) -> None:
    st = read_status(job_dir)
    rd = job_dir / st.get("final_result_dir", "")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if rd.is_dir():
        for p in rd.rglob("*"):
            if p.is_file():
                rel = p.relative_to(rd).as_posix()
                if rel in unexpected_result_files(rd):
                    continue
                t = safe_join(dest, rel)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, t)
    shutil.copy2(job_dir / "job_status.json", dest / "job_status.json")


def run_batch(batch_dir: Path, cfg: Dict[str, Any], mode: str, output: Optional[Path] = None) -> Path:
    batch_dir = Path(batch_dir).resolve()
    jobs_root = batch_dir / "jobs"
    if not jobs_root.is_dir():
        raise SystemExit(f"{batch_dir} has no jobs/ folder; is this a batch generated by the framework?")
    out = Path(output) if output else batch_dir.parent / f"{batch_dir.name}_results"
    out.mkdir(parents=True, exist_ok=True)
    wid = worker_id(cfg)
    summary: List[Dict[str, Any]] = []
    for jd in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
        r = execute_job(jd, cfg, mode, wid)
        _copy_result(jd, out / jd.name)
        summary.append({k: r.get(k) for k in ("job_id", "state", "attempts")} | {"candidate_id": read_json(jd / "job.json").get("candidate_id") if (jd / "job.json").is_file() else None})
    batch = read_json(batch_dir / "batch.json") if (batch_dir / "batch.json").is_file() else {}
    write_json_atomic(out / "batch_results.json", {"batch_id": batch.get("batch_id", batch_dir.name), "worker": wid, "mode": mode,
                                                   "worker_version": WORKER_VERSION, "finished": now_iso(), "jobs": summary})
    print(f"wrote {out} ({sum(1 for s in summary if s['state'] == 'COMPLETED')}/{len(summary)} completed)")
    return out


# ------------------------------------------------------------------ shared folder
def heartbeat(queue: Path, cfg: Dict[str, Any], mode: str, state: str, job: Optional[str] = None) -> None:
    write_json_atomic(queue / "workers" / f"{worker_id(cfg)}.json",
                      {"worker": worker_id(cfg), "mode": mode, "state": state, "job": job, "time": now_iso(), "unix": time.time(),
                       "worker_version": WORKER_VERSION, "aedt_version": cfg.get("aedt_version")})


def claim_next(queue: Path) -> Optional[Path]:
    """Atomically move the oldest pending job to running/. os.rename fails if another worker took it first."""
    pend = queue / "pending"
    for jd in sorted((p for p in pend.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name):
        target = queue / "running" / jd.name
        try:
            os.rename(jd, target)
        except OSError:
            continue
        return target
    return None


def run_queue(queue: Path, cfg: Dict[str, Any], mode: str, once: bool = False, max_jobs: Optional[int] = None) -> int:
    queue = Path(queue).resolve()
    for d in QUEUE_DIRS + ("workers",):
        (queue / d).mkdir(parents=True, exist_ok=True)
    n = 0
    wid = worker_id(cfg)
    while True:
        heartbeat(queue, cfg, mode, "idle")
        jd = claim_next(queue)
        if jd is None:
            if once:
                break
            time.sleep(float(cfg.get("poll_interval_s", 10)))
            continue
        heartbeat(queue, cfg, mode, "busy", jd.name)
        r = execute_job(jd, cfg, mode, wid)
        dest = queue / ("completed" if r["state"] == JobState.COMPLETED.value else "failed") / jd.name
        os.rename(jd, dest)
        n += 1
        if max_jobs is not None and n >= max_jobs:
            break
    heartbeat(queue, cfg, mode, "stopped")
    return n


# ------------------------------------------------------------------ template verification
def verify_template(template_json: Path, cfg: Dict[str, Any], baseline: Optional[Path] = None, sim_config: Optional[Path] = None) -> Dict[str, Any]:
    """Build a verify_template job from a template registration and run the 12-step suite on this PC."""
    from hfss_worker.package import write_job_package
    from hfss_worker.protocol import load_key, sanitize_id, sha256_file

    tmpl = read_json(template_json)
    if not tmpl.get("sha256") and tmpl.get("aedt_project") and Path(tmpl["aedt_project"]).is_file():
        tmpl["sha256"] = sha256_file(Path(tmpl["aedt_project"]))
    params = read_json(baseline)["parameters"] if baseline else tmpl.get("baseline_parameters", [])
    if not params:
        raise SystemExit("verification needs baseline parameters: --baseline file or template 'baseline_parameters'")
    scfg = read_json(sim_config) if sim_config else tmpl.get("verification_config")
    if not scfg:
        raise SystemExit("verification needs a simulation config: --sim-config file or template 'verification_config'")
    root = Path(cfg.get("validation_record", "worker_state/validation_suite.json")).parent / "verify"
    job_id = sanitize_id(f"verify_{tmpl['template_id']}_{time.strftime('%Y%m%d_%H%M%S')}")
    jd = write_job_package(root, job_id=job_id, candidate_id="baseline", parent_design_id="baseline", kind="verify_template",
                           simulation_type="HFSS", aedt_version=str(cfg.get("aedt_version", "2024.2")), framework_version=f"worker-{WORKER_VERSION}",
                           parameters=params, simulation_config=scfg, template=tmpl, key=load_key(None, cfg.get("job_key_file")))
    r = execute_job(jd, cfg, "HFSS", worker_id(cfg))
    st = read_status(jd)
    rows = read_json(jd / st["final_result_dir"] / "validation.json").get("rows", []) if (jd / st["final_result_dir"] / "validation.json").is_file() else []
    print(f"\nTemplate {tmpl['template_id']}: {r['state']}")
    for row in rows:
        print(f"  test_{row['id']}_{row['name']:<14} {row['status']:<5} {row['detail']}")
    return {"result": r, "rows": rows, "job_dir": str(jd)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="manual-transfer batch folder")
    g.add_argument("--queue", help="shared simulation_queue folder")
    g.add_argument("--verify-template", help="template registration JSON to verify")
    g.add_argument("--check-env", action="store_true")
    ap.add_argument("--output", help="results folder for --input (default <batch>_results)")
    ap.add_argument("--baseline", help="baseline parameters JSON for --verify-template")
    ap.add_argument("--sim-config", help="simulation_config JSON for --verify-template")
    ap.add_argument("--config", help="worker config (default hfss_worker/config.yaml)")
    ap.add_argument("--mock", action="store_true", help="MOCK mode: software test only, results are not physics")
    ap.add_argument("--once", action="store_true", help="queue mode: exit when the pending folder is empty")
    ap.add_argument("--max-jobs", type=int)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    mode = "MOCK" if a.mock else "HFSS"
    if a.check_env:
        print(json.dumps(check_env(cfg), indent=2))
        return 0
    if a.verify_template:
        r = verify_template(Path(a.verify_template), cfg, Path(a.baseline) if a.baseline else None, Path(a.sim_config) if a.sim_config else None)
        return 0 if r["result"]["state"] == "COMPLETED" else 1
    if a.input:
        run_batch(Path(a.input), cfg, mode, Path(a.output) if a.output else None)
        return 0
    run_queue(Path(a.queue), cfg, mode, once=a.once, max_jobs=a.max_jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
