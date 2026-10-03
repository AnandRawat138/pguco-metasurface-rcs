# ---- BEGIN runner_core (inlined into run_hfss.py; standard library + PyAEDT only) ----
"""Body of the self-contained ``run_hfss.py`` executed in AEDT's Python on the HFSS PC.

It never builds geometry or guesses a setup. It opens a *verified template
project*, sets only explicitly mapped variables (with units), checks every
configuration step against the template registration, solves, extracts the
requested outputs and writes the result package. Every check is one of the
12 rows of ``validation.json``; any failed check ends the job as
FAILED_VALIDATION instead of producing a silently wrong simulation.

Exit codes: 0 COMPLETED, 2 FAILED_VALIDATION, 1 FAILED.
"""
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback

# ---- INLINE-SKIP-BEGIN ----
from hfss_worker.runner.version_compatibility import (  # noqa: F401  (inlined copy is used in run_hfss.py)
    COMPAT_LOG, analyze, boundary_names, configure_sweep, excitation_names, excitation_names_by_type, expression_complex,
    expression_real, get_setup, get_variable, import_pyaedt, material_exists, normalize_aedt_version, open_hfss, parse_freq_hz,
    release, save_project, set_variable, setup_names, solution_data, solution_type, sweep_names, sweep_values_hz,
    validate_design, variable_names, versions,
)
# ---- INLINE-SKIP-END ----

VALIDATION_STEPS = [
    ("01", "connection", "PyAEDT imports and AEDT starts with the template project"),
    ("02", "version", "AEDT version matches the version the job was built for"),
    ("03", "open_template", "Template file matches its registered SHA-256; design and solution type match"),
    ("04", "variables", "Every mapped design variable exists, is set with units and reads back"),
    ("05", "geometry", "AEDT design validation passes after the variable update"),
    ("06", "materials", "Registered materials exist in the project"),
    ("07", "excitations", "Registered excitations and angle/polarization controls exist and match the requested excitation"),
    ("08", "boundaries", "Registered boundaries exist"),
    ("09", "setup_sweep", "Setup and sweep exist; requested frequency sweep is applied and reads back"),
    ("10", "solve", "Every requested incidence condition solves"),
    ("11", "extraction", "Requested outputs are extracted, finite, on the requested grid and passive"),
    ("12", "cleanup", "Result package written and AEDT released"),
]

UNIT_SCALE = {"mm": 1e-3, "um": 1e-6, "nm": 1e-9, "cm": 1e-2, "m": 1.0, "mil": 2.54e-5, "in": 0.0254, "meter": 1.0,
              "deg": 1.0, "degree": 1.0, "rad": 1.0, "ghz": 1e9, "mhz": 1e6, "hz": 1.0, "": 1.0}


class ValidationFailure(Exception):
    pass


class Recorder:
    def __init__(self, log):
        self.rows = {sid: {"id": sid, "name": name, "check": desc, "status": "NOT_RUN", "detail": ""} for sid, name, desc in VALIDATION_STEPS}
        self.log = log

    def ok(self, sid, detail=""):
        self.rows[sid].update(status="PASS", detail=str(detail))
        self.log("[%s %s] PASS %s" % (sid, self.rows[sid]["name"], detail))

    def skip(self, sid, detail=""):
        self.rows[sid].update(status="SKIP", detail=str(detail))
        self.log("[%s %s] SKIP %s" % (sid, self.rows[sid]["name"], detail))

    def fail(self, sid, detail):
        self.rows[sid].update(status="FAIL", detail=str(detail))
        self.log("[%s %s] FAIL %s" % (sid, self.rows[sid]["name"], detail))
        raise ValidationFailure("%s %s: %s" % (sid, self.rows[sid]["name"], detail))

    def as_list(self):
        return [self.rows[sid] for sid, _, _ in VALIDATION_STEPS]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonable(o, depth=0):
    if depth > 8:
        return str(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, dict) or hasattr(o, "items"):
        try:
            return {str(k): _jsonable(v, depth + 1) for k, v in o.items()}
        except Exception:
            return str(o)
    if isinstance(o, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in o]
    return str(o)


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _load(job_dir, name):
    with open(os.path.join(job_dir, name)) as f:
        return json.load(f)


def _expression(value, unit):
    v = float(value)
    if not math.isfinite(v):
        raise ValidationFailure("non-finite parameter value %r" % value)
    unit = str(unit or "")
    if unit and not unit.replace("_", "").isalnum():
        raise ValidationFailure("invalid unit %r" % unit)
    return "%.12g%s" % (v, unit)


def _close(a, b, rel=1e-6, abs_=1e-15):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def _norm_expr(s):
    return str(s).replace(" ", "").lower()


def _incidences(cfg_exc):
    thetas = [float(t) for t in cfg_exc.get("theta_deg", [0.0])] or [0.0]
    phis = [float(p) for p in cfg_exc.get("phi_deg", [0.0])] or [0.0]
    return [(t, p) for t in thetas for p in phis]


def run(job_dir, out_dir, pyaedt_module=None, non_graphical=True, cores=4, keep_project=False):
    """Execute one job package. Returns (state, error_message)."""
    job_dir = os.path.abspath(job_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_f = open(os.path.join(out_dir, "hfss.log"), "a")
    t_start = time.time()

    def log(msg):
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        log_f.write(line + "\n")
        log_f.flush()
        print(line)

    def progress(state):
        _write_json(os.path.join(out_dir, "progress.json"), {"state": state, "time": time.time()})

    rec = Recorder(log)
    job = _load(job_dir, "job.json")
    params = _load(job_dir, "parameters.json")
    cfg = _load(job_dir, "simulation_config.json")
    tmpl = _load(job_dir, "template.json")
    meta = {"job_id": job.get("job_id"), "candidate_id": job.get("candidate_id"), "template_id": tmpl.get("template_id"),
            "template_sha256_registered": tmpl.get("sha256"), "template_sha256": None, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "solves": []}
    files = {}
    state, error = "COMPLETED", None
    app = None
    released = False
    try:
        progress("RUNNING")
        log("job %s (candidate %s) template %s" % (job.get("job_id"), job.get("candidate_id"), tmpl.get("template_id")))
        # ---------------- 03 (file part): template integrity, copy into the job's work folder
        src = tmpl.get("aedt_project")
        if not src or not os.path.isfile(src):
            rec.fail("03", "template project not found: %s" % src)
        h = _sha256(src)
        meta["template_sha256"] = h
        if tmpl.get("sha256"):
            if h != tmpl["sha256"]:
                rec.fail("03", "template file changed since registration (sha256 %s != %s); re-verify it" % (h[:12], tmpl["sha256"][:12]))
        work = out_dir.rstrip("/\\") + "_work"
        os.makedirs(work, exist_ok=True)
        project = os.path.join(work, os.path.basename(src))
        shutil.copy2(src, project)
        # ---------------- 01 connection
        try:
            if pyaedt_module is None:
                mod, ns, pv = import_pyaedt()
            else:
                mod, ns, pv = pyaedt_module, getattr(pyaedt_module, "__name__", "pyaedt"), str(getattr(pyaedt_module, "__version__", "unknown"))
            app = open_hfss(mod, project, tmpl["design"], job.get("aedt_version"), non_graphical=non_graphical, new_desktop=True)
        except ValidationFailure:
            raise
        except Exception as e:
            rec.fail("01", "could not start AEDT / open the template: %s" % e)
        rec.ok("01", "PyAEDT %s (%s)" % (pv, ns))
        # ---------------- 02 version
        vers = versions(app, ns, pv)
        meta["versions"] = vers
        want = normalize_aedt_version(job.get("aedt_version")) if job.get("aedt_version") else None
        if want and vers.get("aedt_version") and vers["aedt_version"] != want:
            rec.fail("02", "AEDT %s running but the job targets %s" % (vers["aedt_version"], want))
        rec.ok("02", "AEDT %s, Python %s, %s" % (vers.get("aedt_version"), vers["python_version"], vers["os"]))
        # ---------------- 03 design / solution type
        dname = getattr(app, "design_name", None)
        if dname is not None and dname != tmpl["design"]:
            rec.fail("03", "opened design %r, expected %r" % (dname, tmpl["design"]))
        st = solution_type(app)
        if tmpl.get("solution_type") and st is not None and str(st).lower() != str(tmpl["solution_type"]).lower():
            rec.fail("03", "solution type %r, template registered %r" % (st, tmpl["solution_type"]))
        rec.ok("03", "design %s, solution type %s, sha256 ok" % (tmpl["design"], st))
        # ---------------- 04 variables
        existing = set(variable_names(app))
        plist = params["parameters"]
        missing = [p["variable"] for p in plist if p["variable"] not in existing]
        if missing:
            rec.fail("04", "variables missing in the template: %s" % ", ".join(missing))
        set_log = []
        for p in plist:
            expr = _expression(p["value"], p.get("unit", ""))
            set_variable(app, p["variable"], expr)
            got_expr, got_si = get_variable(app, p["variable"])
            want_si = float(p["value"]) * UNIT_SCALE.get(str(p.get("unit", "")).lower(), float("nan"))
            if got_si is not None and math.isfinite(want_si) and str(p.get("unit", "")).lower() not in ("deg", "degree", "rad"):
                ok = _close(got_si, want_si)
            else:
                ok = got_expr is not None and _norm_expr(got_expr) == _norm_expr(expr)
            if not ok:
                rec.fail("04", "%s read back %r (si %r), expected %s" % (p["variable"], got_expr, got_si, expr))
            set_log.append({"variable": p["variable"], "set": expr, "read_back": got_expr})
        meta["variables"] = set_log
        rec.ok("04", "%d variables set and read back" % len(set_log))
        # ---------------- 05 geometry
        v = validate_design(app)
        if v is False:
            rec.fail("05", "AEDT validation failed after setting the variables (overlaps, invalid geometry or setup); see hfss.log")
        if v is None:
            rec.skip("05", "validate_simple not available in this PyAEDT")
        else:
            rec.ok("05", "validate_simple passed")
        # ---------------- 06 materials
        mats = tmpl.get("materials") or []
        if not mats:
            rec.skip("06", "no materials registered")
        else:
            bad = [m for m in mats if material_exists(app, m) is False]
            if bad:
                rec.fail("06", "materials missing: %s" % ", ".join(bad))
            rec.ok("06", ", ".join(mats))
        # ---------------- 07 excitations
        texc = tmpl.get("excitation", {})
        cexc = cfg.get("excitation", {})
        if cexc.get("type") and texc.get("type") and cexc["type"] != texc["type"]:
            rec.fail("07", "requested excitation %r but the template is registered for %r" % (cexc["type"], texc["type"]))
        present = excitation_names(app)
        miss = [n for n in texc.get("names", []) if n not in present]
        if miss:
            rec.fail("07", "excitations missing: %s (present: %s)" % (", ".join(miss), ", ".join(present)))
        incid = _incidences(cexc)
        angle_vars = texc.get("angle_variables", {}) or {}
        fixed = {"theta": float(texc.get("fixed_theta_deg", 0.0)), "phi": float(texc.get("fixed_phi_deg", 0.0))}
        need_theta = any(t != fixed["theta"] for t, _ in incid)
        need_phi = any(p != fixed["phi"] for _, p in incid)
        for key, need in (("theta", need_theta), ("phi", need_phi)):
            var = angle_vars.get(key)
            if var and var not in existing:
                rec.fail("07", "%s angle variable %r is not in the design" % (key, var))
            if need and not var:
                rec.fail("07", "requested %s angles %s need a registered %s variable" % (key, sorted(set(x[0 if key == 'theta' else 1] for x in incid)), key))
        pols = [str(p).upper() for p in cexc.get("polarizations", ["TE"])]
        pmode = (texc.get("polarization") or {}).get("mode", "floquet_modes")
        if pmode == "variable":
            pv_ = texc["polarization"]["variable"]
            if pv_ not in existing:
                rec.fail("07", "polarization variable %r is not in the design" % pv_)
            missing_p = [p for p in pols if p not in texc["polarization"].get("values", {})]
            if missing_p:
                rec.fail("07", "no polarization value registered for %s" % ", ".join(missing_p))
        meta["excitations"] = {"present": present, "by_type": excitation_names_by_type(app), "requested": cexc, "template": texc}
        rec.ok("07", "%s; %d incidence condition(s); polarizations %s via %s" % (", ".join(texc.get("names", [])) or "none registered", len(incid), pols, pmode))
        # ---------------- 08 boundaries
        bnames = boundary_names(app)
        missb = [b for b in tmpl.get("boundaries", []) if b not in bnames]
        if missb:
            rec.fail("08", "boundaries missing: %s (present: %s)" % (", ".join(missb), ", ".join(bnames)))
        meta["boundaries"] = bnames
        if tmpl.get("boundaries"):
            rec.ok("08", ", ".join(tmpl["boundaries"]))
        else:
            rec.skip("08", "no boundaries registered")
        # ---------------- 09 setup / sweep
        setup_name, sweep_name = tmpl["setup"], tmpl["sweep"]
        if setup_name not in setup_names(app):
            rec.fail("09", "setup %r not found (present: %s)" % (setup_name, setup_names(app)))
        setup = get_setup(app, setup_name)
        if sweep_name not in sweep_names(setup):
            rec.fail("09", "sweep %r not found in setup %r" % (sweep_name, setup_name))
        fr = cfg["frequency"]
        props = configure_sweep(setup, sweep_name, fr["start_hz"], fr["stop_hz"], fr["points"])
        try:
            got = (parse_freq_hz(props["RangeStart"]), parse_freq_hz(props["RangeEnd"]), int(props["RangeCount"]))
        except Exception as e:
            rec.fail("09", "sweep props unreadable after update: %s (%s)" % (props, e))
        if not (_close(got[0], fr["start_hz"]) and _close(got[1], fr["stop_hz"]) and got[2] == int(fr["points"])):
            rec.fail("09", "sweep reads back %s, requested %s" % (got, (fr["start_hz"], fr["stop_hz"], fr["points"])))
        meta["setup"] = _jsonable(getattr(setup, "props", {}))
        meta["sweep"] = props
        rec.ok("09", "%s : %s, %g-%g GHz, %d points" % (setup_name, sweep_name, fr["start_hz"] / 1e9, fr["stop_hz"] / 1e9, fr["points"]))
        # ---------------- 10 + 11 solve and extract per incidence condition
        outputs = cfg.get("outputs", ["sparameters"])
        tout = tmpl.get("outputs", {})
        for o in outputs:
            if o not in tout:
                rec.fail("07", "output %r requested but the template registers no extraction for it" % o)
        sp_rows, rcs_rows, ff_rows = [], [], []
        pol_passes = [[p] for p in pols] if pmode == "variable" else [pols]
        setup_sweep = "%s : %s" % (setup_name, sweep_name)
        n_solves = 0
        for theta, phi in incid:
            for pass_pols in pol_passes:
                if "theta" in angle_vars:
                    set_variable(app, angle_vars["theta"], _expression(theta, "deg"))
                if "phi" in angle_vars:
                    set_variable(app, angle_vars["phi"], _expression(phi, "deg"))
                if pmode == "variable":
                    set_variable(app, texc["polarization"]["variable"], texc["polarization"]["values"][pass_pols[0]])
                progress("SOLVING")
                t0 = time.time()
                log("solving theta=%g phi=%g pol=%s" % (theta, phi, pass_pols))
                try:
                    ok = analyze(app, setup_name, cores=cores)
                except Exception as e:
                    raise RuntimeError("solve failed at theta=%g phi=%g: %s" % (theta, phi, e))
                if not ok:
                    raise RuntimeError("solve failed at theta=%g phi=%g" % (theta, phi))
                meta["solves"].append({"theta_deg": theta, "phi_deg": phi, "pols": pass_pols, "runtime_s": round(time.time() - t0, 2)})
                n_solves += 1
                progress("EXTRACTING")
                if "sparameters" in outputs:
                    keys = [k for k in tout["sparameters"] if k.rsplit("_", 1)[-1].upper() in pass_pols]
                    if not keys:
                        rec.fail("11", "no S-parameter expressions registered for polarizations %s" % pass_pols)
                    sd = solution_data(app, [tout["sparameters"][k] for k in keys], setup_sweep)
                    f = sweep_values_hz(sd)
                    cols = {}
                    for k in keys:
                        re_, im_ = expression_complex(sd, tout["sparameters"][k])
                        if len(re_) != len(f) or len(im_) != len(f):
                            rec.fail("11", "%s has %d/%d values for %d frequencies" % (k, len(re_), len(im_), len(f)))
                        cols[k] = (re_, im_)
                    for i, fi in enumerate(f):
                        row = {"theta_inc": theta, "phi_inc": phi, "frequency": fi}
                        for k, (re_, im_) in cols.items():
                            mag = math.hypot(re_[i], im_[i])
                            row.update({k + "_re": re_[i], k + "_im": im_[i], k + "_mag": mag,
                                        k + "_phase_deg": math.degrees(math.atan2(im_[i], re_[i]))})
                        sp_rows.append(row)
                if "rcs" in outputs:
                    rc = tout["rcs"]
                    for pol in pass_pols:
                        expr = rc["expressions"].get(pol)
                        if not expr:
                            rec.fail("11", "no RCS expression registered for %s" % pol)
                        sd = solution_data(app, [expr], setup_sweep, rc.get("report_category"), rc.get("context"))
                        f = sweep_values_hz(sd)
                        vals = expression_real(sd, expr)
                        if len(vals) != len(f):
                            rec.fail("11", "RCS %s has %d values for %d frequencies" % (pol, len(vals), len(f)))
                        for fi, v in zip(f, vals):
                            m2 = 10 ** (v / 10.0) if rc.get("quantity", "m2").lower() == "dbsm" else v
                            rcs_rows.append({"pol": pol, "theta_inc": theta, "phi_inc": phi, "frequency": fi, "rcs_m2": m2,
                                             "rcs_dbsm": 10 * math.log10(m2) if m2 > 0 else float("-inf")})
                if "farfield" in outputs:
                    ffc = tout["farfield"]
                    for pol in pass_pols:
                        expr = ffc["expressions"].get(pol)
                        if not expr:
                            rec.fail("11", "no far-field expression registered for %s" % pol)
                        for fhz in ffc.get("frequencies_hz", []):
                            var = {"Freq": ["%.9gGHz" % (fhz / 1e9)], "Theta": ["All"], "Phi": ["All"]}
                            sd = solution_data(app, [expr], setup_sweep, ffc.get("report_category", "Far Fields"), ffc.get("context"),
                                               variations=var, primary_sweep="Theta")
                            th = [float(x) for x in sd.primary_sweep_values]
                            vals = expression_real(sd, expr)
                            for t_, v in zip(th, vals):
                                ff_rows.append({"frequency": fhz, "theta_inc": theta, "phi_inc": phi, "pol": pol, "quantity": expr,
                                                "theta": t_, "phi": ffc.get("phi_cut_deg", 0.0), "value": v})
        rec.ok("10", "%d solve(s)" % n_solves)
        # ---------------- 11 checks
        npts = int(cfg["frequency"]["points"])
        problems = []
        if "sparameters" in outputs:
            per = {}
            for r in sp_rows:
                per.setdefault((r["theta_inc"], r["phi_inc"]), []).append(r)
            for key, rows in per.items():
                if len(rows) != npts:
                    problems.append("S-parameters at %s have %d frequencies, requested %d (check the sweep type)" % (key, len(rows), npts))
            tol = float(cfg.get("checks", {}).get("passivity_tol", 0.05))
            for r in sp_rows:
                for k, v in r.items():
                    if isinstance(v, float) and not math.isfinite(v):
                        problems.append("non-finite %s at %g Hz" % (k, r["frequency"]))
                for pol in pols:
                    s11, s21 = r.get("S11_%s_mag" % pol), r.get("S21_%s_mag" % pol)
                    if s11 is not None:
                        tot = s11 ** 2 + (s21 ** 2 if s21 is not None else 0.0)
                        if tot > 1 + tol:
                            problems.append("non-passive %s: |S11|^2+|S21|^2 = %.3f at %.4g GHz" % (pol, tot, r["frequency"] / 1e9))
        if "rcs" in outputs:
            for r in rcs_rows:
                if not (math.isfinite(r["rcs_m2"]) and r["rcs_m2"] >= 0):
                    problems.append("invalid RCS %r at %g Hz" % (r["rcs_m2"], r["frequency"]))
        if problems:
            rec.fail("11", "; ".join(problems[:6]) + (" (+%d more)" % (len(problems) - 6) if len(problems) > 6 else ""))
        rec.ok("11", "%d S-parameter rows, %d RCS rows, %d far-field rows" % (len(sp_rows), len(rcs_rows), len(ff_rows)))
        # ---------------- write outputs
        for name, rows in (("sparameters.csv", sp_rows), ("rcs.csv", rcs_rows), ("farfield.csv", ff_rows)):
            if rows:
                with open(os.path.join(out_dir, name), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                files[name] = _sha256(os.path.join(out_dir, name))
        if keep_project:
            save_project(app)
    except ValidationFailure as e:
        state, error = "FAILED_VALIDATION", str(e)
    except Exception as e:
        state, error = "FAILED", "%s\n%s" % (e, traceback.format_exc())
        log("ERROR " + error)
    finally:
        if app is not None:
            try:
                release(app)
                released = True
            except Exception as e:
                log("release_desktop failed: %s" % e)
        if not keep_project:
            shutil.rmtree(out_dir.rstrip("/\\") + "_work", ignore_errors=True)
    # ---------------- 12 cleanup / package
    try:
        if state == "COMPLETED":
            if app is not None and not released:
                rec.rows["12"].update(status="FAIL", detail="AEDT not released")
                state, error = "FAILED", "AEDT could not be released"
            else:
                rec.rows["12"].update(status="PASS", detail="result package written; AEDT released")
        for r in rec.as_list():
            if r["status"] == "NOT_RUN":
                r["status"] = "SKIP"
                r["detail"] = r["detail"] or "not reached"
        runtime = round(time.time() - t_start, 2)
        meta.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "runtime_s": runtime, "compat_log": list(COMPAT_LOG),
                     "frequency": cfg.get("frequency"), "excitation": cfg.get("excitation"), "outputs": cfg.get("outputs"),
                     "extraction": tmpl.get("outputs"), "mesh": (meta.get("setup") or {}).get("MaximumPasses")})
        _write_json(os.path.join(out_dir, "validation.json"), {"job_id": job.get("job_id"), "state": state, "rows": rec.as_list()})
        _write_json(os.path.join(out_dir, "solver_metadata.json"), meta)
        result = {"protocol_version": job.get("protocol_version"), "job_id": job.get("job_id"), "candidate_id": job.get("candidate_id"),
                  "kind": job.get("kind", "candidate"), "status": state, "error": error, "simulation_type": "HFSS",
                  "fidelity": "VALIDATED_SIMULATION" if state == "COMPLETED" else None, "solver": "HFSS",
                  "versions": meta.get("versions"), "runtime_s": runtime, "files": files,
                  "frequency_unit": "Hz", "angle_unit": "deg", "rcs_unit": "m2"}
        _write_json(os.path.join(out_dir, "result.json"), result)
    finally:
        progress(state)
        log("finished: %s%s" % (state, (" - " + error.splitlines()[0]) if error else ""))
        log_f.close()
    return state, error


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Run one HFSS job package (generated by metasurface_optimizer).")
    ap.add_argument("--job", default=os.path.dirname(os.path.abspath(__file__)), help="job package folder (default: this script's folder)")
    ap.add_argument("--out", default=None, help="result folder (default: <job>/result_manual)")
    ap.add_argument("--graphical", action="store_true", help="show the AEDT window")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--keep-project", action="store_true", help="keep the solved .aedt copy in the result folder")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.job, "result_manual")
    state, _ = run(a.job, out, non_graphical=not a.graphical, cores=a.cores, keep_project=a.keep_project)
    return {"COMPLETED": 0, "FAILED_VALIDATION": 2}.get(state, 1)
# ---- END runner_core ----
