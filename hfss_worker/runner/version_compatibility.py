# ---- BEGIN version_compatibility (inlined into run_hfss.py; standard library only) ----
"""PyAEDT / AEDT version-compatibility layer.

Every PyAEDT call made by run_hfss.py goes through this module so that API
differences between PyAEDT generations stay in one place:

* package name: ``ansys.aedt.core`` (PyAEDT >= 0.9) or ``pyaedt`` (older);
* ``Hfss(...)`` keywords: ``project/design/version/new_desktop`` (>= 0.9) vs
  ``projectname/designname/specified_version/new_desktop_session`` (< 0.9);
* ``SolutionData`` accessors: ``get_expression_data`` (newest) vs
  ``data_real/data_imag`` (0.9-0.14) and their legacy signatures;
* variable, excitation, boundary, setup and sweep accessors.

Probed against PyAEDT 0.12.0 (the release line paired with AEDT 2024 R2).
Anything not probed falls back defensively and records which path it took in
``COMPAT_LOG`` so the solver metadata shows exactly what was used.
"""
import importlib
import inspect
import platform
import sys

COMPAT_LOG = []


def _log(msg):
    COMPAT_LOG.append(msg)


def import_pyaedt():
    """Return (module, namespace, version_string)."""
    for name in ("ansys.aedt.core", "pyaedt"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        ver = getattr(mod, "__version__", None) or getattr(mod, "version", None) or "unknown"
        _log("pyaedt namespace: %s %s" % (name, ver))
        return mod, name, str(ver)
    raise ImportError("PyAEDT is not installed in this Python (tried ansys.aedt.core and pyaedt)")


def version_tuple(v):
    out = []
    for p in str(v).replace("-", ".").split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        if not num:
            break
        out.append(int(num))
    return tuple(out)


def normalize_aedt_version(v):
    """'2024.2', '2024 R2', '242', 'v242' -> '2024.2'."""
    s = str(v).strip().lower().replace("v", "").replace(" ", "")
    if "r" in s:
        y, r = s.split("r", 1)
        return "%s.%s" % (y, r)
    if s.isdigit() and len(s) == 3:
        return "20%s.%s" % (s[:2], s[2])
    return s


def _accepts(func, name):
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return True
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _kw(func, candidates):
    """Pick, for each canonical keyword, the spelling ``func`` accepts. candidates: {value_key: [spellings]}."""
    out = {}
    for spellings, value in candidates:
        for s in spellings:
            if _accepts(func, s):
                out[s] = value
                break
        else:
            _log("keyword %s not accepted by %s; omitted" % (spellings[0], getattr(func, "__name__", func)))
    return out


def open_hfss(mod, project, design, version, non_graphical=True, new_desktop=True, close_on_exit=True):
    hfss_cls = getattr(mod, "Hfss")
    kw = _kw(hfss_cls, [
        (("project", "projectname"), str(project)),
        (("design", "designname"), design),
        (("version", "specified_version"), version),
        (("non_graphical",), bool(non_graphical)),
        (("new_desktop", "new_desktop_session"), bool(new_desktop)),
        (("close_on_exit",), bool(close_on_exit)),
    ])
    _log("Hfss(%s)" % ", ".join(sorted(kw)))
    return hfss_cls(**kw)


def versions(app=None, pyaedt_namespace=None, pyaedt_version=None):
    aedt = None
    if app is not None:
        for attr in ("aedt_version_id", "_aedt_version"):
            aedt = getattr(app, attr, None)
            if aedt:
                break
        if not aedt:
            try:
                aedt = app.desktop_class.aedt_version_id
            except Exception:
                aedt = None
    return {
        "aedt_version": normalize_aedt_version(aedt) if aedt else None,
        "pyaedt_namespace": pyaedt_namespace,
        "pyaedt_version": pyaedt_version,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "os": platform.platform(),
        "machine": platform.node(),
    }


def solution_type(app):
    return getattr(app, "solution_type", None)


# ---------------------------------------------------------------- variables
def variable_names(app):
    vm = app.variable_manager
    names = []
    for attr in ("design_variable_names", "project_variable_names"):
        try:
            names += list(getattr(vm, attr))
        except Exception:
            pass
    if not names:
        try:
            names = list(vm.variable_names)
        except Exception:
            names = list(getattr(vm, "variables", {}).keys())
    return names


def set_variable(app, name, expression):
    vm = app.variable_manager
    try:
        ok = vm.set_variable(name, expression=expression)
    except TypeError:
        ok = vm.set_variable(name, expression)
    if ok is False:
        raise RuntimeError("AEDT refused to set %s = %s" % (name, expression))
    return True


def get_variable(app, name):
    """Return (expression_string, evaluated_si_value_or_None)."""
    vm = app.variable_manager
    var = None
    try:
        var = vm[name]
    except Exception:
        try:
            var = vm.variables[name]
        except Exception:
            var = None
    if var is None:
        return None, None
    expr = getattr(var, "expression", None) or getattr(var, "string_value", None)
    si = None
    for attr in ("si_value", "numeric_value"):
        try:
            si = float(getattr(var, attr))
            break
        except Exception:
            continue
    return expr, si


# ---------------------------------------------------------------- design objects
def _names(obj):
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [str(k) for k in obj.keys()]
    out = []
    for o in obj:
        out.append(str(getattr(o, "name", o)))
    return out


def excitation_names(app):
    for attr in ("excitations", "excitation_names", "design_excitations"):
        try:
            v = getattr(app, attr)
            names = _names(v() if callable(v) else v)
            if names:
                return names
        except Exception:
            continue
    try:
        return _names(app.get_all_sources())
    except Exception:
        return []


def excitation_names_by_type(app):
    try:
        return {str(k): _names(v) for k, v in dict(app.excitations_by_type).items()}
    except Exception:
        return {}


def boundary_names(app):
    try:
        return _names(app.boundaries)
    except Exception:
        return []


def material_exists(app, name):
    mats = getattr(app, "materials", None)
    if mats is None:
        return None
    for meth in ("exists_material", "checkifmaterialexists"):
        f = getattr(mats, meth, None)
        if f is not None:
            try:
                return bool(f(name))
            except Exception:
                continue
    try:
        return name.lower() in [m.lower() for m in mats.material_keys]
    except Exception:
        return None


def setup_names(app):
    try:
        return list(app.setup_names)
    except Exception:
        return _names(getattr(app, "setups", []))


def get_setup(app, name):
    s = app.get_setup(name)
    if s is None:
        raise KeyError("setup %s not found" % name)
    return s


def sweep_names(setup):
    for meth in ("get_sweep_names",):
        f = getattr(setup, meth, None)
        if f is not None:
            try:
                return list(f())
            except Exception:
                pass
    return _names(getattr(setup, "sweeps", []))


def get_sweep(setup, name):
    f = getattr(setup, "get_sweep", None)
    if f is not None:
        sw = f(name)
        if sw is not None:
            return sw
    for sw in getattr(setup, "sweeps", []):
        if getattr(sw, "name", None) == name:
            return sw
    raise KeyError("sweep %s not found" % name)


def _fmt_ghz(hz):
    return "%.9gGHz" % (float(hz) / 1e9)


def configure_sweep(setup, sweep_name, f_start_hz, f_stop_hz, n_points):
    """Apply a linear-count sweep and read it back. Returns the props that AEDT now holds."""
    sw = get_sweep(setup, sweep_name)
    props = sw.props
    props["RangeType"] = "LinearCount"
    props["RangeStart"] = _fmt_ghz(f_start_hz)
    props["RangeEnd"] = _fmt_ghz(f_stop_hz)
    props["RangeCount"] = int(n_points)
    ok = sw.update()
    if ok is False:
        raise RuntimeError("AEDT rejected the sweep update")
    sw = get_sweep(setup, sweep_name)
    p = sw.props
    return {k: p.get(k) for k in ("RangeType", "RangeStart", "RangeEnd", "RangeCount", "Type")}


def parse_freq_hz(s):
    s = str(s).strip()
    units = (("thz", 1e12), ("ghz", 1e9), ("mhz", 1e6), ("khz", 1e3), ("hz", 1.0))
    low = s.lower()
    for u, k in units:
        if low.endswith(u):
            return float(low[: -len(u)]) * k
    return float(s)


# ---------------------------------------------------------------- solve / post
def validate_design(app):
    f = getattr(app, "validate_simple", None)
    if f is None:
        _log("validate_simple not available")
        return None
    r = f()
    return bool(r)


def analyze(app, setup, cores=4, tasks=1, gpus=0):
    f = app.analyze_setup
    kw = _kw(f, [(("cores", "num_cores"), int(cores)), (("tasks", "num_tasks"), int(tasks)), (("gpus", "num_gpu"), int(gpus)),
                 (("blocking",), True)])
    r = f(setup, **kw)
    return r is not False


def solution_data(app, expressions, setup_sweep, report_category=None, context=None, variations=None, primary_sweep=None):
    f = app.post.get_solution_data
    kw = {"expressions": expressions, "setup_sweep_name": setup_sweep}
    if report_category:
        kw["report_category"] = report_category
    if context:
        kw["context"] = context
    if variations:
        kw["variations"] = variations
    if primary_sweep:
        kw["primary_sweep_variable"] = primary_sweep
    kw = {k: v for k, v in kw.items() if _accepts(f, k)}
    sd = f(**kw)
    if not sd:
        raise RuntimeError("no solution data for %s (%s)" % (expressions, setup_sweep))
    return sd


def _floats(v):
    return [float(x) for x in v]


def sweep_values_hz(sd):
    vals = None
    for attr in ("primary_sweep_values",):
        try:
            vals = list(getattr(sd, attr))
        except Exception:
            vals = None
    if vals is None:
        raise RuntimeError("SolutionData has no primary_sweep_values")
    unit = "GHz"
    try:
        units = sd.units_sweeps
        prim = getattr(sd, "primary_sweep", None) or "Freq"
        unit = units.get(prim, unit) or unit
    except Exception:
        pass
    scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}.get(str(unit).lower(), 1e9)
    return [float(v) * scale for v in vals]


def expression_complex(sd, expr):
    """(re, im) lists for ``expr``."""
    g = getattr(sd, "get_expression_data", None)
    if g is not None:
        try:
            _, re_ = g(expr, formula="real")
            _, im_ = g(expr, formula="imag")
            return _floats(re_), _floats(im_)
        except Exception as e:
            _log("get_expression_data failed (%s); trying data_real/data_imag" % e)
    try:
        return _floats(sd.data_real(expr)), _floats(sd.data_imag(expr))
    except TypeError:
        return _floats(sd.data_real(expr, False)), _floats(sd.data_imag(expr, False))


def expression_real(sd, expr):
    g = getattr(sd, "get_expression_data", None)
    if g is not None:
        try:
            _, v = g(expr, formula="real")
            return _floats(v)
        except Exception as e:
            _log("get_expression_data failed (%s); trying data_real" % e)
    try:
        return _floats(sd.data_real(expr))
    except TypeError:
        return _floats(sd.data_real(expr, False))


def save_project(app, path=None):
    f = app.save_project
    try:
        return f(file_name=path) if path else f()
    except TypeError:
        return f(path) if path else f()


def release(app):
    try:
        return app.release_desktop(close_projects=True, close_desktop=True)
    except TypeError:
        return app.release_desktop(True, True)
# ---- END version_compatibility ----
