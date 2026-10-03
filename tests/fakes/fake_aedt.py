"""A FAKE PyAEDT for testing the HFSS automation without AEDT.

It is NOT an electromagnetic solver. It reads a JSON "project" file (the fake
.aedt) describing variables, ports, boundaries, setups and sweeps, and answers
with a closed-form lossless resonator so the runner's control flow, checks and
file formats can be exercised. Both PyAEDT API generations are emulated:
``make_module(legacy=False)`` mimics ansys.aedt.core >= 0.9 keywords,
``legacy=True`` mimics pyaedt < 0.9 (projectname/designname/specified_version).
Tests only; never used for any result.
"""
from __future__ import annotations

import json
import os
import math
import types

UNITS = {"mm": 1e-3, "um": 1e-6, "m": 1.0, "deg": 1.0, "": 1.0, "ghz": 1e9}


def _parse(expr):
    s = str(expr).strip()
    for u in sorted(UNITS, key=len, reverse=True):
        if u and s.lower().endswith(u):
            return float(s[: -len(u)]) * UNITS[u]
    return float(s)


class _Var:
    def __init__(self, expr):
        self.expression = str(expr)
        try:
            self.si_value = _parse(expr)
        except ValueError:
            self.si_value = None


class _VM:
    def __init__(self, app, variables):
        self.app = app
        self.variables = {k: _Var(v) for k, v in variables.items()}

    @property
    def design_variable_names(self):
        return list(self.variables)

    @property
    def project_variable_names(self):
        return []

    def set_variable(self, name, expression=None, **kw):
        if name not in self.variables and not self.app.cfg.get("allow_new_variables", True):
            return False
        if self.app.cfg.get("refuse_variable") == name:
            return False
        self.variables[name] = _Var(expression)
        return True

    def __getitem__(self, k):
        return self.variables[k]


class _Sweep:
    def __init__(self, name, props, cfg):
        self.name, self.props, self.cfg = name, dict(props), cfg

    def update(self):
        if self.cfg.get("sweep_ignores_update"):
            self.props["RangeCount"] = 3
        return True


class _Setup:
    def __init__(self, name, d, cfg):
        self.name = name
        self.props = {"MaximumPasses": d.get("passes", 10), "Frequency": d.get("frequency", "10GHz")}
        self.sweeps = [_Sweep(n, p, cfg) for n, p in d.get("sweeps", {}).items()]

    def get_sweep_names(self):
        return [s.name for s in self.sweeps]

    def get_sweep(self, name):
        return next((s for s in self.sweeps if s.name == name), None)


class _SD:
    def __init__(self, app, expressions, setup_sweep, context):
        self.app = app
        sw = app._setups[setup_sweep.split(":")[0].strip()].get_sweep(setup_sweep.split(":")[1].strip())
        p = sw.props
        n = int(p["RangeCount"])
        a, b = _parse(p["RangeStart"]), _parse(p["RangeEnd"])
        self.f = [a + (b - a) * i / (n - 1) if n > 1 else a for i in range(n)]
        self.primary_sweep_values = [f / 1e9 for f in self.f]
        self.units_sweeps = {"Freq": "GHz"}
        self.expressions = expressions

    def _resp(self, expr):
        v = self.app._vm.variables
        L = v.get("L", _Var("8mm")).si_value or 8e-3
        th = math.radians(v.get("scan_theta", _Var("0deg")).si_value or 0.0)
        f0 = 1.2e8 / L * (1 + 0.05 * math.sin(th) ** 2)
        Q = 6.0
        out = []
        for f in self.f:
            x = Q * (f / f0 - f0 / f)
            s11 = 1 / complex(1, x)
            s21 = complex(0, x) / complex(1, x)
            if self.app.cfg.get("nonpassive"):
                s11 *= 1.5
            if "S(FloquetPort2" in expr or "S(2" in expr:
                out.append(s21)
            elif "RCS" in expr:
                lam = 3e8 / f
                out.append(complex(4 * math.pi * 0.01 ** 2 / lam ** 2 * abs(s11) ** 2, 0))
            else:
                out.append(s11)
        if self.app.cfg.get("short_output"):
            out = out[:-1]
        return out

    def data_real(self, expression=None, convert_to_SI=False):
        return [z.real for z in self._resp(expression)]

    def data_imag(self, expression=None, convert_to_SI=False):
        return [z.imag for z in self._resp(expression)]


class _Post:
    def __init__(self, app):
        self.app = app

    def get_solution_data(self, expressions=None, setup_sweep_name=None, domain=None, variations=None, primary_sweep_variable=None,
                          report_category=None, context=None, **kw):
        self.app.calls.append(("get_solution_data", expressions, setup_sweep_name, report_category))
        if not self.app._solved:
            return False
        return _SD(self.app, expressions, setup_sweep_name, context)


class _Materials:
    def __init__(self, names):
        self.names = [n.lower() for n in names]

    def exists_material(self, name):
        return name.lower() in self.names


class _Bnd:
    def __init__(self, name):
        self.name = name


def _make_app_class(legacy: bool):
    class Hfss:
        def _init(self, project, design, version):
            with open(project) as f:
                self.cfg = json.load(f)
            if design != self.cfg["design"]:
                raise RuntimeError(f"design {design} not in project")
            self.design_name = design
            self.solution_type = self.cfg.get("solution_type", "Modal")
            self.aedt_version_id = self.cfg.get("aedt_version", "2024.2")
            self._vm = _VM(self, self.cfg["variables"])
            self.excitations = list(self.cfg.get("excitations", []))
            self.boundaries = [_Bnd(b) for b in self.cfg.get("boundaries", [])]
            self._setups = {n: _Setup(n, d, self.cfg) for n, d in self.cfg.get("setups", {}).items()}
            self.materials = _Materials(self.cfg.get("materials", []))
            self.post = _Post(self)
            self._solved = False
            self.calls = []
            self.released = False

        @property
        def variable_manager(self):
            return self._vm

        @property
        def setup_names(self):
            return list(self._setups)

        def get_setup(self, name):
            return self._setups.get(name)

        def validate_simple(self, log_file=None):
            return 0 if self.cfg.get("invalid_geometry") else 1

        def analyze_setup(self, name=None, cores=4, tasks=1, gpus=0, blocking=True, **kw):
            self.calls.append(("analyze", name, cores))
            if self.cfg.get("solve_fails"):
                return False
            self._solved = True
            return True

        def save_project(self, file_name=None, overwrite=True, refresh_ids=False):
            return True

        def release_desktop(self, close_projects=True, close_desktop=True):
            self.released = True
            return True

    if legacy:
        def __init__(self, projectname=None, designname=None, solution_type=None, setup_name=None, specified_version=None,
                     non_graphical=False, new_desktop_session=False, close_on_exit=False, student_version=False):
            self._init(projectname, designname, specified_version)
    else:
        def __init__(self, project=None, design=None, solution_type=None, setup=None, version=None, non_graphical=False,
                     new_desktop=False, close_on_exit=False, student_version=False, machine="", port=0, aedt_process_id=None,
                     remove_lock=False):
            self._init(project, design, version)
    Hfss.__init__ = __init__
    return Hfss


def make_module(legacy: bool = False):
    m = types.ModuleType("pyaedt" if legacy else "ansys.aedt.core")
    m.__version__ = "0.8.11" if legacy else "0.12.0"
    m.Hfss = _make_app_class(legacy)
    return m


def fake_project(path, **overrides):
    """Write a fake .aedt (JSON) describing an FSS unit-cell template."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    proj = {"design": "HFSSDesign1", "solution_type": "Modal", "aedt_version": "2024.2",
            "variables": {"p": "10mm", "L": "8mm", "W": "1mm", "scan_theta": "0deg", "scan_phi": "0deg"},
            "excitations": ["FloquetPort1", "FloquetPort2"], "boundaries": ["Master1", "Slave1", "Master2", "Slave2"],
            "materials": ["vacuum", "Rogers RO4003 (tm)"],
            "setups": {"Setup1": {"sweeps": {"Sweep1": {"RangeType": "LinearCount", "RangeStart": "8GHz", "RangeEnd": "12GHz", "RangeCount": 11, "Type": "Interpolating"}}}}}
    proj.update(overrides)
    with open(path, "w") as f:
        json.dump(proj, f)
    return path


def fss_template_dict(aedt_path, sha256=None):
    return {"template_id": "fake_fss", "name": "FAKE unit cell (tests only)", "simulation_type": "periodic_fss",
            "aedt_project": str(aedt_path), "design": "HFSSDesign1", "solution_type": "Modal", "setup": "Setup1", "sweep": "Sweep1",
            "parameter_mapping": {"period": {"variable": "p", "unit": "mm"}, "length_a": {"variable": "L", "unit": "mm"},
                                  "width_a": {"variable": "W", "unit": "mm"}},
            "excitation": {"names": ["FloquetPort1", "FloquetPort2"], "angle_variables": {"theta": "scan_theta", "phi": "scan_phi"},
                           "polarization": {"mode": "floquet_modes"}},
            "boundaries": ["Master1", "Slave1"], "materials": ["Rogers RO4003 (tm)"],
            "outputs": {"sparameters": {"S11_TE": "S(FloquetPort1:1,FloquetPort1:1)", "S21_TE": "S(FloquetPort2:1,FloquetPort1:1)",
                                        "S11_TM": "S(FloquetPort1:2,FloquetPort1:2)", "S21_TM": "S(FloquetPort2:2,FloquetPort1:2)"}},
            "sha256": sha256}
