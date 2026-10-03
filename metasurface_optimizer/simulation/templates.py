"""Registry of verified HFSS template projects.

A template is an existing ``.aedt`` project on the HFSS PC whose geometry,
materials, boundaries, excitations, setup and sweep were built and checked by
a person. The framework only changes the variables named in
``parameter_mapping``. Two kinds are supported:

* ``periodic_fss``: unit cell with master/slave (or lattice-pair) boundaries and
  Floquet ports; S-parameters per TE/TM Floquet mode; scan angle via variables.
* ``finite_rcs``: finite array with a plane-wave excitation and radiation/PML
  boundary; monostatic RCS (and far-field cuts) per polarization and angle.

A template is *usable* only after the 12-step validation suite passed on the
HFSS PC with the same file hash (``verified``).
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hfss_worker.protocol import check_id, now_iso, write_json_atomic
from hfss_worker.validator import suite_passed

KINDS = {"periodic_fss": "floquet", "finite_rcs": "plane_wave"}
_VAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


@dataclass
class TemplateSpec:
    template_id: str
    name: str
    simulation_type: str                       # periodic_fss | finite_rcs
    aedt_project: str                          # absolute path ON THE HFSS PC
    design: str
    setup: str
    sweep: str
    parameter_mapping: Dict[str, Dict[str, Any]]  # schema parameter -> {variable, unit}
    excitation: Dict[str, Any] = field(default_factory=dict)
    boundaries: List[str] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    solution_type: Optional[str] = None
    sha256: Optional[str] = None               # of the .aedt file, computed on the HFSS PC at verification
    rcs_reference: Optional[Dict[str, Any]] = None  # e.g. {"kind": "pec_plate", "area_m2": 0.0144}
    verification: Dict[str, Any] = field(default_factory=dict)
    baseline_reproduction: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    registered: str = field(default_factory=now_iso)

    # ------------------------------------------------------------ checks
    def problems(self, schema=None) -> List[str]:
        p: List[str] = []
        try:
            check_id(self.template_id)
        except ValueError as e:
            p.append(str(e))
        if self.simulation_type not in KINDS:
            p.append(f"simulation_type must be one of {sorted(KINDS)}")
        if not str(self.aedt_project).lower().endswith((".aedt", ".aedtz")):
            p.append("aedt_project must be a .aedt or .aedtz file")
        for k in ("design", "setup", "sweep"):
            if not getattr(self, k):
                p.append(f"{k} is required")
        exc = self.excitation or {}
        if exc.get("type") and exc["type"] != KINDS.get(self.simulation_type):
            p.append(f"excitation type {exc['type']!r} does not fit a {self.simulation_type} template")
        if not exc.get("names"):
            p.append("excitation.names: list the port / incident-wave names the template defines")
        for k, v in (exc.get("angle_variables") or {}).items():
            if k not in ("theta", "phi") or not _VAR.match(str(v)):
                p.append(f"angle variable {k}={v!r} invalid")
        pol = exc.get("polarization") or {}
        if pol.get("mode", "floquet_modes") not in ("floquet_modes", "separate_excitations", "variable"):
            p.append("polarization.mode must be floquet_modes, separate_excitations or variable")
        if pol.get("mode") == "variable" and not (pol.get("variable") and pol.get("values")):
            p.append("polarization mode 'variable' needs variable and values {TE: ..., TM: ...}")
        if not self.parameter_mapping:
            p.append("parameter_mapping is empty")
        for name, m in self.parameter_mapping.items():
            if not _VAR.match(str(m.get("variable", ""))):
                p.append(f"{name}: invalid AEDT variable name {m.get('variable')!r}")
            if schema is not None and name not in schema.names:
                p.append(f"{name} is not a parameter of schema {schema.name}")
        if schema is not None:
            for spec in schema.parameters:
                if not spec.fixed and spec.name not in self.parameter_mapping:
                    p.append(f"free parameter {spec.name} has no AEDT variable mapping")
        outs = self.outputs or {}
        if not outs:
            p.append("outputs: register at least one extraction (sparameters or rcs)")
        if "sparameters" in outs:
            for key, expr in outs["sparameters"].items():
                if not re.match(r"^S\d\d_(TE|TM)$", key):
                    p.append(f"S-parameter key {key!r} must look like S11_TE")
                if not isinstance(expr, str) or not expr:
                    p.append(f"S-parameter {key} needs an AEDT expression")
        if "rcs" in outs and not (outs["rcs"].get("expressions")):
            p.append("rcs.expressions {TE: ..., TM: ...} required")
        return p

    @property
    def verified(self) -> bool:
        v = self.verification or {}
        return bool(v.get("passed")) and bool(self.sha256) and v.get("sha256") == self.sha256

    def status(self) -> str:
        if self.verified:
            return "VERIFIED"
        if self.verification:
            return "VERIFICATION_FAILED" if not self.verification.get("passed") else "STALE (file changed)"
        return "UNVERIFIED"

    def worker_json(self) -> Dict[str, Any]:
        """The registration as the HFSS PC sees it (template.json in each job)."""
        d = asdict(self)
        d["excitation"] = {"type": KINDS[self.simulation_type], **(self.excitation or {})}
        for k in ("verification", "baseline_reproduction", "registered", "name", "notes"):
            d.pop(k, None)
        return d

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verified"] = self.verified
        d["status"] = self.status()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemplateSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: copy.deepcopy(v) for k, v in d.items() if k in known})


class TemplateRegistry:
    """JSON-backed registry (``<root>/registry.json``)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "registry.json"
        self.templates: Dict[str, TemplateSpec] = {}
        if self.path.is_file():
            for d in json.loads(self.path.read_text()).get("templates", []):
                t = TemplateSpec.from_dict(d)
                self.templates[t.template_id] = t

    def save(self) -> None:
        write_json_atomic(self.path, {"templates": [t.to_dict() for t in self.templates.values()]})

    def register(self, spec: TemplateSpec, schema=None, replace: bool = False) -> TemplateSpec:
        probs = spec.problems(schema)
        if probs:
            raise ValueError("template registration invalid: " + "; ".join(probs))
        if spec.template_id in self.templates and not replace:
            raise ValueError(f"template {spec.template_id} already registered")
        if spec.template_id in self.templates:
            old = self.templates[spec.template_id]
            if (old.aedt_project, old.sha256) == (spec.aedt_project, spec.sha256):
                spec.verification = spec.verification or old.verification
        self.templates[spec.template_id] = spec
        self.save()
        return spec

    def get(self, template_id: str) -> TemplateSpec:
        if template_id not in self.templates:
            raise KeyError(f"template {template_id} is not registered")
        return self.templates[template_id]

    def list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.templates.values()]

    def record_verification(self, template_id: str, result_dir: Path) -> TemplateSpec:
        """Import a verify_template result package (from the HFSS PC) and mark the template."""
        t = self.get(template_id)
        rd = Path(result_dir)
        res = json.loads((rd / "result.json").read_text())
        rows = json.loads((rd / "validation.json").read_text()).get("rows", []) if (rd / "validation.json").is_file() else []
        meta = json.loads((rd / "solver_metadata.json").read_text()) if (rd / "solver_metadata.json").is_file() else {}
        if res.get("simulation_type") != "HFSS":
            raise ValueError(f"a {res.get('simulation_type')} run cannot verify an HFSS template")
        sha = meta.get("template_sha256") or t.sha256
        if t.sha256 and sha != t.sha256:
            raise ValueError("verification ran against a different template file hash")
        t.sha256 = sha
        t.verification = {"passed": res.get("status") == "COMPLETED" and suite_passed(rows), "rows": rows, "sha256": sha,
                          "versions": res.get("versions"), "date": now_iso(), "job_id": res.get("job_id"), "state": res.get("status")}
        self.save()
        return t
