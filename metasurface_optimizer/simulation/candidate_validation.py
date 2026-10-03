"""Checks a candidate before it may consume a high-fidelity simulation.

Bounds, geometry (overlaps, minimum gaps/features), manufacturing limits,
materials, frequency range and the simulation configuration are checked on
PC-1, so HFSS time is never spent on a design that is invalid on paper.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from metasurface_optimizer.geometry.parameter_schema import ParameterSchema


def validate_candidate(x: np.ndarray, schema: ParameterSchema, validator, template=None,
                       sim_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    x = np.asarray(x, float)
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, str] = {}
    # bounds
    lo, hi = schema.lower, schema.upper
    out = [f"{n}={v:g} outside [{a:g}, {b:g}]" for n, v, a, b in zip(schema.names, x, lo, hi) if not (a - 1e-9 <= v <= b + 1e-9)]
    if not np.all(np.isfinite(x)):
        out.append("non-finite parameter value")
    checks["bounds"] = "FAIL" if out else "PASS"
    errors += out
    # geometry + manufacturing (the schema validator covers overlaps, gaps, min features, thickness limits)
    try:
        rep = validator.validate(x)
        checks["geometry"] = "PASS" if rep.valid else "FAIL"
        errors += [f"geometry: {v}" for v in rep.violations]
        warnings += [f"geometry: {w}" for w in rep.warnings]
    except Exception as e:  # pragma: no cover - validator bug must not pass silently
        checks["geometry"] = "FAIL"
        errors.append(f"geometry validation crashed: {e}")
    # materials
    mat_err = []
    for spec in schema.parameters:
        v = x[schema.index(spec.name)]
        if spec.role == "permittivity" and v < 1.0:
            mat_err.append(f"{spec.name}={v:g} < 1 (non-physical permittivity)")
        if spec.role == "loss_tangent" and v < 0:
            mat_err.append(f"{spec.name}={v:g} < 0 (active material)")
    checks["material"] = "FAIL" if mat_err else "PASS"
    errors += mat_err
    # simulation configuration / template mapping
    if template is not None:
        miss = [s.name for s in schema.parameters if not s.fixed and s.name not in template.parameter_mapping]
        if miss:
            errors.append("no AEDT variable for free parameter(s): " + ", ".join(miss))
        if not template.verified and (sim_config or {}).get("simulation_type") == "HFSS":
            errors.append(f"template {template.template_id} is not verified (status {template.status()})")
        checks["simulation_config"] = "FAIL" if miss else "PASS"
    if sim_config is not None:
        fr = sim_config.get("frequency", {})
        if not (0 < fr.get("start_hz", 0) < fr.get("stop_hz", 0)) or int(fr.get("points", 0)) < 1:
            errors.append(f"invalid frequency range {fr}")
            checks["frequency"] = "FAIL"
        else:
            checks["frequency"] = "PASS"
    return {"valid": not errors, "errors": errors, "warnings": warnings, "checks": checks}
