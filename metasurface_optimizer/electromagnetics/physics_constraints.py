"""Response-level physics checks combining the individual constraint modules.

Used (a) to audit any response (predicted or supplied) and (b) to report
how many predicted candidates violate physics in ablation studies.
"""
from __future__ import annotations

from typing import Dict

from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.physics.config import PhysicsConfig
from metasurface_optimizer.physics.passivity import passivity_violation
from metasurface_optimizer.physics.phase_consistency import phase_jumps
from metasurface_optimizer.physics.reciprocity import reciprocity_violation
from metasurface_optimizer.physics.spectral_smoothness import roughness


def check_response(resp: EMResponse, cfg: PhysicsConfig) -> Dict[str, object]:
    """Return a dict of violations for every applicable, enabled check."""
    out: Dict[str, object] = {"applied": []}
    s = resp.sparams
    if s is None:
        return out
    pols = {k[2] for k in s.data}
    for pol in sorted(pols):
        if cfg.passivity_active and s.has(1, 1, pol):
            out[f"passivity_{pol}"] = passivity_violation(s.get(1, 1, pol), s.data.get((2, 1, pol)))
            out["applied"].append(f"passivity_{pol}")
        if cfg.reciprocity_active and s.has(2, 1, pol) and s.has(1, 2, pol):
            out[f"reciprocity_{pol}"] = reciprocity_violation(s.get(2, 1, pol), s.get(1, 2, pol))
            out["applied"].append(f"reciprocity_{pol}")
        if s.has(1, 1, pol):
            out[f"phase_{pol}"] = phase_jumps(s.get(1, 1, pol))
            if cfg.use_smoothness:
                out[f"roughness_{pol}"] = roughness(s.get(1, 1, pol).real) + roughness(s.get(1, 1, pol).imag)
    out["any_violation"] = any(isinstance(v, dict) and v.get("violated") for v in out.values())
    return out
