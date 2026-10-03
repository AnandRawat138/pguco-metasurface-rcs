"""Geometry constraints g_i(x) <= 0 and generic polygon validation.

Two complementary layers:

* **Analytic constraints** supplied by each template (smooth functions of x,
  cheap, used inside optimisers), plus schema-level linear constraints.
* **Generic polygon checks** run on the constructed geometry: validity /
  self-intersection, overlap of neighbouring periodic copies (minimum gap),
  minimum feature size via morphological erosion, disconnected metal and
  substrate-thickness limits. These are used to reject designs before
  ranking and before export.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon

from metasurface_optimizer.geometry.geometry_generator import CellGeometry, GeometryGenerator
from metasurface_optimizer.geometry.parameter_schema import ParameterSchema


@dataclass
class ValidationReport:
    valid: bool
    constraint_values: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_violation(self) -> float:
        """Sum of positive constraint values (0 for a feasible design)."""
        return float(sum(max(0.0, v) for v in self.constraint_values.values()))

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "constraint_values": self.constraint_values,
            "violations": self.violations,
            "warnings": self.warnings,
            "total_violation": self.total_violation,
        }


def _min_feature_violation(poly: MultiPolygon, w_min: float) -> float:
    """Positive if some feature is narrower than w_min (opening test).

    The morphological opening ``buffer(-w/2).buffer(+w/2)`` removes every
    feature thinner than ``w``; the relative area removed is the violation.
    """
    if poly.is_empty or w_min <= 0:
        return -1.0
    opened = poly.buffer(-w_min / 2, join_style=2).buffer(w_min / 2, join_style=2)
    lost = poly.difference(opened).area / max(poly.area, 1e-12)
    return lost - 1e-3  # tolerate numerical corner losses


def _periodic_min_gap(poly: MultiPolygon, px: float, py: float) -> float:
    """Minimum distance between metal and its periodic neighbours and between disjoint parts."""
    if poly.is_empty:
        return np.inf
    d = np.inf
    for dx, dy in ((px, 0), (0, py), (px, py), (px, -py)):
        d = min(d, poly.distance(affinity.translate(poly, dx, dy)))
    parts = list(poly.geoms)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            d = min(d, parts[i].distance(parts[j]))
    return float(d)


class GeometryValidator:
    """Evaluates all geometry constraints of a design."""

    def __init__(self, schema: ParameterSchema, generator: Optional[GeometryGenerator] = None, generic_checks: bool = True):
        self.schema = schema
        self.generator = generator or (GeometryGenerator(schema) if schema.template else None)
        self.generic_checks = generic_checks
        self.mf = dict(schema.manufacturing)

    def constraint_values(self, x: np.ndarray, generic: Optional[bool] = None) -> Dict[str, float]:
        """All g_i(x); feasible iff every value <= 0."""
        generic = self.generic_checks if generic is None else generic
        g: Dict[str, float] = {}
        lo, hi = self.schema.lower, self.schema.upper
        g["bounds"] = float(np.max(np.maximum(lo - x, x - hi) / np.maximum(hi - lo, 1e-12)))
        g.update({f"linear:{k}": v for k, v in self.schema.linear_constraint_values(x).items()})
        if self.generator is not None:
            geom = self.generator.build(x)
            g.update({f"template:{k}": v for k, v in geom.analytic_constraints.items()})
            if generic:
                g.update(self._generic(geom))
        return g

    def _generic(self, geom: CellGeometry) -> Dict[str, float]:
        g: Dict[str, float] = {}
        for layer in geom.metal_layers:
            poly = layer.polygons
            g[f"{layer.name}:self_intersection"] = 0.0 if poly.is_valid else 1.0
            if poly.is_valid and len(poly.geoms) > 1:
                total = sum(p.area for p in poly.geoms)
                g[f"{layer.name}:overlap"] = (total - poly.area) / max(total, 1e-12) - 1e-9 if total - poly.area > 1e-9 else -1.0
            if "min_gap" in self.mf:
                gap = _periodic_min_gap(poly if poly.is_valid else poly.buffer(0), geom.period_x, geom.period_y)
                g[f"{layer.name}:periodic_gap"] = self.mf["min_gap"] - gap
            if "min_feature" in self.mf and poly.is_valid:
                g[f"{layer.name}:min_feature"] = _min_feature_violation(poly, self.mf["min_feature"])
            if geom.expected_metal_parts is not None and poly.is_valid:
                n = len(poly.geoms)
                g[f"{layer.name}:disconnected"] = float(n != geom.expected_metal_parts) - 0.5
            bx = poly.bounds if not poly.is_empty else (0, 0, 0, 0)
            g[f"{layer.name}:inside_cell"] = max(
                abs(bx[0]) - geom.period_x / 2, abs(bx[2]) - geom.period_x / 2,
                abs(bx[1]) - geom.period_y / 2, abs(bx[3]) - geom.period_y / 2,
            ) if not poly.is_empty else -1.0
        if "max_substrate_thickness" in self.mf:
            sub = [l for l in geom.layers if l.material == "dielectric"]
            if sub:
                g["max_substrate_thickness"] = sum(l.thickness for l in sub) - self.mf["max_substrate_thickness"]
        return g

    def validate(self, x: np.ndarray, generic: Optional[bool] = None, tol: float = 1e-9) -> ValidationReport:
        try:
            g = self.constraint_values(np.asarray(x, float), generic)
        except Exception as exc:  # geometry construction failure is itself a violation
            return ValidationReport(False, {"construction_error": 1.0}, [f"construction_error: {exc}"])
        viol = [k for k, v in g.items() if v > tol]
        return ValidationReport(not viol, g, viol)

    def is_valid(self, x: np.ndarray, generic: Optional[bool] = None) -> bool:
        return self.validate(x, generic).valid

    def penalty(self, x: np.ndarray, generic: bool = False) -> float:
        """J_constraint(x) = sum_i max(0, g_i(x))^2 (analytic constraints only by default, for speed)."""
        g = self.constraint_values(np.asarray(x, float), generic)
        return float(sum(max(0.0, v) ** 2 for v in g.values()))

    def filter_valid(self, X: np.ndarray, generic: Optional[bool] = None) -> np.ndarray:
        return np.array([self.is_valid(x, generic) for x in np.atleast_2d(X)], dtype=bool)
