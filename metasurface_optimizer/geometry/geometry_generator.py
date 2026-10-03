"""Parameterised unit-cell geometry templates.

A template turns a dict of physical parameter values into a
:class:`CellGeometry`: in-plane metal polygons (shapely) plus a layer stack.
The geometry is the single source of truth for raster masks, constraint
checks and STL export, and is kept separate from any mesh.

Templates read parameters by *template variable name*. A schema may map its
own parameter names onto template variables through
``schema.metadata['template_map'] = {template_var: schema_param}``.
Missing optional variables fall back to the defaults listed in each template.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from metasurface_optimizer.geometry.parameter_schema import ParameterSchema


@dataclass
class Layer:
    """One layer of the stack. ``polygons`` is None for a full-cell layer."""

    name: str
    z0: float
    thickness: float
    material: str
    polygons: Optional[MultiPolygon] = None
    eps_r: Optional[float] = None
    tan_delta: Optional[float] = None


@dataclass
class CellGeometry:
    """Geometry of one periodic cell (or super-cell), lengths in schema units."""

    period_x: float
    period_y: float
    layers: List[Layer]
    template: str
    expected_metal_parts: Optional[int] = None
    analytic_constraints: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def metal_layers(self) -> List[Layer]:
        return [l for l in self.layers if l.polygons is not None]

    @property
    def total_thickness(self) -> float:
        return max(l.z0 + l.thickness for l in self.layers)

    def top_metal(self) -> MultiPolygon:
        ml = self.metal_layers
        return ml[-1].polygons if ml else MultiPolygon()


def _as_multi(g) -> MultiPolygon:
    if g.is_empty:
        return MultiPolygon()
    if isinstance(g, Polygon):
        return MultiPolygon([g])
    if isinstance(g, MultiPolygon):
        return g
    polys = [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]
    return MultiPolygon(polys)


def centered_rect(lx: float, ly: float, cx: float = 0.0, cy: float = 0.0, angle_deg: float = 0.0) -> Polygon:
    r = box(cx - lx / 2, cy - ly / 2, cx + lx / 2, cy + ly / 2)
    return affinity.rotate(r, angle_deg, origin=(cx, cy)) if angle_deg else r


def rotated_extent(lx: float, ly: float, angle_deg: float) -> tuple[float, float]:
    """Axis-aligned bounding-box extents of an lx-by-ly rectangle rotated by angle."""
    a = math.radians(angle_deg)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return lx * c + ly * s, lx * s + ly * c


# ---------------------------------------------------------------- templates
TemplateFn = Callable[[Mapping[str, float], Mapping[str, float]], CellGeometry]
_TEMPLATES: Dict[str, Dict[str, Any]] = {}


def register_template(name: str, variables: Dict[str, Optional[float]], description: str = ""):
    """Decorator registering a template. ``variables`` maps var -> default (None = required)."""

    def deco(fn: TemplateFn) -> TemplateFn:
        _TEMPLATES[name] = {"fn": fn, "variables": variables, "description": description}
        return fn

    return deco


def available_templates() -> Dict[str, Dict[str, Any]]:
    return {k: {"variables": v["variables"], "description": v["description"]} for k, v in _TEMPLATES.items()}


def _stack(period_x, period_y, h_sub, eps, tand, t_metal, top: MultiPolygon, ground: bool) -> List[Layer]:
    layers: List[Layer] = []
    z = 0.0
    if ground:
        layers.append(Layer("ground", z, t_metal, "metal"))
        z += t_metal
    layers.append(Layer("substrate", z, h_sub, "dielectric", eps_r=eps, tan_delta=tand))
    z += h_sub
    layers.append(Layer("top_metal", z, t_metal, "metal", polygons=top))
    return layers


_COMMON = {"substrate_thickness": None, "eps_r": 4.3, "tan_delta": 0.02, "metal_thickness": 0.035, "ground": 1.0}


@register_template(
    "rect_patch",
    {"period": None, "patch_length": None, "patch_width": None, "rotation": 0.0, **_COMMON},
    "Rectangular (optionally rotated) metal patch on a grounded or free-standing substrate.",
)
def _rect_patch(v: Mapping[str, float], mf: Mapping[str, float]) -> CellGeometry:
    P = v["period"]
    patch = centered_rect(v["patch_length"], v["patch_width"], angle_deg=v["rotation"])
    ex, ey = rotated_extent(v["patch_length"], v["patch_width"], v["rotation"])
    g_min = mf.get("min_gap", 0.1)
    w_min = mf.get("min_feature", 0.1)
    cons = {
        "gap_x": g_min - (P - ex),
        "gap_y": g_min - (P - ey),
        "min_width": w_min - min(v["patch_length"], v["patch_width"]),
    }
    top = _as_multi(patch)
    return CellGeometry(P, P, _stack(P, P, v["substrate_thickness"], v["eps_r"], v["tan_delta"], v["metal_thickness"], top, v["ground"] > 0.5), "rect_patch", 1, cons)


@register_template(
    "checkerboard_rect_patch",
    {
        "period": None, "length_a": None, "width_a": None, "length_b": None, "width_b": None,
        "rotation": 0.0, **_COMMON,
    },
    "2x2 coding super-cell (A B / B A) of rectangular patches: the classic diffusion/RCS-reduction checkerboard.",
)
def _checker(v: Mapping[str, float], mf: Mapping[str, float]) -> CellGeometry:
    P = v["period"]
    polys = []
    cons: Dict[str, float] = {}
    g_min = mf.get("min_gap", 0.1)
    w_min = mf.get("min_feature", 0.1)
    for state, (ix, iy) in (("a", (0, 0)), ("b", (1, 0)), ("b", (0, 1)), ("a", (1, 1))):
        L, W = v[f"length_{state}"], v[f"width_{state}"]
        cx, cy = (ix - 0.5) * P, (iy - 0.5) * P
        polys.append(centered_rect(L, W, cx, cy, v["rotation"]))
    for state in ("a", "b"):
        L, W = v[f"length_{state}"], v[f"width_{state}"]
        ex, ey = rotated_extent(L, W, v["rotation"])
        cons[f"gap_x_{state}"] = g_min - (P - ex)
        cons[f"gap_y_{state}"] = g_min - (P - ey)
        cons[f"min_width_{state}"] = w_min - min(L, W)
    top = MultiPolygon(polys)  # validity (no overlap) is checked by GeometryValidator
    return CellGeometry(2 * P, 2 * P, _stack(2 * P, 2 * P, v["substrate_thickness"], v["eps_r"], v["tan_delta"], v["metal_thickness"], top, v["ground"] > 0.5), "checkerboard_rect_patch", 4, cons, {"cell_period": P})


@register_template(
    "square_loop",
    {"period": None, "loop_length": None, "trace_width": None, "rotation": 0.0, **_COMMON},
    "Square metal loop (ring) FSS element.",
)
def _loop(v: Mapping[str, float], mf: Mapping[str, float]) -> CellGeometry:
    P, L, w = v["period"], v["loop_length"], v["trace_width"]
    outer = centered_rect(L, L)
    inner = centered_rect(max(L - 2 * w, 1e-6), max(L - 2 * w, 1e-6))
    ring = affinity.rotate(outer.difference(inner), v["rotation"], origin=(0, 0))
    ex, _ = rotated_extent(L, L, v["rotation"])
    cons = {
        "gap": mf.get("min_gap", 0.1) - (P - ex),
        "min_trace": mf.get("min_feature", 0.1) - w,
        "loop_open": mf.get("min_feature", 0.1) - (L - 2 * w),
    }
    return CellGeometry(P, P, _stack(P, P, v["substrate_thickness"], v["eps_r"], v["tan_delta"], v["metal_thickness"], _as_multi(ring), v["ground"] > 0.5), "square_loop", 1, cons)


@register_template(
    "slotted_patch",
    {"period": None, "patch_length": None, "patch_width": None, "slot_length": None, "slot_width": None, "rotation": 0.0, **_COMMON},
    "Rectangular patch with a central rectangular slot (aperture).",
)
def _slotted(v: Mapping[str, float], mf: Mapping[str, float]) -> CellGeometry:
    P = v["period"]
    patch = centered_rect(v["patch_length"], v["patch_width"])
    slot = centered_rect(v["slot_length"], v["slot_width"])
    shape = affinity.rotate(patch.difference(slot), v["rotation"], origin=(0, 0))
    ex, ey = rotated_extent(v["patch_length"], v["patch_width"], v["rotation"])
    w_min = mf.get("min_feature", 0.1)
    cons = {
        "gap_x": mf.get("min_gap", 0.1) - (P - ex),
        "gap_y": mf.get("min_gap", 0.1) - (P - ey),
        "slot_margin_x": w_min - 0.5 * (v["patch_length"] - v["slot_length"]),
        "slot_margin_y": w_min - 0.5 * (v["patch_width"] - v["slot_width"]),
        "min_slot_width": w_min - v["slot_width"],
    }
    return CellGeometry(P, P, _stack(P, P, v["substrate_thickness"], v["eps_r"], v["tan_delta"], v["metal_thickness"], _as_multi(shape), v["ground"] > 0.5), "slotted_patch", 1, cons)


@register_template(
    "cross_dipole",
    {"period": None, "arm_length": None, "arm_width": None, "rotation": 0.0, **_COMMON},
    "Cross-shaped dipole FSS element.",
)
def _cross(v: Mapping[str, float], mf: Mapping[str, float]) -> CellGeometry:
    P, L, w = v["period"], v["arm_length"], v["arm_width"]
    cross = unary_union([centered_rect(L, w), centered_rect(w, L)])
    cross = affinity.rotate(cross, v["rotation"], origin=(0, 0))
    ex, _ = rotated_extent(L, w, v["rotation"])
    ex = max(ex, rotated_extent(w, L, v["rotation"])[0])
    cons = {"gap": mf.get("min_gap", 0.1) - (P - ex), "min_width": mf.get("min_feature", 0.1) - w}
    return CellGeometry(P, P, _stack(P, P, v["substrate_thickness"], v["eps_r"], v["tan_delta"], v["metal_thickness"], _as_multi(cross), v["ground"] > 0.5), "cross_dipole", 1, cons)


# ---------------------------------------------------------------- facade
class GeometryGenerator:
    """Builds :class:`CellGeometry` objects for designs of a given schema."""

    def __init__(self, schema: ParameterSchema, template: Optional[str] = None):
        self.schema = schema
        self.template = template or schema.template
        if self.template not in _TEMPLATES:
            raise ValueError(f"Unknown or missing geometry template {self.template!r}; available: {sorted(_TEMPLATES)}")
        self.template_map: Dict[str, str] = dict(schema.metadata.get("template_map", {}))
        self.manufacturing = dict(schema.manufacturing)

    def template_values(self, values: Mapping[str, Any]) -> Dict[str, float]:
        spec = _TEMPLATES[self.template]["variables"]
        out: Dict[str, float] = {}
        for var, default in spec.items():
            key = self.template_map.get(var, var)
            if key in values:
                out[var] = float(values[key])
            elif var == "metal_thickness" and "metal_thickness" in self.manufacturing:
                out[var] = float(self.manufacturing["metal_thickness"])
            elif default is not None:
                out[var] = float(default)
            else:
                raise KeyError(f"Template {self.template!r} requires {var!r} (map it via metadata.template_map)")
        return out

    def build(self, x: np.ndarray | Mapping[str, Any]) -> CellGeometry:
        values = x if isinstance(x, Mapping) else self.schema.to_dict_values(x)
        tv = self.template_values(values)
        geom = _TEMPLATES[self.template]["fn"](tv, self.manufacturing)
        geom.meta["template_values"] = tv
        if "max_substrate_thickness" in self.manufacturing:
            geom.analytic_constraints["max_substrate_thickness"] = tv["substrate_thickness"] - self.manufacturing["max_substrate_thickness"]
        return geom

    def rasterize(self, x: np.ndarray | Mapping[str, Any], resolution: int = 64) -> np.ndarray:
        return rasterize_geometry(self.build(x), resolution)


def rasterize_geometry(geom: CellGeometry, resolution: int = 64) -> np.ndarray:
    """Binary mask (resolution x resolution) of the top metal layer over one cell; row 0 = +y."""
    import shapely

    xs = (np.arange(resolution) + 0.5) / resolution * geom.period_x - geom.period_x / 2
    ys = (np.arange(resolution) + 0.5) / resolution * geom.period_y - geom.period_y / 2
    X, Y = np.meshgrid(xs, ys[::-1])
    top = geom.top_metal()
    if top.is_empty:
        return np.zeros((resolution, resolution), dtype=bool)
    return shapely.contains_xy(top, X, Y)
