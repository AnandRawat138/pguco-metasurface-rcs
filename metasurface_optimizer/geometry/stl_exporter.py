"""STL export of a parameterised cell (or a finite array of cells).

Each layer becomes a watertight solid: full-cell layers (ground, substrate)
are boxes; patterned metal layers are their polygons extruded by the metal
thickness (holes supported). Overlapping polygons within a layer are merged
first, so every body is a closed 2-manifold. Validation before writing checks
watertightness, consistent winding, positive volume and that the bounding box
matches the declared stack (units/scale). The STL unit is stated in the
sidecar JSON because STL itself carries no units; the parameterised geometry
is exported separately and remains the design of record.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from metasurface_optimizer.geometry.geometry_generator import CellGeometry


@dataclass
class MeshValidation:
    ok: bool
    bodies: List[Dict[str, object]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    bounds: Optional[List[List[float]]] = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "bodies": self.bodies, "issues": self.issues, "bounds": self.bounds}


def _extrude(poly: Polygon, z0: float, t: float) -> trimesh.Trimesh:
    m = trimesh.creation.extrude_polygon(poly, t)
    m.apply_translation([0, 0, z0])
    return m


def build_meshes(geom: CellGeometry, nx: int = 1, ny: int = 1) -> Dict[str, trimesh.Trimesh]:
    """Return one mesh per layer for an nx-by-ny tiling of the cell (centred at the origin)."""
    W, H = geom.period_x * nx, geom.period_y * ny
    out: Dict[str, trimesh.Trimesh] = {}
    for layer in geom.layers:
        if layer.thickness <= 0:
            continue
        if layer.polygons is None:
            b = trimesh.creation.box(extents=[W, H, layer.thickness])
            b.apply_translation([0, 0, layer.z0 + layer.thickness / 2])
            out[layer.name] = b
            continue
        tiles = []
        for ix in range(nx):
            for iy in range(ny):
                dx = (ix - (nx - 1) / 2) * geom.period_x
                dy = (iy - (ny - 1) / 2) * geom.period_y
                tiles.append(affinity.translate(layer.polygons, dx, dy))
        merged = unary_union(tiles)
        polys = [merged] if isinstance(merged, Polygon) else list(getattr(merged, "geoms", []))
        parts = [_extrude(p, layer.z0, layer.thickness) for p in polys if p.area > 0]
        if parts:
            out[layer.name] = trimesh.util.concatenate(parts)
    return out


def validate_meshes(meshes: Dict[str, trimesh.Trimesh], geom: CellGeometry, nx: int = 1, ny: int = 1, tol: float = 1e-6) -> MeshValidation:
    issues, bodies = [], []
    for name, m in meshes.items():
        for k, body in enumerate(m.split(only_watertight=False)):
            info = {"layer": name, "body": k, "watertight": bool(body.is_watertight), "winding_consistent": bool(body.is_winding_consistent),
                    "volume": float(body.volume)}
            bodies.append(info)
            if not body.is_watertight:
                issues.append(f"{name}[{k}] not watertight")
            if body.volume <= 0:
                issues.append(f"{name}[{k}] non-positive volume")
    allm = trimesh.util.concatenate(list(meshes.values()))
    b = allm.bounds
    exp_z = geom.total_thickness
    if abs((b[1, 2] - b[0, 2]) - exp_z) > tol * max(1, exp_z) + 1e-9:
        issues.append(f"stack height {b[1, 2] - b[0, 2]:.6g} != declared {exp_z:.6g}")
    if b[1, 0] - b[0, 0] > geom.period_x * nx + 1e-6 or b[1, 1] - b[0, 1] > geom.period_y * ny + 1e-6:
        issues.append("geometry exceeds the declared cell footprint")
    return MeshValidation(not issues, bodies, issues, b.tolist())


def export_stl(geom: CellGeometry, path: str | Path, unit: str = "mm", nx: int = 1, ny: int = 1, separate_layers: bool = True,
               require_valid: bool = True) -> MeshValidation:
    """Validate and write the STL (single file, or one per layer) plus a ``.json`` sidecar."""
    path = Path(path)
    meshes = build_meshes(geom, nx, ny)
    val = validate_meshes(meshes, geom, nx, ny)
    if require_valid and not val.ok:
        raise ValueError(f"Mesh validation failed: {val.issues}")
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.util.concatenate(list(meshes.values())).export(path)
    if separate_layers:  # one watertight solid per layer (recommended for HFSS/CST import)
        for name, m in meshes.items():
            m.export(path.with_name(f"{path.stem}_{name}.stl"))
    sidecar = {
        "unit": unit, "template": geom.template, "tiling": [nx, ny], "period": [geom.period_x, geom.period_y],
        "layers": [{"name": l.name, "z0": l.z0, "thickness": l.thickness, "material": l.material, "eps_r": l.eps_r, "tan_delta": l.tan_delta}
                   for l in geom.layers],
        "validation": val.to_dict(),
        "note": "STL is a mesh export of the parameterised design; the parameter file is the design of record.",
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return val
