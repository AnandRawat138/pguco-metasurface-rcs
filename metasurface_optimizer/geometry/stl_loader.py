"""STL import with *non-unique*, confidence-rated parameter inference.

An STL is a triangle mesh, not a parameterised electromagnetic design. This
loader therefore:

1. imports and preserves the original file (copy + SHA-256),
2. reports mesh health (watertight, bounds, units are unknown unless stated),
3. detects layers from the z-distribution of horizontal faces,
4. slices the top metal layer into a 2-D cross-section and rasterises it,
5. optionally fits a parameterised template to that mask (via
   :mod:`image_parser`) and reports the fit quality as a confidence level,
6. never claims the inferred parameters are the original design; the user
   must confirm or map parameters manually.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh

from metasurface_optimizer.core.provenance import Confidence


@dataclass
class STLAnalysis:
    path: str
    sha256: str
    watertight: bool
    n_faces: int
    bounds: List[List[float]]
    layers_z: List[float]
    layer_thicknesses: List[float]
    top_mask: Optional[np.ndarray] = None
    mask_extent: Optional[List[float]] = None
    inferred: Dict[str, object] = field(default_factory=dict)
    confidence: Confidence = Confidence.LOW
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "watertight": self.watertight, "n_faces": self.n_faces, "bounds": self.bounds,
                "layers_z": self.layers_z, "layer_thicknesses": self.layer_thicknesses, "inferred": self.inferred,
                "confidence": self.confidence.value, "warnings": self.warnings,
                "statement": "Inferred parameters are an approximate, non-unique interpretation of a mesh; verify before use."}


def load_stl(path: str | Path, preserve_dir: Optional[str | Path] = None, resolution: int = 128, unit_hint: Optional[str] = None) -> STLAnalysis:
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if preserve_dir:
        Path(preserve_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, Path(preserve_dir) / f"original_{sha[:12]}_{path.name}")
    mesh = trimesh.load(path, force="mesh")
    warnings = []
    if unit_hint is None:
        warnings.append("STL carries no units; scale must be confirmed by the user")
    bodies = mesh.split(only_watertight=False)
    watertight = bool(all(b.is_watertight for b in bodies))
    # layer interfaces: z of horizontal faces
    n = mesh.face_normals
    horiz = np.abs(n[:, 2]) > 0.99
    zs = mesh.triangles_center[horiz, 2]
    levels: List[float] = []
    for z in np.sort(zs):
        if not levels or abs(z - levels[-1]) > 1e-6 * max(1.0, abs(z)) + 1e-9:
            levels.append(float(z))
    thick = list(np.diff(levels)) if len(levels) > 1 else []
    ana = STLAnalysis(str(path), sha, watertight, int(len(mesh.faces)), mesh.bounds.tolist(), levels, [float(t) for t in thick],
                      warnings=warnings)
    ana.inferred["n_bodies"] = len(bodies)
    if not watertight:
        ana.warnings.append("Combined mesh is not a single closed manifold (normal for stacked layers sharing faces); "
                            "export/import layers as separate STL files for per-body watertightness checks")
    if len(levels) >= 2:
        # top metal pattern = union of upward-facing triangles on the highest level
        from shapely.geometry import Polygon as _P
        from shapely.ops import unary_union as _uu
        import shapely

        top = horiz & (n[:, 2] > 0) & (np.abs(mesh.triangles_center[:, 2] - levels[-1]) < 1e-6 * max(1.0, abs(levels[-1])) + 1e-9)
        tris = [_P(t[:, :2]) for t in mesh.triangles[top]]
        shape = _uu([t for t in tris if t.area > 0])
        b = mesh.bounds
        full_area = (b[1, 0] - b[0, 0]) * (b[1, 1] - b[0, 1])
        ana.inferred["top_fill_factor"] = float(shape.area / full_area) if full_area else None
        ana.inferred["top_layer_polygons"] = len(getattr(shape, "geoms", [shape]))
        xs = b[0, 0] + (np.arange(resolution) + 0.5) / resolution * (b[1, 0] - b[0, 0])
        ys = b[1, 1] - (np.arange(resolution) + 0.5) / resolution * (b[1, 1] - b[0, 1])
        X, Y = np.meshgrid(xs, ys)
        ana.top_mask = shapely.contains_xy(shape, X, Y)
        ana.mask_extent = [float(b[0, 0]), float(b[1, 0]), float(b[0, 1]), float(b[1, 1])]
        if ana.inferred["top_fill_factor"] and ana.inferred["top_fill_factor"] > 0.98:
            ana.warnings.append("Top layer covers the full footprint: no patterned layer detected on top")
        from metasurface_optimizer.geometry.image_parser import analyze_mask

        mpp = (b[1, 0] - b[0, 0]) / resolution
        ma = analyze_mask(ana.top_mask, mm_per_pixel=mpp)
        ana.inferred["components"] = [c.__dict__ for c in ma.components]
        ana.inferred["pixel_size"] = mpp
        ana.inferred["stack_thicknesses_bottom_to_top"] = ana.layer_thicknesses
    ana.confidence = Confidence.MEDIUM if ana.top_mask is not None and ana.inferred.get("components") else Confidence.LOW
    return ana
