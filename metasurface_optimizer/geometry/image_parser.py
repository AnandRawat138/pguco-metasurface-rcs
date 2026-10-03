"""Unit-cell image (PNG/JPG/BMP) -> binary mask -> approximate parameters.

Steps: grayscale -> Otsu threshold (or user threshold) -> optional inversion
so that metal = 1 -> connected components -> per-component moments
(area, centroid, orientation, principal extents) -> optional fit to a
rectangular-patch template with IoU-based confidence.

Pixel -> length conversion requires the user to provide ``mm_per_pixel`` or
the cell period (the image is then assumed to span exactly one period).
Without it, only dimensionless ratios are reported. Results are never claimed
exact; confidence is HIGH only for clean, high-resolution, well-fitting masks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from scipy import ndimage

from metasurface_optimizer.core.provenance import Confidence


def otsu_threshold(gray: np.ndarray) -> float:
    hist, edges = np.histogram(gray.ravel(), bins=256, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    return float(edges[np.argmax(sigma_b)])


def load_mask(path: str | Path, threshold: Optional[float] = None, metal_is_dark: Optional[bool] = None) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("L"), float) / 255.0
    t = otsu_threshold(img) if threshold is None else threshold
    mask = img > t
    if metal_is_dark is None:
        # heuristic: the border of a unit-cell image is usually background
        border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        metal_is_dark = border.mean() > 0.5
    return ~mask if metal_is_dark else mask


@dataclass
class ComponentInfo:
    area_px: int
    centroid_px: List[float]
    orientation_deg: float
    major_px: float
    minor_px: float
    bbox_px: List[int]
    rect_iou: float


@dataclass
class MaskAnalysis:
    shape: List[int]
    fill_factor: float
    n_components: int
    components: List[ComponentInfo] = field(default_factory=list)
    estimates: Dict[str, float] = field(default_factory=dict)
    confidence: Confidence = Confidence.LOW
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"shape": self.shape, "fill_factor": self.fill_factor, "n_components": self.n_components,
                "components": [c.__dict__ for c in self.components], "estimates": self.estimates, "confidence": self.confidence.value,
                "notes": self.notes, "source": "image_estimate"}


def _component(mask: np.ndarray) -> ComponentInfo:
    """Fit the minimum-area rotated rectangle to a component (rotating calipers via shapely)."""
    from shapely.geometry import MultiPoint

    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    # pixel corners so that a one-pixel-wide feature has non-zero width
    pts = np.concatenate([np.column_stack([xs + dx, ys + dy]) for dx in (0, 1) for dy in (0, 1)])
    rect = MultiPoint(pts).minimum_rotated_rectangle
    c = np.asarray(rect.exterior.coords)[:4]
    e1, e2 = c[1] - c[0], c[2] - c[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    major_vec = e1 if l1 >= l2 else e2
    L, W = max(l1, l2), min(l1, l2)
    ang = np.degrees(np.arctan2(-major_vec[1], major_vec[0]))  # image y axis points down
    ang = (ang + 90) % 180 - 90
    if L / max(W, 1e-9) < 1.05:  # square: orientation only defined modulo 90 deg
        ang = (ang + 45) % 90 - 45
    iou = float(len(xs) / max(rect.area, 1e-9))
    return ComponentInfo(int(len(xs)), [float(cx), float(cy)], float(ang), float(L), float(W),
                         [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], min(iou, 1.0))


def analyze_mask(mask: np.ndarray, mm_per_pixel: Optional[float] = None, period_mm: Optional[float] = None, min_area_px: int = 4) -> MaskAnalysis:
    mask = np.asarray(mask, bool)
    lab, n = ndimage.label(mask)
    comps = [_component(lab == k) for k in range(1, n + 1) if (lab == k).sum() >= min_area_px]
    res = MaskAnalysis(list(mask.shape), float(mask.mean()), len(comps), comps)
    if mm_per_pixel is None and period_mm is not None:
        mm_per_pixel = period_mm / mask.shape[1]
        res.notes.append("scale from period: image assumed to span exactly one period")
    if comps:
        big = max(comps, key=lambda c: c.area_px)
        res.estimates["aspect_ratio"] = big.major_px / max(big.minor_px, 1e-9)
        res.estimates["rotation_deg"] = big.orientation_deg
        res.estimates["length_over_cell"] = big.major_px / mask.shape[1]
        res.estimates["width_over_cell"] = big.minor_px / mask.shape[1]
        if mm_per_pixel:
            res.estimates["length_mm"] = big.major_px * mm_per_pixel
            res.estimates["width_mm"] = big.minor_px * mm_per_pixel
            res.estimates["pixel_uncertainty_mm"] = mm_per_pixel
        else:
            res.notes.append("no scale given: only dimensionless ratios reported")
        res_px = min(mask.shape)
        if big.rect_iou > 0.95 and res_px >= 200 and mm_per_pixel:
            res.confidence = Confidence.HIGH
        elif big.rect_iou > 0.85 and res_px >= 64:
            res.confidence = Confidence.MEDIUM
        else:
            res.confidence = Confidence.LOW
    return res
