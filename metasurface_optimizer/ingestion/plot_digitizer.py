"""Semi-automatic digitisation of response curves from published figures.

Pipeline: figure -> axis calibration -> curve extraction -> response vector.

* Axis calibration: the user supplies two reference points per axis
  (pixel <-> value); :func:`detect_axes` proposes the plot frame from the
  longest dark horizontal/vertical lines, but values still have to be typed in.
* Curve extraction: pixels close to a target colour (or darkest non-axis pixels)
  are collected; per pixel column the median row is taken.
* Output: frequency/value pairs with ``source = plot_digitized``, a LOW
  confidence label and an uncertainty equal to one pixel in data units (plus
  the line thickness). Values are rounded to that resolution so no precision
  beyond the plot is fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from metasurface_optimizer.core.provenance import Confidence, ValueSource


@dataclass
class AxisCalibration:
    x_pixels: Tuple[float, float]
    x_values: Tuple[float, float]
    y_pixels: Tuple[float, float]
    y_values: Tuple[float, float]
    x_log: bool = False
    y_log: bool = False

    def px_to_x(self, px: np.ndarray) -> np.ndarray:
        a, b = self.x_values
        if self.x_log:
            a, b = np.log10(a), np.log10(b)
        v = a + (np.asarray(px) - self.x_pixels[0]) * (b - a) / (self.x_pixels[1] - self.x_pixels[0])
        return 10**v if self.x_log else v

    def py_to_y(self, py: np.ndarray) -> np.ndarray:
        a, b = self.y_values
        if self.y_log:
            a, b = np.log10(a), np.log10(b)
        v = a + (np.asarray(py) - self.y_pixels[0]) * (b - a) / (self.y_pixels[1] - self.y_pixels[0])
        return 10**v if self.y_log else v

    def resolution(self) -> Tuple[float, float]:
        return (abs(self.x_values[1] - self.x_values[0]) / abs(self.x_pixels[1] - self.x_pixels[0]),
                abs(self.y_values[1] - self.y_values[0]) / abs(self.y_pixels[1] - self.y_pixels[0]))


@dataclass
class DigitizedCurve:
    x: np.ndarray
    y: np.ndarray
    x_uncertainty: float
    y_uncertainty: float
    source: ValueSource = ValueSource.PLOT_DIGITIZED
    confidence: Confidence = Confidence.LOW
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"x": self.x.tolist(), "y": self.y.tolist(), "x_uncertainty": self.x_uncertainty, "y_uncertainty": self.y_uncertainty,
                "source": self.source.value, "confidence": self.confidence.value, "notes": self.notes,
                "label": "REPORTED (plot-digitized, approximate)"}


def load_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), float) / 255.0


def detect_axes(img: np.ndarray, dark: float = 0.35, min_frac: float = 0.5) -> Dict[str, Optional[int]]:
    """Propose the plot frame: rows/cols where long runs of dark pixels exist (left/bottom axis lines)."""
    g = img.mean(2) if img.ndim == 3 else img
    d = g < dark
    rows = np.where(d.mean(1) > min_frac)[0]
    cols = np.where(d.mean(0) > min_frac)[0]
    return {"x_axis_row": int(rows.max()) if len(rows) else None, "top_row": int(rows.min()) if len(rows) else None,
            "y_axis_col": int(cols.min()) if len(cols) else None, "right_col": int(cols.max()) if len(cols) else None,
            "note": "proposal only; confirm the frame and type the axis values"}


def extract_curve(img: np.ndarray, cal: AxisCalibration, color: Optional[Tuple[float, float, float]] = None, tol: float = 0.15,
                  roi: Optional[Tuple[int, int, int, int]] = None, n_points: Optional[int] = None) -> DigitizedCurve:
    """Extract one curve. ``color`` = RGB in [0,1] of the trace (None = darkest pixels); roi = (r0, r1, c0, c1)."""
    H, W = img.shape[:2]
    r0, r1, c0, c1 = roi or (0, H, 0, W)
    sub = img[r0:r1, c0:c1]
    if color is None:
        mask = sub.mean(2) < 0.3
    else:
        mask = np.linalg.norm(sub - np.asarray(color)[None, None], axis=2) < tol
    xs, ys, thick = [], [], []
    for c in range(mask.shape[1]):
        r = np.where(mask[:, c])[0]
        if len(r) and len(r) < 0.5 * mask.shape[0]:  # skip columns dominated by axes/grid
            xs.append(c + c0); ys.append(np.median(r) + r0); thick.append(len(r))
    if not xs:
        raise ValueError("No curve pixels found; adjust the colour, tolerance or region of interest")
    xv, yv = cal.px_to_x(np.array(xs, float)), cal.py_to_y(np.array(ys, float))
    o = np.argsort(xv)
    xv, yv = xv[o], yv[o]
    rx, ry = cal.resolution()
    ry_eff = ry * max(1.0, float(np.median(thick)) / 2)
    if n_points:
        xg = np.linspace(xv.min(), xv.max(), n_points)
        yv = np.interp(xg, xv, yv); xv = xg
    # never report more precision than the plot supports
    dec_y = max(0, int(-np.floor(np.log10(ry_eff))) if ry_eff > 0 else 3)
    dec_x = max(0, int(-np.floor(np.log10(rx))) if rx > 0 else 3)
    return DigitizedCurve(np.round(xv, dec_x), np.round(yv, dec_y), rx, ry_eff,
                          notes=[f"pixel resolution x={rx:.3g}, y={ry:.3g}; line thickness ~{np.median(thick):.0f}px"])
