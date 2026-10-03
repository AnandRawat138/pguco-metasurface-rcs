"""Physically interpretable features extracted from an electromagnetic response.

These feed the Baseline Analysis tab, the report generator and the
physics-aware feature extractor. Every function is deterministic and works
on whatever frequency grid the data comes with (no hard-coded bands).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from metasurface_optimizer.electromagnetics.rcs import RCSData, to_dbsm
from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.electromagnetics.sparameter import SParameterSet


def band_mask(frequency: np.ndarray, band: Optional[Tuple[float, float]]) -> np.ndarray:
    if band is None:
        return np.ones_like(frequency, dtype=bool)
    m = (frequency >= band[0] - 1e-9 * band[1]) & (frequency <= band[1] * (1 + 1e-9))
    if not m.any():
        raise ValueError(f"No frequency samples inside band {band}")
    return m


def resonances_from_magnitude(frequency: np.ndarray, mag_db: np.ndarray, prominence_db: float = 3.0) -> List[Dict[str, float]]:
    """Dips of a magnitude response (e.g. |S11| absorption or |S21| stop-band nulls)."""
    idx, props = find_peaks(-mag_db, prominence=prominence_db)
    return [{"frequency": float(frequency[i]), "level_db": float(mag_db[i]), "prominence_db": float(p)} for i, p in zip(idx, props["prominences"])]


def zero_phase_crossings(frequency: np.ndarray, s: np.ndarray) -> List[float]:
    """Frequencies where the reflection phase crosses 0 deg (AMC / in-phase resonance)."""
    ph = np.angle(s)
    out = []
    for k in range(len(ph) - 1):
        if ph[k] * ph[k + 1] < 0 and abs(ph[k] - ph[k + 1]) < np.pi:  # exclude +-180 wraps
            t = ph[k] / (ph[k] - ph[k + 1])
            out.append(float(frequency[k] + t * (frequency[k + 1] - frequency[k])))
    return out


def threshold_bandwidth(frequency: np.ndarray, level_db: np.ndarray, threshold_db: float) -> Dict[str, float]:
    """Largest contiguous band where ``level_db <= threshold_db``; returns absolute and fractional bandwidth."""
    ok = level_db <= threshold_db
    best = (0, -1)
    start = None
    for k, v in enumerate(np.append(ok, False)):
        if v and start is None:
            start = k
        elif not v and start is not None:
            if k - 1 - start > best[1] - best[0]:
                best = (start, k - 1)
            start = None
    if best[1] < best[0]:
        return {"f_low": float("nan"), "f_high": float("nan"), "bandwidth": 0.0, "fractional_bandwidth": 0.0, "covered_fraction": float(ok.mean())}
    fl, fh = frequency[best[0]], frequency[best[1]]
    return {
        "f_low": float(fl), "f_high": float(fh), "bandwidth": float(fh - fl),
        "fractional_bandwidth": float(2 * (fh - fl) / (fh + fl)) if fh + fl > 0 else 0.0,
        "covered_fraction": float(ok.mean()),
    }


def sparam_features(s: SParameterSet, band=None) -> Dict[str, object]:
    out: Dict[str, object] = {}
    m = band_mask(s.frequency, band)
    for (i, j, p) in sorted(set(s.data) | set(s.magnitude_only_db)):
        db = s.db(i, j, p)
        key = f"S{i}{j}_{p}"
        out[f"{key}_mean_db"] = float(np.mean(db[m]))
        out[f"{key}_min_db"] = float(np.min(db[m]))
        out[f"{key}_resonances"] = resonances_from_magnitude(s.frequency, db)
        out[f"{key}_bw_-10dB"] = threshold_bandwidth(s.frequency[m], db[m], -10.0)
        if (i, j, p) in s.data and i == j:
            out[f"{key}_zero_phase_crossings"] = zero_phase_crossings(s.frequency, s.data[(i, j, p)])
    return out


def angular_entropy(rcs: RCSData, pol: int, inc: int, freq: int, eps: float = 1e-12, normalized: bool = True) -> float:
    """H_ang = -sum_i p_i log(p_i + eps), optionally divided by log(N)."""
    p = rcs.angular_power_distribution(pol, inc, freq)
    h = float(-np.sum(p * np.log(p + eps)))
    return h / np.log(len(p)) if normalized else h


def rcs_features(rcs: RCSData, band=None, reduction_threshold_db: float = -10.0) -> Dict[str, object]:
    out: Dict[str, object] = {}
    m = band_mask(rcs.frequency, band)
    try:
        spec = rcs.specular()
    except ValueError:
        spec = None
    for pi, pol in enumerate(rcs.polarizations):
        for ii, th in enumerate(rcs.theta_inc):
            tag = f"{pol}_inc{th:g}"
            if rcs.monostatic is not None:
                d = to_dbsm(rcs.monostatic[pi, ii, m])
                out[f"mono_{tag}"] = {"mean_dbsm": float(d.mean()), "max_dbsm": float(d.max()), "min_dbsm": float(d.min()), "std_db": float(d.std())}
            if spec is not None:
                out[f"spec_{tag}_mean_dbsm"] = float(to_dbsm(spec[pi, ii, m]).mean())
            if rcs.reference is not None and spec is not None:
                red = to_dbsm(spec[pi, ii]) - to_dbsm(rcs.reference[pi, ii])
                out[f"reduction_{tag}"] = {
                    "mean_db": float(red[m].mean()),
                    "bw": threshold_bandwidth(rcs.frequency, red, reduction_threshold_db),
                }
            if rcs.bistatic is not None:
                ent = [angular_entropy(rcs, pi, ii, k) for k in np.where(m)[0]]
                out[f"entropy_{tag}_mean"] = float(np.mean(ent))
    return out


def response_summary(resp: EMResponse, band=None) -> Dict[str, object]:
    out: Dict[str, object] = {"fidelity": resp.fidelity.value}
    if resp.sparams is not None:
        out["sparams"] = sparam_features(resp.sparams, band)
    if resp.rcs is not None:
        out["rcs"] = rcs_features(resp.rcs, band)
    return out
