"""Research metrics (Sec. 41).

Prediction:   MAE, RMSE, R^2
Optimisation: Delta RCS (dB), Delta BW (fractional bandwidth), Delta H_ang, Delta J (%)
Efficiency:   E = Delta J / N_HF   (improvement per high-fidelity simulation)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean(0)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def prediction_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    return {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "R2": r2(y, yhat)}


def efficiency(delta_J: float, n_hf: int) -> float:
    """E = Delta J / N_HF (Delta J > 0 = improvement)."""
    return float(delta_J / n_hf) if n_hf > 0 else float("nan")


def calibration(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    """Fraction of validated values inside mu +/- k sigma and mean standardised error (uncertainty quality)."""
    y, mu, sigma = (np.asarray(a, float) for a in (y, mu, sigma))
    z = np.abs(y - mu) / np.maximum(sigma, 1e-12)
    return {"coverage_1sigma": float(np.mean(z <= 1)), "coverage_2sigma": float(np.mean(z <= 2)), "mean_abs_z": float(np.mean(z))}


def objective_deltas(J_old: np.ndarray, J_new: np.ndarray, names, kinds) -> Dict[str, Dict[str, float]]:
    """Per-objective changes with physically named deltas where meaningful."""
    out = {}
    for n, k, a, b in zip(names, kinds, J_old, J_new):
        d = {"old": float(a), "new": float(b), "delta": float(b - a), "delta_pct": float((b - a) / max(abs(a), 1e-12) * 100)}
        if k in ("specular_rcs", "broadband_rcs", "angular_robustness", "polarization_robustness"):
            d["meaning"] = "Delta RCS (in the objective's scale; negative = lower RCS)"
        elif k == "angular_entropy":
            d["meaning"] = "Delta(-H_ang); negative = more diffuse scattering"
        elif k == "reduction_bandwidth":
            d["meaning"] = "Delta(-FBW); negative = wider reduction bandwidth"
        out[n] = d
    return out
