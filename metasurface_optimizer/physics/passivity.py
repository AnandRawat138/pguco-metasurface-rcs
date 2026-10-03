"""Passivity constraint for passive structures.

For a lossless/lossy *passive* N-port with power-normalised waves and matched
port definitions, the scattering matrix satisfies ``I - S^H S >= 0``. For the
two-port (reflection + transmission) case under a single incident mode this
reduces to (Eq. P1)

    |S11|^2 + |S21|^2 <= 1.

For a ground-backed reflector (no transmission) it reduces to |S11| <= 1.

The constraint is only meaningful when the structure is passive AND the
S-parameters are power-normalised to the same port modes (e.g. Floquet ports
of the same order). It is therefore applied only when
``PhysicsConfig.passive`` is true. Higher-order Floquet modes, cross-polar
channels or grating lobes that are not included in the data can carry power,
so the bound is necessary but not sufficient.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def power_balance(s11: np.ndarray, s21: Optional[np.ndarray] = None) -> np.ndarray:
    """|S11|^2 + |S21|^2 per frequency (|S11|^2 if no transmission data)."""
    p = np.abs(s11) ** 2
    if s21 is not None:
        p = p + np.abs(s21) ** 2
    return p


def passivity_violation(s11: np.ndarray, s21: Optional[np.ndarray] = None, tol: float = 1e-6) -> Dict[str, float]:
    """Max and mean of max(0, |S11|^2 + |S21|^2 - 1)."""
    v = np.maximum(power_balance(s11, s21) - 1.0 - tol, 0.0)
    return {"max": float(v.max()), "mean": float(v.mean()), "violated": bool((v > 0).any())}


def passivity_loss_torch(s11_re, s11_im, s21_re=None, s21_im=None):
    """L_pass = mean_f max(0, |S11|^2 + |S21|^2 - 1)^2   (Eq. P1-loss), torch tensors (..., F)."""
    import torch

    p = s11_re**2 + s11_im**2
    if s21_re is not None:
        p = p + s21_re**2 + s21_im**2
    return torch.relu(p - 1.0).pow(2).mean()


def project_passive(s11: np.ndarray, s21: Optional[np.ndarray] = None) -> tuple:
    """Scale (S11, S21) per frequency onto the passive set when the bound is exceeded."""
    p = power_balance(s11, s21)
    k = np.where(p > 1.0, 1.0 / np.sqrt(p), 1.0)
    return s11 * k, (None if s21 is None else s21 * k)
