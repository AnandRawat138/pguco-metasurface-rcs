"""Spectral smoothness regulariser (Eq. P4).

    L_smooth = sum_f | d^2 y_hat / d f^2 |^2   ~=  mean_f (y_{f+1} - 2 y_f + y_{f-1})^2 / df^4

computed on the normalised frequency axis (so the weight is grid independent).

To avoid over-constraining genuine resonances, a *resonance-preserving*
variant down-weights curvature where the observed training data itself is
strongly curved: w_f = 1 / (1 + (c_f / c_ref)^2) with c_f the curvature of
the mean training spectrum. With ``robust=True`` a Huber penalty replaces the
square so isolated sharp features are penalised only linearly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def second_difference(y: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.diff(y, n=2, axis=axis)


def roughness(y: np.ndarray) -> float:
    """Mean squared second difference on a unit-normalised frequency axis."""
    n = y.shape[-1]
    h = 1.0 / max(n - 1, 1)
    return float(np.mean((second_difference(y) / h**2) ** 2))


def resonance_weights(reference: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    """Weights w_f (length F-2) that are small where the reference spectra are sharply curved."""
    c = np.abs(second_difference(np.atleast_2d(reference))).mean(0)
    cref = np.quantile(c, quantile) + 1e-12
    return 1.0 / (1.0 + (c / cref) ** 2)


def smoothness_loss_torch(y, weights: Optional["torch.Tensor"] = None, robust: bool = False, delta: float = 1.0):
    """y: (..., F) torch tensor on the normalised axis. Returns scalar loss."""
    import torch

    d2 = y[..., 2:] - 2 * y[..., 1:-1] + y[..., :-2]
    pen = torch.nn.functional.huber_loss(d2, torch.zeros_like(d2), reduction="none", delta=delta) if robust else d2**2
    if weights is not None:
        pen = pen * weights
    return pen.mean()
