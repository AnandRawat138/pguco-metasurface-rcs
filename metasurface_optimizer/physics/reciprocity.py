"""Reciprocity S_ij = S_ji (Eq. P2).

Holds for structures made of reciprocal media (no magnetised ferrites,
no active/time-modulated elements) when ports i and j use matching mode
definitions and reference planes. Applied only when
``PhysicsConfig.reciprocal`` is true and both S_ij and S_ji are present.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def reciprocity_violation(s_ij: np.ndarray, s_ji: np.ndarray) -> Dict[str, float]:
    d = np.abs(s_ij - s_ji)
    scale = np.maximum(np.abs(s_ij) + np.abs(s_ji), 1e-12)
    return {"max_abs": float(d.max()), "max_rel": float((2 * d / scale).max())}


def reciprocity_loss_torch(sij_re, sij_im, sji_re, sji_im):
    """L_rec = mean_f |S_ij - S_ji|^2."""
    return ((sij_re - sji_re) ** 2 + (sij_im - sji_im) ** 2).mean()
