"""Phase-consistency checks for complex responses.

1. Representation consistency: when a response is predicted as (Re, Im),
   magnitude and phase are *derived*, so they are consistent by construction.
   When a model predicts magnitude and phase separately the pair is
   recombined here, never mixed.
2. Phase continuity (Eq. P3): on a sufficiently dense grid the unwrapped
   phase should not jump by more than ``max_step_deg`` between adjacent
   frequency samples except at genuine resonances (near-zero magnitude).
   Jumps where |S| is not small indicate under-sampling or a non-physical
   prediction.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def phase_jumps(s: np.ndarray, max_step_deg: float = 90.0, low_mag: float = 0.1) -> Dict[str, object]:
    ph = np.degrees(np.unwrap(np.angle(s)))
    step = np.abs(np.diff(ph))
    near_zero = np.minimum(np.abs(s[:-1]), np.abs(s[1:])) < low_mag
    bad = (step > max_step_deg) & ~near_zero
    return {"n_suspicious": int(bad.sum()), "max_step_deg": float(step.max()) if len(step) else 0.0, "indices": np.where(bad)[0].tolist()}


def phase_continuity_loss_torch(re, im, eps: float = 1e-8):
    """Penalise large changes of the unit phasor between adjacent samples, weighted by magnitude.

    L = mean_f | s_{f+1}/|s_{f+1}| - s_f/|s_f| |^2 * min(|s_f|, |s_{f+1}|)
    (small near resonant nulls where large phase changes are physical).
    """
    import torch

    mag = torch.sqrt(re**2 + im**2 + eps)
    ur, ui = re / mag, im / mag
    d = (ur[..., 1:] - ur[..., :-1]) ** 2 + (ui[..., 1:] - ui[..., :-1]) ** 2
    w = torch.minimum(mag[..., 1:], mag[..., :-1])
    return (d * w).mean()
