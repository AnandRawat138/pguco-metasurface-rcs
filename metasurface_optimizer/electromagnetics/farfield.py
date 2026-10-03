"""Far-field components E_theta(f, theta, phi), E_phi(f, theta, phi) and conversion to RCS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FarField:
    """Scattered far-field normalised to r (i.e. r*E_s), shape (F, T, Phi) per component.

    ``incident_amplitude`` is |E_i| at the target (V/m).
    """

    frequency: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    e_theta: np.ndarray
    e_phi: np.ndarray
    incident_amplitude: float = 1.0
    polarization: str = "TE"
    theta_inc: float = 0.0

    def __post_init__(self) -> None:
        exp = (len(self.frequency), len(self.theta), len(self.phi))
        for name in ("e_theta", "e_phi"):
            arr = np.asarray(getattr(self, name), complex)
            if arr.shape != exp:
                raise ValueError(f"{name} shape {arr.shape} != {exp}")
            setattr(self, name, arr)

    def rcs(self, component: Optional[str] = None) -> np.ndarray:
        """sigma = 4 pi |r E_s|^2 / |E_i|^2 (total, or a single co/cross component)."""
        if component == "theta":
            e2 = np.abs(self.e_theta) ** 2
        elif component == "phi":
            e2 = np.abs(self.e_phi) ** 2
        else:
            e2 = np.abs(self.e_theta) ** 2 + np.abs(self.e_phi) ** 2
        return 4 * np.pi * e2 / self.incident_amplitude**2

    def axial_ratio_db(self) -> np.ndarray:
        """Axial ratio of the scattered field (useful for polarization-control surfaces)."""
        a, b = self.e_theta, self.e_phi
        s0 = np.abs(a) ** 2 + np.abs(b) ** 2
        s3 = 2 * np.imag(a * np.conj(b))
        sin2chi = np.clip(s3 / np.maximum(s0, 1e-30), -1, 1)
        chi = 0.5 * np.arcsin(sin2chi)
        ar = 1.0 / np.maximum(np.abs(np.tan(chi)), 1e-6)
        return 20 * np.log10(ar)
