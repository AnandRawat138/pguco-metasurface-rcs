"""Radar cross-section data: sigma(f, theta, phi) for monostatic and bistatic cases.

Angle conventions (degrees):

* incidence direction is ``(theta_i, phi_i)``: the wave arrives *from* that direction;
* monostatic (back-scatter) observation is ``(theta_s, phi_s) = (theta_i, phi_i)``;
* the specular (mirror) direction is ``(theta_s, phi_s) = (theta_i, phi_i + 180)``;
  at normal incidence it coincides with the monostatic direction.

``sigma`` is stored linearly in m^2; ``sigma_dBsm = 10 log10(sigma)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def to_dbsm(sigma: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(sigma, float), 1e-30))


def from_dbsm(dbsm: np.ndarray) -> np.ndarray:
    return 10.0 ** (np.asarray(dbsm, float) / 10.0)


@dataclass
class RCSData:
    """RCS container.

    Shapes: ``bistatic`` (P, I, F, T, Phi); ``monostatic`` and ``reference`` (P, I, F)
    where P = polarizations, I = incidence angles, F = frequencies, T/Phi = observation grid.
    ``reference`` is the RCS of an equal-size PEC plate (or any user-chosen reference)
    used for RCS-reduction metrics.
    """

    frequency: np.ndarray
    theta_inc: np.ndarray
    polarizations: List[str]
    phi_inc: float = 0.0
    theta_s: Optional[np.ndarray] = None
    phi_s: Optional[np.ndarray] = None
    bistatic: Optional[np.ndarray] = None
    monostatic: Optional[np.ndarray] = None
    reference: Optional[np.ndarray] = None
    reference_label: str = "PEC plate of equal size"
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequency = np.atleast_1d(np.asarray(self.frequency, float))
        self.theta_inc = np.atleast_1d(np.asarray(self.theta_inc, float))
        P, I, F = len(self.polarizations), len(self.theta_inc), len(self.frequency)
        if self.bistatic is not None:
            self.bistatic = np.asarray(self.bistatic, float)
            if self.theta_s is None or self.phi_s is None:
                raise ValueError("bistatic RCS requires theta_s and phi_s grids")
            exp = (P, I, F, len(self.theta_s), len(self.phi_s))
            if self.bistatic.shape != exp:
                raise ValueError(f"bistatic shape {self.bistatic.shape} != {exp}")
        for name in ("monostatic", "reference"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.asarray(arr, float)
                if arr.shape != (P, I, F):
                    raise ValueError(f"{name} shape {arr.shape} != {(P, I, F)}")
                setattr(self, name, arr)
        if self.monostatic is None and self.bistatic is not None:
            self.monostatic = self._lookup(self.phi_inc)

    # ---------------------------------------------------------- lookups
    def _nearest(self, grid: np.ndarray, v: float, periodic: bool = False) -> int:
        d = np.abs(grid - v)
        if periodic:
            d = np.minimum(d, 360.0 - d)
        return int(np.argmin(d))

    def _lookup(self, phi_obs: float) -> np.ndarray:
        """sigma at theta_s = theta_inc, phi_s = phi_obs (nearest grid point)."""
        P, I, F = len(self.polarizations), len(self.theta_inc), len(self.frequency)
        out = np.empty((P, I, F))
        jp = self._nearest(np.mod(self.phi_s, 360.0), np.mod(phi_obs, 360.0), periodic=True)
        for ii, th in enumerate(self.theta_inc):
            jt = self._nearest(self.theta_s, abs(th))
            # a negative incidence angle in the phi_inc plane is the same as +theta in phi_inc+180
            jpp = jp if th >= 0 else self._nearest(np.mod(self.phi_s, 360.0), np.mod(phi_obs + 180, 360.0), True)
            out[:, ii, :] = self.bistatic[:, ii, :, jt, jpp]
        return out

    def specular(self) -> np.ndarray:
        """sigma in the mirror-reflection direction, shape (P, I, F)."""
        if self.bistatic is not None:
            return self._lookup(self.phi_inc + 180.0)
        # without bistatic data, specular is only known at normal incidence
        if self.monostatic is None:
            raise ValueError("No RCS data available")
        if not np.allclose(self.theta_inc, 0.0):
            raise ValueError("Specular RCS at oblique incidence requires bistatic data")
        return self.monostatic

    def pol_index(self, pol: str) -> int:
        return self.polarizations.index(pol)

    def inc_index(self, theta: float) -> int:
        return self._nearest(self.theta_inc, theta)

    def reduction_db(self, kind: str = "specular") -> np.ndarray:
        """10 log10(sigma / sigma_ref): negative = reduction relative to the reference."""
        if self.reference is None:
            raise ValueError("No reference RCS available for reduction metric")
        s = self.specular() if kind == "specular" else self.monostatic
        return to_dbsm(s) - to_dbsm(self.reference)

    # ---------------------------------------------------- angular power
    def solid_angle_weights(self) -> np.ndarray:
        """Quadrature weights sin(theta) dtheta dphi on the observation grid, shape (T, Phi)."""
        th = np.radians(self.theta_s)
        ph = np.radians(self.phi_s)
        dth = np.gradient(th) if len(th) > 1 else np.array([1.0])
        dph = np.gradient(ph) if len(ph) > 1 else np.array([2 * np.pi])
        return np.outer(np.sin(th) * np.abs(dth), np.abs(dph)) + 1e-30

    def angular_power_distribution(self, pol: int, inc: int, freq: int) -> np.ndarray:
        """Normalised scattered power p_i = P_i / sum_j P_j over observation cells (flattened).

        P_i = sigma_i * dOmega_i, i.e. power scattered into the solid-angle cell i.
        """
        if self.bistatic is None:
            raise ValueError("Angular distribution requires bistatic RCS")
        P = self.bistatic[pol, inc, freq] * self.solid_angle_weights()
        P = P.ravel()
        return P / max(P.sum(), 1e-300)

    def total_scattered(self) -> np.ndarray:
        """Integral of sigma over the observation hemisphere / 4 pi (shape P, I, F) — proportional to scattered power."""
        if self.bistatic is None:
            raise ValueError("requires bistatic RCS")
        w = self.solid_angle_weights()
        return np.einsum("pifta,ta->pif", self.bistatic, w) / (4 * np.pi)

    # ---------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        def tl(a):
            return None if a is None else np.asarray(a).tolist()

        return {
            "frequency": tl(self.frequency), "theta_inc": tl(self.theta_inc), "polarizations": self.polarizations,
            "phi_inc": self.phi_inc, "theta_s": tl(self.theta_s), "phi_s": tl(self.phi_s),
            "bistatic": tl(self.bistatic), "monostatic": tl(self.monostatic), "reference": tl(self.reference),
            "reference_label": self.reference_label, "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RCSData":
        def arr(k):
            return None if d.get(k) is None else np.asarray(d[k], float)

        return cls(arr("frequency"), arr("theta_inc"), list(d["polarizations"]), d.get("phi_inc", 0.0), arr("theta_s"),
                   arr("phi_s"), arr("bistatic"), arr("monostatic"), arr("reference"), d.get("reference_label", ""), d.get("meta", {}))


def monostatic_from_table(rows: List[dict], freq_unit: str = "ghz") -> RCSData:
    """Build monostatic RCSData from long-format rows with keys
    ``frequency``, ``theta_inc`` (deg), ``pol``, and ``rcs_dbsm`` or ``rcs_m2``
    (optionally ``ref_dbsm`` / ``ref_m2``)."""
    scale = {"hz": 1, "mhz": 1e6, "ghz": 1e9}[freq_unit.lower()]
    freqs = sorted({float(r["frequency"]) for r in rows})
    incs = sorted({float(r.get("theta_inc", 0.0)) for r in rows})
    pols = sorted({str(r.get("pol", "TE")).upper() for r in rows})
    mono = np.full((len(pols), len(incs), len(freqs)), np.nan)
    ref = np.full_like(mono, np.nan)
    for r in rows:
        k = (pols.index(str(r.get("pol", "TE")).upper()), incs.index(float(r.get("theta_inc", 0.0))), freqs.index(float(r["frequency"])))
        mono[k] = float(r["rcs_m2"]) if "rcs_m2" in r else from_dbsm(float(r["rcs_dbsm"]))
        if "ref_m2" in r or "ref_dbsm" in r:
            ref[k] = float(r["ref_m2"]) if "ref_m2" in r else from_dbsm(float(r["ref_dbsm"]))
    if np.isnan(mono).any():
        raise ValueError("RCS table is not a complete (pol, angle, frequency) grid")
    return RCSData(np.asarray(freqs) * scale, np.asarray(incs), pols, monostatic=mono,
                   reference=None if np.isnan(ref).all() else ref)
