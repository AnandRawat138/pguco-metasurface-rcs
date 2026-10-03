"""SYNTHETIC analytic benchmark oracle for checkerboard (two-state) RCS-reduction surfaces.

THIS IS NOT A FULL-WAVE SOLVER AND NOT HFSS/CST.

It exists so that the optimisation framework can be exercised end to end, and
benchmarked against baselines with a known, cheap "ground truth", when no real
high-fidelity data is available. All its outputs carry
``Fidelity.SYNTHETIC_ORACLE`` and are reported as such.

Model (approximate, closed form):

1. Each unit cell (state A or B) is a patch array on a grounded dielectric
   slab, described by the quasi-static high-impedance-surface model of
   Luukkonen et al., IEEE Trans. Antennas Propag. 56(6), 2008:

   * grid impedance  Z_g = -j eta_eff / (2 alpha),
     alpha = (k_eff D / pi) ln(1 / sin(pi w / (2 D))), eps_eff = (eps_r + 1)/2,
     with the TM oblique-incidence correction (1 - k0^2 sin^2(theta) / (2 k_eff^2));
   * grounded-slab impedance  TE: j omega mu0 tan(k_z h) / k_z,
     TM: j k_z tan(k_z h) / (omega eps0 eps_r),  k_z = k0 sqrt(eps_r - sin^2 theta);
   * Z_s = Z_g || Z_slab and Gamma = (Z_s - eta)/(Z_s + eta) with
     eta = eta0/cos(theta) (TE) or eta0 cos(theta) (TM).

   Dielectric loss enters through eps_r (1 - j tan_delta); a small sheet
   resistance models conductor loss. Patch rotation psi is handled
   approximately by rotating the principal-axis grid admittances (cross-
   polarisation is neglected).

2. The array is a checkerboard of M x M-cell tiles of states A and B. The
   bistatic RCS is computed with scalar physical optics:

       sigma(theta_s, phi_s) = (4 pi / lambda^2) cos^2(theta_s) |sum_n Gamma_n a e^{j k rho_n . (s_t + i_t)}|^2 |EF|^2

   with a = D^2 and a cell element factor EF (sinc of the cell aperture).
   The reference is a PEC plate (Gamma = -1) of identical size.

Limitations: no mutual coupling between dissimilar neighbouring cells, no
edge diffraction, no surface waves, no cross-polarisation. The model is good
enough to produce realistic trends (resonance shifts, 180-degree phase
difference bandwidth, grating lobes) but must never be used to claim
physical performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from metasurface_optimizer.core.provenance import Fidelity, Provenance
from metasurface_optimizer.electromagnetics.rcs import RCSData
from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.electromagnetics.sparameter import SParameterSet
from metasurface_optimizer.solvers.base import HighFidelitySolver

C0 = 299_792_458.0
MU0 = 4e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0**2)
ETA0 = MU0 * C0


@dataclass
class OracleConditions:
    """Simulation conditions; all values are configuration, nothing is hard-coded in the model."""

    f_start: float = 6e9
    f_stop: float = 14e9
    n_freq: int = 41
    theta_inc: Sequence[float] = (0.0, 20.0, 40.0)
    polarizations: Sequence[str] = ("TE", "TM")
    phi_inc: float = 0.0
    theta_s_step: float = 3.0
    phi_s_step: float = 15.0
    tile_cells: int = 4
    n_tiles: int = 6
    layout: str = "checkerboard"  # checkerboard | stripes
    sheet_resistance: float = 0.05  # ohm/sq, conductor loss
    length_unit_m: float = 1e-3
    noise_db: float = 0.0  # optional synthetic "measurement" noise
    noise_seed: int = 0

    @classmethod
    def from_mapping(cls, m: Optional[Mapping[str, Any]]) -> "OracleConditions":
        if not m:
            return cls()
        known = {k: v for k, v in m.items() if k in cls.__dataclass_fields__}
        for k, f in cls.__dataclass_fields__.items():
            if k in known and f.type in ("float", "int") and isinstance(known[k], str):
                known[k] = float(known[k]) if f.type == "float" else int(float(known[k]))
            if k in ("theta_inc",) and k in known:
                known[k] = tuple(float(v) for v in known[k])
        return cls(**known)


def cell_reflection(
    f: np.ndarray, period: float, length: float, width: float, rotation_deg: float,
    h: float, eps_r: float, tan_d: float, theta_deg: float, pol: str, rs: float = 0.05,
) -> np.ndarray:
    """Complex co-polarised reflection coefficient of one patch-array state (SI units, lengths in m)."""
    w0 = 2 * np.pi * f
    k0 = w0 / C0
    th = np.radians(theta_deg)
    eps_c = eps_r * (1 - 1j * tan_d)
    eps_eff = (eps_c + 1) / 2
    k_eff = k0 * np.sqrt(eps_eff)
    eta_eff = ETA0 / np.sqrt(eps_eff)

    def grid_admittance(gap: float) -> np.ndarray:
        gap = max(gap, 1e-6 * period)
        alpha = (k_eff * period / np.pi) * np.log(1.0 / np.sin(np.pi * gap / (2 * period)))
        zg = -1j * eta_eff / (2 * alpha)
        if pol == "TM":
            zg = zg / (1 - k0**2 * np.sin(th) ** 2 / (2 * k_eff**2))
        return 1.0 / (zg + rs)

    # principal axes u (along length), v (along width); E-field of TE (phi_inc=0) is along y, TM along x
    yu, yv = grid_admittance(period - length), grid_admittance(period - width)
    psi = np.radians(rotation_deg)
    if pol == "TE":  # E along y
        yg = yu * np.sin(psi) ** 2 + yv * np.cos(psi) ** 2
    else:  # E along x
        yg = yu * np.cos(psi) ** 2 + yv * np.sin(psi) ** 2
    zg = 1.0 / yg
    kz = k0 * np.sqrt(eps_c - np.sin(th) ** 2)
    if pol == "TE":
        zslab = 1j * w0 * MU0 * np.tan(kz * h) / kz
        eta = ETA0 / np.cos(th)
    else:
        zslab = 1j * kz * np.tan(kz * h) / (w0 * EPS0 * eps_c)
        eta = ETA0 * np.cos(th)
    zs = zg * zslab / (zg + zslab)
    return (zs - eta) / (zs + eta)


def _parity_sums(u: np.ndarray, n: int, m: int, d: float):
    """Sums of exp(j u x_k) over cells whose tile index along one axis is even / odd.

    Factorised as (sum over the m cells of one tile) x (sum over tiles), which
    costs O(m + n/m) exponentials instead of O(n). Returns (even, odd).
    """
    n_tiles = n // m
    x0 = -(n - 1) / 2 * d
    cell = np.exp(1j * u[..., None] * (np.arange(m) * d)).sum(-1)
    t = np.arange(n_tiles)
    tiles = np.exp(1j * u[..., None] * (x0 + t * m * d))
    return cell * tiles[..., t % 2 == 0].sum(-1), cell * tiles[..., t % 2 == 1].sum(-1)


class AnalyticCheckerboardOracle(HighFidelitySolver):
    """Synthetic oracle (see module docstring). Expects template variables of ``checkerboard_rect_patch``."""

    name = "analytic_checkerboard_oracle"
    fidelity = Fidelity.SYNTHETIC_ORACLE

    def __init__(self, conditions: Optional[Mapping[str, Any]] = None, template_map: Optional[Dict[str, str]] = None):
        self.cond = OracleConditions.from_mapping(conditions)
        self.template_map = template_map or {}

    def _v(self, values: Mapping[str, Any], var: str, default: Optional[float] = None) -> float:
        key = self.template_map.get(var, var)
        if key in values:
            return float(values[key])
        if default is None:
            raise KeyError(f"Oracle needs {var!r}")
        return default

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["conditions"] = self.cond.__dict__ | {"theta_inc": list(self.cond.theta_inc), "polarizations": list(self.cond.polarizations)}
        d["warning"] = "Synthetic analytic model; NOT a full-wave electromagnetic simulation."
        return d

    def evaluate(self, values: Mapping[str, Any], conditions: Optional[Mapping[str, Any]] = None) -> EMResponse:
        c = OracleConditions.from_mapping({**self.cond.__dict__, **(conditions or {})})
        u = c.length_unit_m
        P = self._v(values, "period") * u
        La, Wa = self._v(values, "length_a") * u, self._v(values, "width_a") * u
        Lb, Wb = self._v(values, "length_b") * u, self._v(values, "width_b") * u
        rot = self._v(values, "rotation", 0.0)
        h = self._v(values, "substrate_thickness") * u
        er = self._v(values, "eps_r", 4.3)
        td = self._v(values, "tan_delta", 0.02)

        f = np.linspace(c.f_start, c.f_stop, c.n_freq)
        th_inc = np.asarray(c.theta_inc, float)
        pols = list(c.polarizations)
        theta_s = np.arange(0.0, 90.0, c.theta_s_step)
        phi_s = np.arange(0.0, 360.0, c.phi_s_step)

        N = c.n_tiles * c.tile_cells  # cells per side
        A = (N * P) ** 2
        lam = C0 / f
        k = 2 * np.pi / lam
        TS, PS = np.meshgrid(np.radians(theta_s), np.radians(phi_s), indexing="ij")
        sx, sy = np.sin(TS) * np.cos(PS), np.sin(TS) * np.sin(PS)

        bist = np.empty((len(pols), len(th_inc), len(f), len(theta_s), len(phi_s)))
        ref = np.empty((len(pols), len(th_inc), len(f)))
        sdata = {}
        rng = np.random.default_rng(c.noise_seed)
        for pi, pol in enumerate(pols):
            for ii, ti in enumerate(th_inc):
                ga = cell_reflection(f, P, La, Wa, rot, h, er, td, ti, pol, c.sheet_resistance)
                gb = cell_reflection(f, P, Lb, Wb, rot, h, er, td, ti, pol, c.sheet_resistance)
                if ti == 0.0:
                    sdata[(1, 1, pol)] = ga
                    sdata[(1, 1, pol + "_B")] = gb
                ix = np.sin(np.radians(ti)) * np.cos(np.radians(c.phi_inc))
                iy = np.sin(np.radians(ti)) * np.sin(np.radians(c.phi_inc))
                ux = k[:, None, None] * (sx[None] + ix)
                uy = k[:, None, None] * (sy[None] + iy)
                ex, ox = _parity_sums(ux, N, c.tile_cells, P)
                ey, oy = _parity_sums(uy, N, c.tile_cells, P)
                if c.layout == "checkerboard":
                    sa, sb = ex * ey + ox * oy, ex * oy + ox * ey
                else:  # stripes along x
                    tot_y = ey + oy
                    sa, sb = ex * tot_y, ox * tot_y
                ef = np.sinc(ux * P / (2 * np.pi)) * np.sinc(uy * P / (2 * np.pi))
                af = (ga[:, None, None] * sa + gb[:, None, None] * sb) * P**2 * ef
                sig = 4 * np.pi / lam[:, None, None] ** 2 * np.cos(TS)[None] ** 2 * np.abs(af) ** 2
                if c.noise_db > 0:
                    sig = sig * 10 ** (rng.normal(0, c.noise_db, sig.shape) / 10)
                bist[pi, ii] = sig
                ref[pi, ii] = 4 * np.pi * A**2 * np.cos(np.radians(ti)) ** 2 / lam**2
        rcs = RCSData(f, th_inc, pols, c.phi_inc, theta_s, phi_s, bist, None, ref)
        sp = SParameterSet(f, sdata, port_definition="Floquet, normal incidence; pol suffix _B = state-B cell")
        prov = Provenance(Fidelity.SYNTHETIC_ORACLE, self.name, "Analytic HIS + physical-optics model; not full-wave")
        return EMResponse(prov, sp, rcs, None, {"oracle_conditions": {k_: (list(v) if isinstance(v, (tuple, list)) else v) for k_, v in c.__dict__.items()}})
