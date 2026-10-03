"""Random-phase ring+disc metasurface of Song et al., IEEE AWPL 15, 329-331 (2016) - APPROXIMATE model.

THIS IS NOT A FULL-WAVE SOLVER AND NOT HFSS. Every output is labelled ``SYNTHETIC_ORACLE``.

Design (from the paper): 10 mm cells on a grounded FR4 slab (eps_r 4.4, h = 3 mm); each cell is a
metal ring (outer radius R1, inner R2 = k2 R1, k2 = 0.92) around a disc (R3 = k3 R1, k3 = 0.8),
R1 in [1.5, 4.5] mm. 30 x 30 cells (300 mm) form four interleaved sub-arrays (2 x 2 super-cell:
[[s1, s2], [s3, s4]]) designed at f1..f4 = 7.2, 8, 9, 10.8 GHz. In sub-array s every cell gets a
uniformly random target reflection phase at f_s; R1 follows from the phase(R1, f_s) curve (paper
Fig. 2). Claimed: > 10 dB backscatter RCS reduction against an equal PEC plate over 7-12 GHz.

Array parameterisation used for optimisation (``layout``): the random numbers u_ij in [0, 1) are fixed
by ``layout_seed``; the target phase of a cell of sub-array s is

    phi_ij = phi_s(R1_max) + u_ij * span * (phi_s(R1_min) - phi_s(R1_max))

and R1_ij = phi_s^{-1}(phi_ij) on the (monotonised, unwrapped) curve. Free design variables: f1..f4,
``phase_span``, ``r1_min``, ``r1_max`` (the paper's design: 7.2, 8, 9, 10.8 GHz, 1, 1.5, 4.5 mm).

Cell model (source ``fig2_calibrated``): an equivalent circuit (ring and disc as two series-LC grid
branches over the grounded slab, Luukkonen-type gap capacitance) gives |Gamma| and the frequency
dependence; its phase is then corrected by delta(R1, f), the circular difference to the paper's
Fig. 2 HFSS curves (``FIG2_*``, read by hand from the figure, about +-10 deg), linear in f between
7.2 and 10.8 GHz and held constant outside. So the model reproduces Fig. 2 at the four published
frequencies and interpolates in between. The correction is only valid for the paper's cell
(k2 = 0.92, k3 = 0.8, h = 3 mm, eps_r = 4.4); other values use the uncorrected circuit and are
reported as such. Source ``table`` replaces all of this with HFSS unit-cell results (R1 x f grid,
from the Template-A job batch) once they exist: :meth:`RingDiscCell.from_table`.

Array model: scalar physical optics as in ``analytic_oracle`` (sum over cells of Gamma_n a e^{jk.r}
with a cell element factor; PEC-plate reference of equal size); monostatic RCS at (theta_i, phi_i).
No coupling between dissimilar neighbours, no edge diffraction, no surface waves - the paper itself
attributes its resonance shifts to coupling, which this model cannot reproduce.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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

PAPER = {"period": 10.0, "k2": 0.92, "k3": 0.8, "substrate_thickness": 3.0, "eps_r": 4.4, "tan_delta": 0.02,
         "f1": 7.2, "f2": 8.0, "f3": 9.0, "f4": 10.8, "phase_span": 1.0, "r1_min": 1.5, "r1_max": 4.5, "n_cells": 30}
# Fig. 2 of the paper (HFSS, normal incidence), read by hand from the figure: +-10 deg, unwrapped.
FIG2_R1_MM = np.array([1.5, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5])
FIG2_F_GHZ = np.array([7.2, 8.0, 9.0, 10.8])
FIG2_PHASE_DEG = np.array([[109, 106, 95, 71, 6, -44, -107, -141, -163, -192],
                           [95, 82, 62, 21, -88, -115, -154, -175, -190, -230],
                           [73, 57, 24, -59, -142, -174, -195, -221, -248, -364],
                           [21, -12, -77, -175, -236, -296, -379, -447, -470, -500]], float)
# circuit parameters fitted to FIG2 (scratch fit, 23 deg RMS before the delta correction)
CIRCUIT = {"ring_gap_scale": 0.747, "ring_l": 2.347, "disc_l": 1.324, "disc_c": 0.0238, "r_s": 0.0}


def circuit_gamma(f: np.ndarray, r1_mm: np.ndarray, period_mm: float = 10.0, h_mm: float = 3.0, eps_r: float = 4.4, tan_d: float = 0.02,
                  k2: float = 0.92, k3: float = 0.8, theta_deg: float = 0.0, pol: str = "TE", p: Mapping[str, float] = CIRCUIT) -> np.ndarray:
    """Equivalent-circuit reflection, shape (len(r1), len(f))."""
    f = np.atleast_1d(np.asarray(f, float))[None, :]
    r1 = np.atleast_1d(np.asarray(r1_mm, float))[:, None] * 1e-3
    D, h = period_mm * 1e-3, h_mm * 1e-3
    w = 2 * np.pi * f
    k0 = w / C0
    th = np.radians(theta_deg)
    ec = eps_r * (1 - 1j * tan_d)
    ee = (ec + 1) / 2
    gap = np.clip(D - 2 * r1 * p["ring_gap_scale"], 1e-6 * D, D)
    Cr = EPS0 * ee * D / np.pi * np.log(1 / np.sin(np.pi * gap / (2 * D)))
    Lr = MU0 * p["ring_l"] * r1
    Cd = p["disc_c"] * EPS0 * ee * 2 * np.pi * k3 * r1 * np.log(1 + k3 / max(k2 - k3, 1e-3))
    Ld = MU0 * p["disc_l"] * r1
    Yg = 1 / (1j * w * Lr + 1 / (1j * w * Cr) + p["r_s"]) + 1 / (1j * w * Ld + 1 / (1j * w * Cd) + p["r_s"])
    if pol == "TM":
        Yg = Yg * (1 - k0**2 * np.sin(th) ** 2 / (2 * k0**2 * ee))
    kz = k0 * np.sqrt(ec - np.sin(th) ** 2)
    if pol == "TE":
        zsl, eta = 1j * w * MU0 * np.tan(kz * h) / kz, ETA0 / np.cos(th)
    else:
        zsl, eta = 1j * kz * np.tan(kz * h) / (w * EPS0 * ec), ETA0 * np.cos(th)
    zs = 1 / (Yg + 1 / zsl)
    return (zs - eta) / (zs + eta)


def _is_paper_cell(period, h, eps_r, k2, k3) -> bool:
    return all(abs(a - b) < 1e-6 for a, b in ((period, 10.0), (h, 3.0), (eps_r, 4.4), (k2, 0.92), (k3, 0.8)))


class RingDiscCell:
    """Cell reflection Gamma(R1, f, theta, pol) from the Fig.-2-calibrated circuit or an HFSS table."""

    def __init__(self, period=10.0, h=3.0, eps_r=4.4, tan_d=0.02, k2=0.92, k3=0.8):
        self.period, self.h, self.eps_r, self.tan_d, self.k2, self.k3 = period, h, eps_r, tan_d, k2, k3
        self.table: Optional[Dict[str, np.ndarray]] = None
        self.calibrated = _is_paper_cell(period, h, eps_r, k2, k3)
        if self.calibrated:
            g = circuit_gamma(FIG2_F_GHZ * 1e9, FIG2_R1_MM, period, h, eps_r, tan_d, k2, k3)
            self._delta = np.angle(np.exp(1j * np.radians(FIG2_PHASE_DEG.T)) / g)  # (R1, 4) rad

    @property
    def source(self) -> str:
        return "hfss_table" if self.table is not None else ("fig2_calibrated_circuit" if self.calibrated else "uncalibrated_circuit")

    @classmethod
    def from_table(cls, path: str | Path, **kw) -> "RingDiscCell":
        """CSV with header r1_mm,f_ghz,re,im on a full R1 x f grid (normal incidence), e.g. collected HFSS Template-A results."""
        a = np.genfromtxt(path, delimiter=",", names=True)
        r1, f = np.unique(a["r1_mm"]), np.unique(a["f_ghz"])
        G = np.full((len(r1), len(f)), np.nan, complex)
        for row in a:
            G[np.searchsorted(r1, row["r1_mm"]), np.searchsorted(f, row["f_ghz"])] = row["re"] + 1j * row["im"]
        if np.isnan(G).any():
            raise ValueError(f"{path}: table is not a complete R1 x f grid")
        c = cls(**kw)
        c.table = {"r1": r1, "f": f * 1e9, "G": G}
        return c

    def gamma(self, f: np.ndarray, r1_mm: np.ndarray, theta_deg: float = 0.0, pol: str = "TE") -> np.ndarray:
        """Shape (len(r1), len(f))."""
        f = np.atleast_1d(np.asarray(f, float))
        r1 = np.atleast_1d(np.asarray(r1_mm, float))
        if self.table is not None:  # bilinear in (R1, f) on Re/Im; oblique incidence uses the circuit's angle ratio
            t = self.table
            G = np.array([np.interp(np.clip(r1, t["r1"][0], t["r1"][-1]), t["r1"], t["G"][:, j].real)
                          + 1j * np.interp(np.clip(r1, t["r1"][0], t["r1"][-1]), t["r1"], t["G"][:, j].imag) for j in range(len(t["f"]))]).T
            G = np.array([np.interp(f, t["f"], g.real) + 1j * np.interp(f, t["f"], g.imag) for g in G])
            if theta_deg:
                ratio = circuit_gamma(f, r1, self.period, self.h, self.eps_r, self.tan_d, self.k2, self.k3, theta_deg, pol) / \
                        circuit_gamma(f, r1, self.period, self.h, self.eps_r, self.tan_d, self.k2, self.k3, 0.0, pol)
                G = G * ratio
            return G
        g = circuit_gamma(f, r1, self.period, self.h, self.eps_r, self.tan_d, self.k2, self.k3, theta_deg, pol)
        if not self.calibrated:
            return g
        fg = FIG2_F_GHZ * 1e9
        d = np.empty((len(r1), len(f)))
        # delta at the 4 frequencies, interpolated in R1 on the unit circle, then linearly in f (unwrapped)
        dR = np.stack([np.angle(np.interp(r1, FIG2_R1_MM, np.cos(self._delta[:, k])) + 1j * np.interp(r1, FIG2_R1_MM, np.sin(self._delta[:, k])))
                       for k in range(len(fg))], 1)  # (R1, 4)
        for i in range(len(r1)):
            un = np.unwrap(dR[i])
            d[i] = np.interp(f, fg, un)
        return g * np.exp(1j * d)


def layout_r1(cell: RingDiscCell, values: Mapping[str, float], n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """R1 (n x n, mm) and sub-array index (n x n) of the random-phase layout (module docstring)."""
    rng = np.random.default_rng(seed)
    u = rng.random((n, n))
    sub = 2 * (np.arange(n)[:, None] % 2) + (np.arange(n)[None, :] % 2)
    r_lo, r_hi = float(values["r1_min"]), float(values["r1_max"])
    grid = np.linspace(r_lo, r_hi, 301)
    R1 = np.empty((n, n))
    for s in range(4):
        fs = float(values[f"f{s + 1}"]) * 1e9
        ph = np.unwrap(np.angle(cell.gamma([fs], grid)[:, 0]))
        ph = np.minimum.accumulate(ph)  # monotone non-increasing in R1 (as in Fig. 2)
        tgt = ph[-1] + u[sub == s] * float(values.get("phase_span", 1.0)) * (ph[0] - ph[-1])
        R1[sub == s] = np.interp(tgt, ph[::-1], grid[::-1])
    return R1, sub


@dataclass
class ArrayConditions:
    f_start: float = 6e9
    f_stop: float = 13e9
    n_freq: int = 71
    theta_inc: Sequence[float] = (0.0, 15.0, 30.0)
    polarizations: Sequence[str] = ("TE", "TM")
    phi_inc: float = 0.0
    theta_s_step: float = 3.0
    phi_s_step: float = 15.0
    n_cells: int = 30
    layout_seed: int = 1
    cell_table: Optional[str] = None

    @classmethod
    def from_mapping(cls, m: Optional[Mapping[str, Any]]) -> "ArrayConditions":
        m = dict(m or {})
        out = {}
        for k, fdef in cls.__dataclass_fields__.items():
            if k not in m:
                continue
            v = m[k]
            if k in ("theta_inc", "polarizations"):
                v = tuple(float(a) for a in v) if k == "theta_inc" else tuple(v)
            elif k in ("n_freq", "n_cells", "layout_seed"):
                v = int(float(v))
            elif k != "cell_table":
                v = float(v)
            out[k] = v
        return cls(**out)


def array_rcs(cell: RingDiscCell, R1: np.ndarray, period_mm: float, c: ArrayConditions, bistatic: bool = True) -> RCSData:
    n = R1.shape[0]
    P = period_mm * 1e-3
    f = np.linspace(c.f_start, c.f_stop, c.n_freq)
    lam = C0 / f
    k = 2 * np.pi / lam
    xs = (np.arange(n) - (n - 1) / 2) * P
    X, Y = np.meshgrid(xs, xs[::-1])
    r1f = R1.ravel()
    uniq, inv = np.unique(np.round(r1f, 4), return_inverse=True)
    theta_s = np.arange(0.0, 90.0, c.theta_s_step) if bistatic else None
    phi_s = np.arange(0.0, 360.0, c.phi_s_step) if bistatic else None
    pols, th_inc = list(c.polarizations), np.asarray(c.theta_inc, float)
    mono = np.empty((len(pols), len(th_inc), len(f)))
    ref = np.empty_like(mono)
    bist = np.empty((len(pols), len(th_inc), len(f), len(theta_s), len(phi_s))) if bistatic else None
    A = (n * P) ** 2
    ph_i = np.radians(c.phi_inc)
    for pi, pol in enumerate(pols):
        for ii, ti in enumerate(th_inc):
            G = cell.gamma(f, uniq, ti, pol)[inv]  # (cells, F)
            ix, iy = np.sin(np.radians(ti)) * np.cos(ph_i), np.sin(np.radians(ti)) * np.sin(ph_i)
            # monostatic: s = -i_t -> phase 2 k i_t . r
            ph = np.exp(1j * 2 * k[None, :] * (ix * X.ravel()[:, None] + iy * Y.ravel()[:, None]))
            ux, uy = 2 * k * ix, 2 * k * iy
            ef = np.sinc(ux * P / (2 * np.pi)) * np.sinc(uy * P / (2 * np.pi))
            af = (G * ph).sum(0) * P**2 * ef
            mono[pi, ii] = 4 * np.pi / lam**2 * np.cos(np.radians(ti)) ** 2 * np.abs(af) ** 2
            ref[pi, ii] = 4 * np.pi * A**2 * np.cos(np.radians(ti)) ** 2 / lam**2
            if bistatic:
                TS, PS = np.meshgrid(np.radians(theta_s), np.radians(phi_s), indexing="ij")
                sx, sy = np.sin(TS) * np.cos(PS), np.sin(TS) * np.sin(PS)
                for kf in range(len(f)):
                    qx, qy = k[kf] * (sx + ix), k[kf] * (sy + iy)
                    e = np.exp(1j * (np.multiply.outer(qx, X.ravel()) + np.multiply.outer(qy, Y.ravel())))
                    a = (e @ G[:, kf]) * P**2 * np.sinc(qx * P / (2 * np.pi)) * np.sinc(qy * P / (2 * np.pi))
                    bist[pi, ii, kf] = 4 * np.pi / lam[kf] ** 2 * np.cos(TS) ** 2 * np.abs(a) ** 2
    return RCSData(f, th_inc, pols, c.phi_inc, theta_s, phi_s, bist, mono, ref,
                   reference_label=f"PEC plate {n * period_mm:g} x {n * period_mm:g} mm (physical optics)")


class RingDiscArrayModel(HighFidelitySolver):
    """Approximate high-fidelity stand-in for the Song et al. metasurface (module docstring)."""

    name = "ring_disc_random_phase_array_model"
    fidelity = Fidelity.SYNTHETIC_ORACLE

    def __init__(self, conditions: Optional[Mapping[str, Any]] = None, root: Optional[Path] = None, bistatic: bool = False):
        self.cond = ArrayConditions.from_mapping(conditions)
        self.root = Path(root) if root else Path(".")
        self.bistatic = bistatic
        self._cells: Dict[Tuple, RingDiscCell] = {}

    def cell(self, v: Mapping[str, float]) -> RingDiscCell:
        key = tuple(float(v.get(k, PAPER[k])) for k in ("period", "substrate_thickness", "eps_r", "tan_delta", "k2", "k3"))
        if key not in self._cells:
            kw = dict(zip(("period", "h", "eps_r", "tan_d", "k2", "k3"), key))
            if self.cond.cell_table:
                p = Path(self.cond.cell_table)
                self._cells[key] = RingDiscCell.from_table(p if p.is_absolute() else self.root / p, **kw)
            else:
                self._cells[key] = RingDiscCell(**kw)
        return self._cells[key]

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["conditions"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.cond.__dict__.items()}
        d["warning"] = "Approximate circuit + physical-optics model calibrated to the paper's Fig. 2; NOT a full-wave simulation."
        return d

    def layout(self, values: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, RingDiscCell]:
        v = {k: float(values.get(k, PAPER[k])) for k in PAPER}
        cell = self.cell(v)
        R1, sub = layout_r1(cell, v, self.cond.n_cells, self.cond.layout_seed)
        return R1, sub, cell

    def evaluate(self, values: Mapping[str, Any], conditions: Optional[Mapping[str, Any]] = None) -> EMResponse:
        c = ArrayConditions.from_mapping({**self.cond.__dict__, **(conditions or {})})
        R1, sub, cell = self.layout(values)
        period = float(values.get("period", PAPER["period"]))
        rcs = array_rcs(cell, R1, period, c, bistatic=self.bistatic)
        f = rcs.frequency
        sp = SParameterSet(f, {(1, 1, "TE"): cell.gamma(f, [float(np.median(R1))])[0]},
                           port_definition="Floquet, normal incidence, median-R1 cell (reference only)")
        prov = Provenance(Fidelity.SYNTHETIC_ORACLE, self.name, f"Approximate model ({cell.source}); not full-wave")
        return EMResponse(prov, sp, rcs, None, {"cell_source": cell.source, "layout_r1_mm": R1.round(4).tolist(), "subarray": sub.tolist(),
                                                "layout_seed": c.layout_seed, "conditions": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.__dict__.items()}})
