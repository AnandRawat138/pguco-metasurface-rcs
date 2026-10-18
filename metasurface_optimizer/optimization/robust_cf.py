"""Robust counterfactual optimization of the Song et al. (2016) metasurface (AEU manuscript, Secs. 3-4).

Implements, in the manuscript's notation:

* design x = [f1, f2, f3, f4, kappa2, kappa3, h] (Eq. 10), layout by the phase-inversion rule (Eq. 11) with a fixed
  seed xi: target phases phi*_n ~ U[0, 2 pi), R1_n = argmin_R |exp(j phi(R; f_s(n), kappa2, kappa3, h)) - exp(j phi*_n)|;
* fabrication constraints g1-g4 (Eq. 12) with w_min = g_min = 0.1 mm, Delta f_min = 0.3 GHz,
  h in {3.0, 3.2} mm (published thickness and standard FR4 stock), and the validity range of the cell database (0.07 <= kappa2 - kappa3 <= 0.20);
* specular reduction with the phase-error expectation (Eqs. 7-9), sigma_delta = 10 deg;
* robust worst case over f in [7, 12] GHz (0.1 GHz), theta_i in {0, 10, 20, 30} deg, TE/TM, eps_r in {4.2, 4.4, 4.6}
  (Eq. 13), counterfactual objective J = -rho_wc + lambda_cf ||w (x - x0)||_1 (Eq. 14), log-sum-exp relaxation (Eq. 15)
  for the surrogate search only;
* low-fidelity model f_LF: equivalent circuit (Eqs. 3-6) whose coefficients beta are fitted (Eq. 17) to the full-wave
  sweep of the baseline cell only; high-fidelity model f_HF: openEMS FDTD cell database (normal incidence) at the
  exact (kappa2, kappa3, h, eps_r) of the design, with physical optics over the 30 x 30 layout; oblique incidence uses
  the circuit's angular ratio Gamma(theta)/Gamma(0) in both models;
* discrepancy model delta = J_HF - J_LF (Eq. 18) learned by an ensemble (Gaussian process, gradient-boosted trees,
  3 neural networks), mean and spread as in Eq. 19; trust region, candidates and EI + diversity acquisition (Eq. 20),
  Algorithm 1; surrogate-free baselines (random search, real-coded GA, PSO) on the same HF model.
"""
from __future__ import annotations

import csv
import glob
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.stats import norm

from metasurface_optimizer.solvers.ring_disc_array import CIRCUIT, FIG2_F_GHZ, FIG2_PHASE_DEG, FIG2_R1_MM, circuit_gamma

NAMES = ["f1", "f2", "f3", "f4", "kappa2", "kappa3", "h"]
UNITS = ["GHz", "GHz", "GHz", "GHz", "", "", "mm"]
X0 = np.array([7.2, 8.0, 9.0, 10.8, 0.92, 0.80, 3.0])
LO = np.array([6.5, 7.3, 8.2, 9.5, 0.86, 0.66, 3.0])
HI = np.array([8.0, 9.0, 10.5, 12.5, 0.933, 0.863, 3.0])  # h fixed at the published 3.0 mm (cell database complete for h = 3.0 only)
SPAN = np.where(HI > LO, HI - LO, 1.0)
H_SET = np.array([3.0])
EPS_SET = (4.2, 4.4, 4.6)
THETA_SET = (0.0, 10.0, 20.0, 30.0)
POLS = ("TE", "TM")
F_BAND = np.round(np.arange(7.0, 12.0001, 0.1), 3)
W_MIN, G_MIN, DF_MIN, R1_MIN, R1_MAX, PERIOD = 0.10, 0.10, 0.3, 1.5, 4.5, 10.0
SIGMA_DELTA = np.radians(10.0)
N_SIDE = 30


# ------------------------------------------------------------------ design space
def snap(x: np.ndarray) -> np.ndarray:
    x = np.array(x, float)
    x[..., 6] = H_SET[np.argmin(np.abs(np.asarray(x[..., 6])[..., None] - H_SET), axis=-1)]
    return np.clip(x, LO, HI)


def to_u(x):
    return (np.asarray(x, float) - LO) / SPAN


def from_u(u):
    return snap(LO + np.clip(u, 0, 1) * (HI - LO))


def constraints(x: np.ndarray) -> np.ndarray:
    """g_j(x) <= 0 (Eq. 12) plus the database validity range of the ring-disc gap ratio."""
    f, k2, k3 = x[:4], x[4], x[5]
    return np.array([W_MIN - (1 - k2) * R1_MIN, G_MIN - (k2 - k3) * R1_MIN, G_MIN - (PERIOD - 2 * R1_MAX),
                     *(f[:3] - f[1:] + DF_MIN), 0.07 - (k2 - k3), (k2 - k3) - 0.20])


def feasible(x) -> bool:
    return bool(np.all(constraints(x) <= 1e-12))


# ------------------------------------------------------------------ cell models
class HFCells:
    """openEMS database Gamma(h, eps, kappa2, gap, R1, f) at normal incidence (``solvers/openems_cell_grid.py``)."""

    def __init__(self, files: Sequence[str]):
        rows = []
        for fn in files:
            with open(fn) as fh:
                rows += [tuple(float(r[k]) for k in ("h", "eps", "k2", "gap", "r1_mm", "f_ghz", "re", "im")) for r in csv.DictReader(fh)]
        a = np.array(rows)
        a = a[np.isin(np.round(a[:, 0], 6), np.round(H_SET, 6))]  # rows of thicknesses outside the design set are ignored
        self.axes = [np.unique(np.round(a[:, i], 6)) for i in range(6)]
        shape = [len(v) for v in self.axes]
        G = np.full(shape, np.nan + 0j)
        idx = [np.searchsorted(self.axes[i], np.round(a[:, i], 6)) for i in range(6)]
        G[tuple(idx)] = a[:, 6] + 1j * a[:, 7]
        if np.isnan(G).any():
            raise ValueError(f"cell database incomplete: {int(np.isnan(G).sum())} of {G.size} values missing")
        self.G = G
        self.h, self.eps, self.k2, self.gap, self.r1, self.f = self.axes
        self.fsel = np.array([int(np.argmin(np.abs(self.f - fb))) for fb in F_BAND])
        self._cache: Dict[tuple, np.ndarray] = {}

    def table(self, k2: float, k3: float, h: float, eps: float) -> np.ndarray:
        """Gamma on (R1 grid, band frequencies) for one cell family; linear in (kappa2, gap)."""
        key = (round(k2, 6), round(k3, 6), round(h, 6), round(eps, 6))
        if key not in self._cache:
            ih, ie = int(np.argmin(np.abs(self.h - h))), int(np.argmin(np.abs(self.eps - eps)))
            sub = self.G[ih, ie][:, :, :, self.fsel]  # (k2, gap, r1, f)
            pt = np.array([[k2, k2 - k3]])
            re = RegularGridInterpolator((self.k2, self.gap), sub.real, bounds_error=False, fill_value=None)(pt)[0]
            im = RegularGridInterpolator((self.k2, self.gap), sub.imag, bounds_error=False, fill_value=None)(pt)[0]
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[key] = re + 1j * im
        return self._cache[key]


@dataclass
class LFCells:
    """Equivalent-circuit cell (Eqs. 3-6) with fitted coefficients beta."""

    beta: Dict[str, float] = field(default_factory=lambda: dict(CIRCUIT))
    r1: np.ndarray = field(default_factory=lambda: np.linspace(R1_MIN, R1_MAX, 61))

    def gamma(self, f_ghz, r1, k2, k3, h, eps, theta=0.0, pol="TE"):
        return circuit_gamma(np.asarray(f_ghz) * 1e9, r1, PERIOD, h, eps, 0.02, k2, k3, theta, pol, self.beta)

    def table(self, k2, k3, h, eps):
        return self.gamma(F_BAND, self.r1, k2, k3, h, eps)

    def ratio(self, r1, k2, k3, h, eps, theta, pol):
        if theta == 0.0:
            return np.ones((len(r1), len(F_BAND)))
        return self.gamma(F_BAND, r1, k2, k3, h, eps, theta, pol) / self.gamma(F_BAND, r1, k2, k3, h, eps, 0.0, pol)

    def fit(self, r1, f_ghz, gamma_ref, k2=0.92, k3=0.80, h=3.0, eps=4.4, starts: int = 40, seed: int = 0) -> Dict[str, float]:
        """Eq. 17: least squares on |exp(j phi_ref) - exp(j phi(beta))| over the given (R1, f) data."""
        target = np.exp(1j * np.angle(gamma_ref))
        keys = ["ring_gap_scale", "ring_l", "disc_l", "disc_c"]

        def res(b):
            p = dict(zip(keys, b), r_s=0.0)
            g = circuit_gamma(np.asarray(f_ghz) * 1e9, r1, PERIOD, h, eps, 0.02, k2, k3, 0.0, "TE", p)
            d = target - np.exp(1j * np.angle(g))
            return np.concatenate([d.real.ravel(), d.imag.ravel()])

        rng = np.random.default_rng(seed)
        best = None
        lo, hi = [0.3, 0.01, 0.01, 0.002], [1.25, 10, 10, 10]
        for s in range(starts):
            b0 = [CIRCUIT[k] for k in keys] if s == 0 else [rng.uniform(l, u) for l, u in zip(lo, hi)]
            try:
                r = least_squares(res, b0, bounds=(lo, hi))
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        self.beta = dict(zip(keys, best.x), r_s=0.0)
        g = circuit_gamma(np.asarray(f_ghz) * 1e9, r1, PERIOD, h, eps, 0.02, k2, k3, 0.0, "TE", self.beta)
        err = np.degrees(np.abs(np.angle(g * np.conj(gamma_ref))))
        return {"beta": self.beta, "rms_phase_err_deg": float(np.sqrt(np.mean(err ** 2))), "max_phase_err_deg": float(err.max())}


# ------------------------------------------------------------------ array model
def _weights(grid: np.ndarray, x: np.ndarray):
    """Linear-interpolation indices and weights of x on grid (clamped)."""
    x = np.clip(x, grid[0], grid[-1])
    i = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2)
    w = (x - grid[i]) / (grid[i + 1] - grid[i])
    return i, w


def _apply(tab: np.ndarray, iw) -> np.ndarray:
    i, w = iw
    return tab[i] * (1 - w)[:, None] + tab[i + 1] * w[:, None]


class ArrayModel:
    """rho(f, theta, pol, eps; x) for the 30 x 30 random-phase layout (Eqs. 7-9, 11) with a given cell model."""

    def __init__(self, cells, lf: LFCells, seed: int = 1, sigma_delta: float = SIGMA_DELTA, thetas=THETA_SET, eps_set=EPS_SET,
                 r1_grid: Optional[np.ndarray] = None, n_side: int = N_SIDE):
        self.cells, self.lf, self.sigma = cells, lf, sigma_delta
        self.thetas, self.eps_set = tuple(thetas), tuple(eps_set)
        rng = np.random.default_rng(seed)
        self.phi_star = rng.uniform(0, 2 * np.pi, (n_side, n_side))
        self.sub = 2 * (np.arange(n_side)[:, None] % 2) + (np.arange(n_side)[None, :] % 2)
        self.r1_grid = r1_grid if r1_grid is not None else getattr(cells, "r1", lf.r1)
        self.fine = np.linspace(R1_MIN, R1_MAX, 301)

    def _interp_r1(self, tab: np.ndarray, r1_cells: np.ndarray) -> np.ndarray:
        """Per-cell Gamma (n_cells, F) by linear interpolation of Re/Im of tab (R1 grid, F)."""
        out = np.empty((len(r1_cells), tab.shape[1]), complex)
        for j in range(tab.shape[1]):
            out[:, j] = np.interp(r1_cells, self.r1_grid, tab[:, j].real) + 1j * np.interp(r1_cells, self.r1_grid, tab[:, j].imag)
        return out

    def layout(self, x: np.ndarray) -> np.ndarray:
        """Eq. 11 with the cell model's own phase curve at nominal eps_r = 4.4."""
        f, k2, k3, h = x[:4], x[4], x[5], x[6]
        tab = self.cells.table(k2, k3, h, 4.4)
        R1 = np.empty(self.phi_star.shape)
        for s in range(4):
            j = int(np.argmin(np.abs(F_BAND - f[s])))
            g = np.interp(self.fine, self.r1_grid, tab[:, j].real) + 1j * np.interp(self.fine, self.r1_grid, tab[:, j].imag)
            ph = np.exp(1j * np.angle(g))
            m = self.sub == s
            d = np.abs(ph[None, :] - np.exp(1j * self.phi_star[m])[:, None])
            R1[m] = self.fine[np.argmin(d, axis=1)]
        return R1

    def rho(self, x: np.ndarray, sigma: Optional[float] = None, eps_set=None, thetas=None, R1=None) -> np.ndarray:
        """Specular reduction (dB, positive = less than the plate), shape (eps, theta, pol, F)."""
        sigma = self.sigma if sigma is None else sigma
        eps_set = eps_set or self.eps_set
        thetas = thetas or self.thetas
        x = np.asarray(x, float)
        k2, k3, h = x[4], x[5], x[6]
        R1 = self.layout(x) if R1 is None else R1
        r1c = R1.ravel()
        N = r1c.size
        wh = _weights(self.r1_grid, r1c)  # same radii for every scenario: interpolation weights computed once
        wl = _weights(self.lf.r1, r1c)
        out = np.empty((len(eps_set), len(thetas), len(POLS), len(F_BAND)))
        for ie, eps in enumerate(eps_set):
            G0 = _apply(self.cells.table(k2, k3, h, eps), wh)
            for it, th in enumerate(thetas):
                for ip, pol in enumerate(POLS):
                    G = G0 if th == 0.0 else G0 * _apply(self.lf.ratio(self.lf.r1, k2, k3, h, eps, th, pol), wl)
                    coh = np.abs(G.sum(0)) ** 2
                    inc = (np.abs(G) ** 2).sum(0)
                    E = np.exp(-sigma ** 2) * coh + (1 - np.exp(-sigma ** 2)) * inc
                    out[ie, it, ip] = -10 * np.log10(E / N ** 2)
        return out

    def _ratio_cells(self, r1c, k2, k3, h, eps, th, pol):
        tab = self.lf.ratio(self.lf.r1, k2, k3, h, eps, th, pol)
        out = np.empty((len(r1c), tab.shape[1]), complex)
        for j in range(tab.shape[1]):
            out[:, j] = np.interp(r1c, self.lf.r1, tab[:, j].real) + 1j * np.interp(r1c, self.lf.r1, tab[:, j].imag)
        return out


def softmin(v: np.ndarray, T: float = 0.5) -> float:
    v = np.asarray(v, float).ravel()
    m = v.min()
    return float(m - T * np.log(np.sum(np.exp(-(v - m) / T))))


@dataclass
class Objective:
    """J = -rho_wc + lambda_cf ||w (x - x0)||_1 (Eq. 14); variants for the ablation."""

    model: ArrayModel
    lam_cf: float = 1.0
    robust: bool = True
    sigma: Optional[float] = None  # None: the model's sigma_delta
    T: float = 0.5

    def scenarios(self, x):
        if self.robust:
            return self.model.rho(x, sigma=self.sigma)
        return self.model.rho(x, sigma=self.sigma, eps_set=(4.4,), thetas=(0.0,))

    def penalty(self, x):
        return self.lam_cf * float(np.sum(np.abs(to_u(x) - to_u(X0))))

    def __call__(self, x) -> Dict[str, float]:
        r = self.scenarios(x)
        return {"rho_wc": float(r.min()), "J": -float(r.min()) + self.penalty(x), "J_soft": -softmin(r, self.T) + self.penalty(x)}


def report(model: ArrayModel, x) -> Dict[str, object]:
    """Metrics reported in the paper, always with the full robust objective and the phase-error term."""
    r = model.rho(x)
    nominal = r[EPS_SET.index(4.4), 0, 0]
    ok = nominal >= 10
    runs, best = [], (0, 0, -1)
    i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if j - i > best[1] - best[0] or best[2] < 0:
                best = (i, j, 1)
            i = j + 1
        else:
            i += 1
    band = (float(F_BAND[best[0]]), float(F_BAND[best[1]])) if best[2] > 0 else None
    fbw = 2 * (band[1] - band[0]) / (band[1] + band[0]) * 100 if band else 0.0
    return {"rho_wc": float(r.min()), "rho_min_normal": float(nominal.min()), "rho_mean_normal": float(nominal.mean()),
            "band_10dB_GHz": band, "fbw_pct": fbw, "fraction_band_10dB": float(ok.mean()),
            "worst_scenario": dict(zip(["eps", "theta", "pol", "f_GHz"], [EPS_SET[int(np.unravel_index(r.argmin(), r.shape)[0])],
                                                                         THETA_SET[int(np.unravel_index(r.argmin(), r.shape)[1])],
                                                                         POLS[int(np.unravel_index(r.argmin(), r.shape)[2])],
                                                                         float(F_BAND[int(np.unravel_index(r.argmin(), r.shape)[3])])]))}


# ------------------------------------------------------------------ surrogate (discrepancy ensemble, Eq. 19)
class DiscrepancyEnsemble:
    def __init__(self, seed: int = 0, n_nets: int = 3):
        self.seed, self.n_nets = seed, n_nets
        self.members = []

    def fit(self, U: np.ndarray, d: np.ndarray) -> "DiscrepancyEnsemble":
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.neural_network import MLPRegressor

        self.mu_d, self.sd_d = float(d.mean()), float(d.std() + 1e-9)
        y = (d - self.mu_d) / self.sd_d
        k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(U.shape[1], 0.3), length_scale_bounds=(1e-2, 1e2), nu=2.5) + \
            WhiteKernel(1e-2, (1e-6, 1e0))
        gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2, random_state=self.seed).fit(U, y)
        gbt = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=self.seed).fit(U, y)
        nets = [MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", solver="adam", alpha=1e-3, learning_rate_init=3e-3,
                             max_iter=3000, random_state=self.seed * 10 + i).fit(U, y) for i in range(self.n_nets)]
        self.members = [gp, gbt] + nets
        return self

    def predict(self, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = np.array([m.predict(U) for m in self.members]) * self.sd_d + self.mu_d
        return P.mean(0), P.std(0)


# ------------------------------------------------------------------ optimizer (Algorithm 1)
@dataclass
class CFConfig:
    n_max: int = 40
    n_init: int = 6
    batch: int = 4
    r0: float = 0.10
    r_max: float = 0.5
    r_min: float = 0.02
    gamma: float = 0.05
    n_random: int = 60
    physics_prior: bool = True
    uncertainty: bool = True
    seed: int = 0


class CounterfactualOptimizer:
    def __init__(self, hf_obj: Objective, lf_obj: Objective, cfg: CFConfig, report_model: Optional[ArrayModel] = None):
        self.hf, self.lf, self.cfg = hf_obj, lf_obj, cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.X: List[np.ndarray] = []
        self.Y: List[Dict[str, float]] = []
        self.LF: List[Dict[str, float]] = []
        self.pred_log: List[Dict[str, float]] = []
        self.curve: List[Dict[str, float]] = []
        self.report_model = report_model

    def _lf(self, x):
        return self.lf(x) if self.cfg.physics_prior else {"J": 0.0, "J_soft": 0.0, "rho_wc": 0.0}

    def evaluate(self, x):
        x = snap(x)
        y = self.hf(x)
        self.X.append(x); self.Y.append(y); self.LF.append(self._lf(x))
        best = min(self.Y, key=lambda v: v["J"])
        self.curve.append({"n_hf": len(self.Y), "best_J": best["J"], "best_rho_wc": best["rho_wc"]})
        return y

    def _sample_tr(self, c, r, n):
        u = to_u(c) + self.rng.uniform(-1, 1, (n, 7)) * r
        return [from_u(v) for v in u]

    def _predict(self, cands):
        U = np.array([to_u(x) for x in cands])
        lf = np.array([self._lf(x)["J_soft"] for x in cands])
        if self.model is None:
            return lf, np.zeros(len(cands))
        d, s = self.model.predict(U)
        return lf + d, s

    def run(self) -> Dict[str, object]:
        cfg = self.cfg
        t0 = time.time()
        self.model = None
        self.evaluate(X0)  # baseline reproduction (counts as one HF evaluation)
        c, r, fails = X0.copy(), np.full(7, cfg.r0), 0
        init = [x for x in self._sample_tr(c, r, 50) if feasible(x)][: cfg.n_init]
        for x in init:
            self.evaluate(x)
        it = 0
        while len(self.Y) < cfg.n_max:
            it += 1
            U = np.array([to_u(x) for x in self.X])
            d = np.array([y["J_soft"] - l["J_soft"] for y, l in zip(self.Y, self.LF)])
            self.model = DiscrepancyEnsemble(seed=cfg.seed + it).fit(U, d)
            # sensitivities of the surrogate mean at the incumbent (central differences in u)
            base_mu = self._predict([c])[0][0]
            sens = np.zeros(7)
            for i in range(7):
                e = np.zeros(7); e[i] = 0.05
                sens[i] = (self._predict([from_u(to_u(c) + e)])[0][0] - self._predict([from_u(to_u(c) - e)])[0][0]) / 0.1
            top = np.argsort(-np.abs(sens))[:3]
            cands = []
            for i in top:  # single-parameter steps
                for a in (0.25, 0.5, 1.0):
                    for sg in (-1, 1):
                        e = np.zeros(7); e[i] = sg * a * r[i]
                        cands.append(from_u(to_u(c) + e))
            for i in range(3):  # two-parameter steps
                for j in range(i + 1, 3):
                    for si in (-1, 1):
                        for sj in (-1, 1):
                            e = np.zeros(7); e[top[i]] = si * 0.5 * r[top[i]]; e[top[j]] = sj * 0.5 * r[top[j]]
                            cands.append(from_u(to_u(c) + e))
            g = -sens / (np.linalg.norm(sens) + 1e-12)  # surrogate descent direction
            for t in (0.25, 0.5, 1.0):
                cands.append(from_u(to_u(c) + t * g * r))
            cands += self._sample_tr(c, r, cfg.n_random)
            seen = {tuple(np.round(x, 6)) for x in self.X}
            uniq, keys = [], set()
            for x in cands:
                k = tuple(np.round(x, 6))
                if k in seen or k in keys or not feasible(x):
                    continue
                keys.add(k); uniq.append(x)
            if not uniq:
                r = np.maximum(r * 0.5, cfg.r_min)
                continue
            mu, s = self._predict(uniq)
            if not cfg.uncertainty:
                s = np.zeros_like(s)
            jstar = min(y["J_soft"] for y in self.Y)
            chosen = []
            ref = [to_u(x) for x in self.X]
            for _ in range(min(cfg.batch, cfg.n_max - len(self.Y), len(uniq))):
                Uc = np.array([to_u(x) for x in uniq])
                D = np.min(np.linalg.norm(Uc[:, None, :] - np.array(ref)[None], axis=2), axis=1) / np.sqrt(7)
                if cfg.uncertainty:
                    z = (jstar - mu) / np.maximum(s, 1e-9)
                    ei = (jstar - mu) * norm.cdf(z) + s * norm.pdf(z)
                    a = ei + cfg.gamma * D
                else:
                    a = -mu + cfg.gamma * D
                k = int(np.argmax(a))
                chosen.append((uniq[k], float(mu[k]), float(s[k])))
                ref.append(Uc[k])
                uniq.pop(k); mu = np.delete(mu, k); s = np.delete(s, k)
            best_before = min(y["J"] for y in self.Y)
            for x, m, sd in chosen:
                y = self.evaluate(x)
                lfv = self.LF[-1]
                self.pred_log.append({"n_hf": len(self.Y), "iteration": it, "pred_J_soft": m, "sd": sd, "hf_J_soft": y["J_soft"],
                                      "abs_err_dB": abs(m - y["J_soft"]), "lf_abs_err_dB": abs(lfv["J_soft"] - y["J_soft"]) if cfg.physics_prior else None})
            best_now = min(range(len(self.Y)), key=lambda i: self.Y[i]["J"])
            if self.Y[best_now]["J"] < best_before - 1e-3 * abs(best_before):
                c = self.X[best_now]; r = np.minimum(r * 1.5, cfg.r_max); fails = 0
            else:
                fails += 1
                if fails >= 2:
                    r = np.maximum(r * 0.5, cfg.r_min); fails = 0
        b = min(range(len(self.Y)), key=lambda i: self.Y[i]["J"])
        return {"x_best": self.X[b].tolist(), "y_best": self.Y[b], "curve": self.curve, "pred_log": self.pred_log, "n_hf": len(self.Y),
                "time_s": time.time() - t0, "X": [x.tolist() for x in self.X], "Y": self.Y}


# ------------------------------------------------------------------ surrogate-free baselines (same HF model)
def _feasible_random(rng, n):
    out = []
    while len(out) < n:
        x = from_u(rng.uniform(0, 1, 7))
        if feasible(x):
            out.append(x)
    return out


def _track(obj, budget):
    X, Y, curve = [], [], []

    def ev(x):
        x = snap(x)
        if not feasible(x):  # infeasible proposals are rejected without consuming budget
            return None
        y = obj(x)
        X.append(x); Y.append(y)
        b = min(Y, key=lambda v: v["J"])
        curve.append({"n_hf": len(Y), "best_J": b["J"], "best_rho_wc": b["rho_wc"]})
        return y
    return X, Y, curve, ev


def random_search(obj, budget, seed):
    rng = np.random.default_rng(seed)
    X, Y, curve, ev = _track(obj, budget)
    ev(X0)
    for x in _feasible_random(rng, budget - 1):
        ev(x)
    b = int(np.argmin([y["J"] for y in Y]))
    return {"x_best": X[b].tolist(), "y_best": Y[b], "curve": curve, "n_hf": len(Y)}


def genetic_algorithm(obj, budget, seed, pop=8, eta_c=15.0, eta_m=20.0):
    rng = np.random.default_rng(seed)
    X, Y, curve, ev = _track(obj, budget)
    P = [X0.copy()] + _feasible_random(rng, pop - 1)
    F = [ev(x)["J"] for x in P]
    while len(Y) < budget:
        kids = []
        while len(kids) < pop and len(Y) + len(kids) < budget * 3:
            a, b = [min(rng.choice(pop, 2, replace=False), key=lambda i: F[i]) for _ in range(2)]
            u1, u2 = to_u(P[a]), to_u(P[b])
            mu = rng.random(7)
            beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (eta_c + 1)), (1 / (2 * (1 - mu))) ** (1 / (eta_c + 1)))
            ch = 0.5 * ((1 + beta) * u1 + (1 - beta) * u2)
            m = rng.random(7) < 1 / 7
            dl = rng.random(7)
            dq = np.where(dl < 0.5, (2 * dl) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - dl)) ** (1 / (eta_m + 1)))
            ch = np.clip(ch + m * dq, 0, 1)
            x = from_u(ch)
            if feasible(x):
                kids.append(x)
        for x in kids:
            if len(Y) >= budget:
                break
            y = ev(x)
            P.append(x); F.append(y["J"])
        order = np.argsort(F)[:pop]  # elitist (mu + lambda) survival
        P, F = [P[i] for i in order], [F[i] for i in order]
    b = int(np.argmin([y["J"] for y in Y]))
    return {"x_best": X[b].tolist(), "y_best": Y[b], "curve": curve, "n_hf": len(Y)}


def particle_swarm(obj, budget, seed, n=8, w=0.7, c1=1.5, c2=1.5):
    rng = np.random.default_rng(seed)
    X, Y, curve, ev = _track(obj, budget)
    P = [to_u(X0)] + [to_u(x) for x in _feasible_random(rng, n - 1)]
    V = [rng.uniform(-0.1, 0.1, 7) for _ in range(n)]
    pb = [p.copy() for p in P]
    pf = [ev(from_u(p))["J"] for p in P]
    g = pb[int(np.argmin(pf))].copy(); gf = min(pf)
    tries = 0
    while len(Y) < budget and tries < budget * 50:
        for i in range(n):
            if len(Y) >= budget:
                break
            V[i] = w * V[i] + c1 * rng.random(7) * (pb[i] - P[i]) + c2 * rng.random(7) * (g - P[i])
            P[i] = np.clip(P[i] + V[i], 0, 1)
            tries += 1
            y = ev(from_u(P[i]))
            if y is None:
                continue
            if y["J"] < pf[i]:
                pb[i], pf[i] = P[i].copy(), y["J"]
            if y["J"] < gf:
                g, gf = P[i].copy(), y["J"]
    b = int(np.argmin([y["J"] for y in Y]))
    return {"x_best": X[b].tolist(), "y_best": Y[b], "curve": curve, "n_hf": len(Y)}


def load_hf(grid_dir: str | Path) -> HFCells:
    return HFCells(sorted(glob.glob(str(Path(grid_dir) / "grid_w*.csv"))))
