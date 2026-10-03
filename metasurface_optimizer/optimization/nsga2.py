"""NSGA-II (Deb et al., IEEE TEC 2002) with constraint domination.

Operates on normalised coordinates z in [0,1]^d. Uses simulated binary
crossover (SBX, eta_c) and polynomial mutation (eta_m). Used (i) on the
surrogate to approximate the predicted Pareto front inside the trust region,
and (ii) as a high-fidelity black-box baseline under the same N_HF budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from metasurface_optimizer.optimization.pareto import crowding_distance, fast_non_dominated_sort
from metasurface_optimizer.solvers.base import BudgetExceeded


@dataclass
class NSGA2Config:
    pop_size: int = 40
    generations: int = 40
    eta_c: float = 15.0
    eta_m: float = 20.0
    p_crossover: float = 0.9
    p_mutation: Optional[float] = None  # default 1/d


def _sbx(a, b, eta, rng):
    u = rng.random(a.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    swap = rng.random(a.shape) < 0.5
    return np.where(swap, c1, c2), np.where(swap, c2, c1)


def _poly_mut(x, eta, pm, rng, lo, hi):
    x = x.copy()
    mask = rng.random(x.shape) < pm
    u = rng.random(x.shape)
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    x = np.where(mask, x + delta * (hi - lo), x)
    return np.clip(x, lo, hi)


def _rank_crowd(F, CV):
    fronts = fast_non_dominated_sort(F, CV)
    rank = np.empty(len(F), int); crowd = np.empty(len(F))
    for r, f in enumerate(fronts):
        rank[f] = r
        crowd[f] = crowding_distance(F[f])
    return rank, crowd


def nsga2(f: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], dim: int, cfg: NSGA2Config, rng: np.random.Generator,
          lo: Optional[np.ndarray] = None, hi: Optional[np.ndarray] = None, init: Optional[np.ndarray] = None):
    """Minimise vector objective f(Z) -> (F, CV). Returns (Z, F, CV, history). Stops early on BudgetExceeded."""
    lo = np.zeros(dim) if lo is None else lo
    hi = np.ones(dim) if hi is None else hi
    pm = cfg.p_mutation or 1.0 / dim
    Z = lo + rng.random((cfg.pop_size, dim)) * (hi - lo)
    if init is not None:
        Z[: len(init)] = np.clip(init[: cfg.pop_size], lo, hi)
    history = []
    try:
        F, CV = f(Z)
    except BudgetExceeded:
        return Z[:0], np.empty((0, 0)), np.empty(0), history
    for gen in range(cfg.generations):
        rank, crowd = _rank_crowd(F, CV)
        # binary tournament
        def pick():
            i, j = rng.integers(0, len(Z), 2)
            if rank[i] != rank[j]:
                return i if rank[i] < rank[j] else j
            return i if crowd[i] > crowd[j] else j
        kids = []
        while len(kids) < cfg.pop_size:
            a, b = Z[pick()], Z[pick()]
            if rng.random() < cfg.p_crossover:
                a, b = _sbx(a, b, cfg.eta_c, rng)
            kids += [_poly_mut(a, cfg.eta_m, pm, rng, lo, hi), _poly_mut(b, cfg.eta_m, pm, rng, lo, hi)]
        Q = np.array(kids[: cfg.pop_size])
        try:
            FQ, CVQ = f(Q)
        except BudgetExceeded:
            break
        RZ, RF, RCV = np.vstack([Z, Q]), np.vstack([F, FQ]), np.concatenate([CV, CVQ])
        fronts = fast_non_dominated_sort(RF, RCV)
        sel = []
        for fr in fronts:
            if len(sel) + len(fr) <= cfg.pop_size:
                sel += list(fr)
            else:
                cd = crowding_distance(RF[fr])
                sel += list(fr[np.argsort(-cd)[: cfg.pop_size - len(sel)]])
                break
        sel = np.array(sel)
        Z, F, CV = RZ[sel], RF[sel], RCV[sel]
        history.append({"generation": gen, "n_front0": int((_rank_crowd(F, CV)[0] == 0).sum())})
    return Z, F, CV, history
