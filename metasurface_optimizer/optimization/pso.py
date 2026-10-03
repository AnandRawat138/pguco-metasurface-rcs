"""Particle swarm optimisation (global-best, inertia weight), budget-aware, normalised coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from metasurface_optimizer.solvers.base import BudgetExceeded


@dataclass
class PSOConfig:
    n_particles: int = 10
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    v_max: float = 0.2
    max_iterations: int = 1000


def pso(f: Callable[[np.ndarray], np.ndarray], dim: int, cfg: PSOConfig, rng: np.random.Generator, init: Optional[np.ndarray] = None):
    X = rng.random((cfg.n_particles, dim))
    if init is not None:
        X[: len(init)] = init[: cfg.n_particles]
    V = rng.uniform(-cfg.v_max, cfg.v_max, X.shape)
    pbest, pval = X.copy(), np.full(cfg.n_particles, np.inf)
    gbest, gval = X[0].copy(), np.inf
    try:
        for it in range(cfg.max_iterations):
            for i in range(cfg.n_particles):
                v = f(X[i:i + 1])[0]
                if v < pval[i]:
                    pval[i], pbest[i] = v, X[i].copy()
                if v < gval:
                    gval, gbest = v, X[i].copy()
            r1, r2 = rng.random(X.shape), rng.random(X.shape)
            V = cfg.w * V + cfg.c1 * r1 * (pbest - X) + cfg.c2 * r2 * (gbest - X)
            V = np.clip(V, -cfg.v_max, cfg.v_max)
            X = np.clip(X + V, 0, 1)
    except BudgetExceeded:
        pass
    return gbest, gval
