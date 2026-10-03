"""Standard Bayesian optimisation baseline: GP (Matern-5/2 ARD) on the scalar composite + EI.

Global (whole design box), no physics, no counterfactual structure, no
response block: the conventional black-box comparator. Initial design is a
Latin hypercube (including x0); EI is maximised over a random pool plus local
perturbations of the incumbent.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from metasurface_optimizer.optimization.acquisition import expected_improvement
from metasurface_optimizer.solvers.base import BudgetExceeded


@dataclass
class BOConfig:
    n_init: int = 10
    pool_size: int = 3000
    max_iterations: int = 1000
    xi: float = 0.0


def bayesian_optimization(f: Callable[[np.ndarray], np.ndarray], dim: int, cfg: BOConfig, rng: np.random.Generator,
                          z0: Optional[np.ndarray] = None):
    Z = qmc.LatinHypercube(d=dim, seed=rng).random(cfg.n_init)
    if z0 is not None:
        Z[0] = z0
    ys = []
    Zs = []
    try:
        for z in Z:
            ys.append(f(z[None])[0]); Zs.append(z)
        for it in range(cfg.max_iterations):
            Xa, ya = np.array(Zs), np.array(ys)
            ok = np.isfinite(ya)
            k = ConstantKernel(1.0) * Matern(np.ones(dim) * 0.3, (1e-2, 1e1), nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e-1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=int(rng.integers(1 << 30))).fit(Xa[ok], ya[ok])
            best = ya[ok].min()
            inc = Xa[ok][np.argmin(ya[ok])]
            pool = np.vstack([rng.random((cfg.pool_size, dim)), np.clip(inc + rng.normal(0, 0.05, (cfg.pool_size // 3, dim)), 0, 1)])
            mu, sd = gp.predict(pool, return_std=True)
            z = pool[np.argmax(expected_improvement(mu, sd, best, cfg.xi))]
            ys.append(f(z[None])[0]); Zs.append(z)
    except BudgetExceeded:
        pass
    ya = np.array(ys)
    k = int(np.nanargmin(ya)) if len(ya) else 0
    return (np.array(Zs)[k] if Zs else None), (ya[k] if len(ya) else np.inf)
