"""Differential evolution DE/rand/1/bin (Storn & Price 1997), budget-aware, normalised coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from metasurface_optimizer.solvers.base import BudgetExceeded


@dataclass
class DEConfig:
    pop_size: int = 10
    F: float = 0.6
    CR: float = 0.9
    max_generations: int = 1000


def differential_evolution(f: Callable[[np.ndarray], np.ndarray], dim: int, cfg: DEConfig, rng: np.random.Generator,
                           init: Optional[np.ndarray] = None):
    P = rng.random((cfg.pop_size, dim))
    if init is not None:
        P[: len(init)] = init[: cfg.pop_size]
    fit = np.full(cfg.pop_size, np.inf)
    try:
        for i in range(cfg.pop_size):
            fit[i] = f(P[i:i + 1])[0]
        for g in range(cfg.max_generations):
            for i in range(cfg.pop_size):
                a, b, c = P[rng.choice([j for j in range(cfg.pop_size) if j != i], 3, replace=False)]
                mutant = np.clip(a + cfg.F * (b - c), 0, 1)
                cross = rng.random(dim) < cfg.CR
                cross[rng.integers(dim)] = True
                trial = np.where(cross, mutant, P[i])
                ft = f(trial[None])[0]
                if ft <= fit[i]:
                    P[i], fit[i] = trial, ft
    except BudgetExceeded:
        pass
    k = int(np.argmin(fit))
    return P[k], fit[k]
