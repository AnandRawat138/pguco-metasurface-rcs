"""Acquisition functions for minimisation (Eq. A1-A4).

Let mu(x), sigma(x) be the surrogate mean / std of the composite objective
J(x), J* the best *high-fidelity* composite value observed so far, xi >= 0 an
improvement margin, and D(x) the normalised distance from x to the nearest
already-evaluated (or already-selected) design.

* A1  EI(x)        = (J* - mu - xi) Phi(z) + sigma phi(z),   z = (J* - mu - xi) / sigma
* A2  proposed     A(x) = EI(x) + beta sigma(x) + gamma D(x)
* A3  LCB          select argmin  mu(x) - kappa sigma(x)   (score = -(mu - kappa sigma))
* A4  mean         score = -mu(x)      (no uncertainty; ablation A5)
* A5  random       score ~ U(0, 1)     (random candidate selection; ablation A7)

D(x) = min_j ||u(x) - u(x_j)||_2 / sqrt(d) with u the unit-cube coordinates of
the free parameters. Batches of K are selected greedily: after each pick the
picked design joins the reference set of D (so the batch spreads out), which
is a deterministic, documented batch heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm


@dataclass
class AcquisitionConfig:
    kind: str = "ei_sigma_diversity"  # ei_sigma_diversity | ei | lcb | mean | random
    beta: float = 0.0
    gamma: float = 0.05
    kappa: float = 1.0
    xi: float = 0.0


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def diversity(U: np.ndarray, U_ref: np.ndarray) -> np.ndarray:
    if U_ref is None or len(U_ref) == 0:
        return np.ones(len(U))
    d = np.sqrt(((U[:, None, :] - U_ref[None, :, :]) ** 2).sum(-1)).min(1)
    return d / np.sqrt(U.shape[1])


def acquisition_scores(cfg: AcquisitionConfig, mu: np.ndarray, sigma: np.ndarray, best: float, U: np.ndarray, U_ref: np.ndarray,
                       rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """Return the total score (higher = better) and its components."""
    D = diversity(U, U_ref)
    if cfg.kind == "random":
        r = (rng or np.random.default_rng(0)).random(len(mu))
        return {"score": r, "D": D}
    if cfg.kind == "mean":
        return {"score": -mu, "D": D}
    if cfg.kind == "lcb":
        return {"score": -(mu - cfg.kappa * sigma), "D": D}
    ei = expected_improvement(mu, sigma, best, cfg.xi)
    if cfg.kind == "ei":
        return {"score": ei, "EI": ei, "D": D}
    return {"score": ei + cfg.beta * sigma + cfg.gamma * D, "EI": ei, "beta_sigma": cfg.beta * sigma, "gamma_D": cfg.gamma * D, "D": D}


def select_batch(cfg: AcquisitionConfig, mu: np.ndarray, sigma: np.ndarray, best: float, U: np.ndarray, U_ref: np.ndarray, k: int,
                 rng: Optional[np.random.Generator] = None, priority: Optional[np.ndarray] = None) -> List[Dict]:
    """Greedy batch selection. ``priority`` (lower = better, e.g. Pareto rank) is applied lexicographically before the score."""
    chosen: List[Dict] = []
    ref = np.array(U_ref) if U_ref is not None and len(U_ref) else np.empty((0, U.shape[1]))
    avail = np.ones(len(mu), bool)
    rng = rng or np.random.default_rng(0)
    rand = rng.random(len(mu)) if cfg.kind == "random" else None
    for _ in range(min(k, len(mu))):
        if cfg.kind == "random":
            comp = {"score": rand, "D": diversity(U, ref)}
        else:
            comp = acquisition_scores(cfg, mu, sigma, best, U, ref, rng)
        score = np.where(avail, comp["score"], -np.inf)
        if priority is not None:
            pr = np.where(avail, priority, np.inf)
            score = np.where(pr == pr.min(), score, -np.inf)
        i = int(np.argmax(score))
        chosen.append({"index": i, **{k_: float(v[i]) for k_, v in comp.items()}})
        avail[i] = False
        ref = np.vstack([ref, U[i:i + 1]])
    return chosen
