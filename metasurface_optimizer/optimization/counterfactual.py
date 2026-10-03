"""Counterfactual candidate generation C (Eq. CF1-CF6).

Given the incumbent design c (initially x0), trust-region radii r (normalised
units) and local sensitivities, generate controlled perturbations
x' = c + Delta x subject to bounds, manufacturing resolution and geometry
constraints:

* CF1 single-variable:  c + a delta_i e_i,  a in A (step multipliers, +/-), i in top-k sensitive parameters
* CF2 pairwise:         c + a delta_i e_i + b delta_j e_j,  (i, j) pairs of the top-k
* CF3 correlated:       c + t v / ||v||  along schema-declared physical directions v
                        (e.g. "scale both patch states") and along the descent
                        direction v = -grad J (from sensitivities)
* CF4 latent:           c + t w_k along principal directions w_k of the evaluated
                        designs' normalised vectors (a linear latent space);
                        ``latent_fn`` can supply directions from a learned encoder
* CF5 trust-region exploration: uniform samples in the box |u - u_c| <= r
* CF6 surrogate-Pareto:  points proposed by NSGA-II run on the surrogate (added
                        by the optimiser, not by this module)

Every candidate is snapped to the manufacturing grid, clipped to bounds and
the trust region, de-duplicated, and rejected if any geometry constraint is
violated. Each keeps a record of what was changed and why.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from metasurface_optimizer.data.dataset import design_hash
from metasurface_optimizer.geometry.parameter_schema import ParameterSchema


@dataclass
class Candidate:
    x: np.ndarray
    kind: str
    parent: np.ndarray
    detail: str = ""
    meta: Dict = field(default_factory=dict)

    def modifications(self, schema: ParameterSchema, tol: float = 1e-9) -> List[Dict]:
        out = []
        for i, p in enumerate(schema.parameters):
            if abs(self.x[i] - self.parent[i]) > tol:
                out.append({"parameter": p.name, "original": float(self.parent[i]), "new": float(self.x[i]), "unit": p.unit,
                            "change": float(self.x[i] - self.parent[i])})
        return out


@dataclass
class CounterfactualConfig:
    top_k: int = 5
    step_multipliers: Sequence[float] = (0.5, 1.0, 2.0, 3.0)
    pairwise: bool = True
    pair_multipliers: Sequence[float] = (1.0, 2.0)
    correlated: bool = True
    correlated_steps: Sequence[float] = (-0.15, -0.08, -0.04, 0.04, 0.08, 0.15)  # in normalised units
    gradient_steps: Sequence[float] = (0.02, 0.05, 0.1, 0.2)
    latent: bool = True
    latent_components: int = 3
    n_random: int = 300
    max_pool: int = 3000


class CounterfactualGenerator:
    def __init__(self, schema: ParameterSchema, feasible: Callable[[np.ndarray], bool], cfg: Optional[CounterfactualConfig] = None,
                 latent_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.schema = schema
        self.feasible = feasible
        self.cfg = cfg or CounterfactualConfig()
        self.latent_fn = latent_fn
        self.free = np.where(schema.free_mask)[0]
        self.stats: Dict[str, int] = {}

    # ----------------------------------------------------------- helpers
    def _project(self, x: np.ndarray, center: np.ndarray, radius: np.ndarray) -> np.ndarray:
        s = self.schema
        u, uc = s.normalize(x), s.normalize(center)
        u = np.clip(u, uc - radius, uc + radius)
        x = s.denormalize(np.clip(u, 0, 1))
        x[~s.free_mask] = center[~s.free_mask]
        return s.snap(x)

    def _ranked(self, sensitivity: Optional[np.ndarray]) -> List[int]:
        if sensitivity is None:
            return list(self.free)
        order = [i for i in np.argsort(-np.asarray(sensitivity)) if i in set(self.free)]
        return order

    # ----------------------------------------------------------- generation
    def generate(self, center: np.ndarray, radius: np.ndarray, rng: np.random.Generator,
                 sensitivity: Optional[np.ndarray] = None, gradient: Optional[np.ndarray] = None,
                 evaluated: Optional[np.ndarray] = None) -> List[Candidate]:
        """Return valid, de-duplicated candidates around ``center``.

        sensitivity: S_hat (d,) used to pick the top-k parameters.
        gradient:    dJ/dx (d,) of the composite objective (physical units) for CF3 descent moves.
        evaluated:   designs already evaluated (excluded; used for CF4 latent directions).
        radius:      trust-region half-widths in normalised units (d,).
        """
        s, cfg = self.schema, self.cfg
        center = np.asarray(center, float)
        radius = np.broadcast_to(np.asarray(radius, float), (s.dim,)).copy()
        top = self._ranked(sensitivity)[: cfg.top_k]
        steps = s.steps
        raw: List[Candidate] = []

        for i in top:  # CF1
            for a in cfg.step_multipliers:
                for sg in (+1, -1):
                    x = center.copy(); x[i] += sg * a * steps[i]
                    raw.append(Candidate(x, "single", center, f"{s.names[i]} {'+' if sg > 0 else '-'}{a:g} delta"))
        if cfg.pairwise:  # CF2
            for i, j in itertools.combinations(top, 2):
                for a, b in itertools.product(cfg.pair_multipliers, repeat=2):
                    for si, sj in itertools.product((1, -1), repeat=2):
                        x = center.copy(); x[i] += si * a * steps[i]; x[j] += sj * b * steps[j]
                        raw.append(Candidate(x, "pairwise", center, f"{s.names[i]} {si * a:+g}d, {s.names[j]} {sj * b:+g}d"))
        if cfg.correlated:  # CF3
            span = s.upper - s.lower
            for cd in s.correlated_directions:
                v = np.zeros(s.dim)
                for k, c in cd.direction.items():
                    v[s.index(k)] = c
                v = v / (np.linalg.norm(v) + 1e-12)
                for t in cfg.correlated_steps:
                    raw.append(Candidate(center + t * v * span, "correlated", center, f"{cd.name} t={t:+g}", {"direction": cd.name}))
            if gradient is not None and np.any(gradient):
                g = np.asarray(gradient, float) * span  # gradient in normalised coordinates
                g[~s.free_mask] = 0
                v = -g / (np.linalg.norm(g) + 1e-12)
                for t in cfg.gradient_steps:
                    raw.append(Candidate(center + t * v * span, "descent", center, f"-grad J direction, step {t:g} (normalised)"))
        if cfg.latent:  # CF4
            dirs = self._latent_dirs(evaluated)
            span = s.upper - s.lower
            for k, w in enumerate(dirs):
                for t in cfg.correlated_steps:
                    raw.append(Candidate(center + t * w * span, "latent", center, f"latent direction {k} t={t:+g}"))
        for _ in range(cfg.n_random):  # CF5
            u = s.normalize(center) + rng.uniform(-1, 1, s.dim) * radius
            raw.append(Candidate(s.denormalize(np.clip(u, 0, 1)), "trust_region_random", center, "exploration sample"))

        return self.finalize(raw, center, radius, evaluated)

    def _latent_dirs(self, evaluated: Optional[np.ndarray]) -> List[np.ndarray]:
        s = self.schema
        if self.latent_fn is not None:
            return list(self.latent_fn(evaluated))
        if evaluated is None or len(evaluated) < 4:
            return []
        U = s.normalize(np.atleast_2d(evaluated))[:, self.free]
        U = U - U.mean(0)
        _, _, vt = np.linalg.svd(U, full_matrices=False)
        out = []
        for k in range(min(self.cfg.latent_components, len(vt))):
            w = np.zeros(s.dim); w[self.free] = vt[k]
            out.append(w)
        return out

    def finalize(self, raw: List[Candidate], center: np.ndarray, radius: np.ndarray, evaluated: Optional[np.ndarray]) -> List[Candidate]:
        seen = {design_hash(x) for x in (np.atleast_2d(evaluated) if evaluated is not None and len(evaluated) else [])}
        seen.add(design_hash(center))
        out: List[Candidate] = []
        self.stats = {"generated": len(raw), "rejected_geometry": 0, "duplicates": 0}
        for c in raw:
            c.x = self._project(c.x, center, radius)
            h = design_hash(c.x)
            if h in seen:
                self.stats["duplicates"] += 1
                continue
            seen.add(h)
            if not self.feasible(c.x):
                self.stats["rejected_geometry"] += 1
                continue
            out.append(c)
            if len(out) >= self.cfg.max_pool:
                break
        self.stats["accepted"] = len(out)
        return out
