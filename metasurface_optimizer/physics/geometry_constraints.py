"""Geometry constraint penalty used in the optimisation objective (Eq. G1).

    J_constraint(x) = sum_i max(0, g_i(x))^2

with g_i from :class:`~metasurface_optimizer.geometry.geometry_validator.GeometryValidator`
(analytic template constraints + schema linear constraints). Designs with any
g_i > 0 are additionally *rejected* before ranking; the penalty only shapes
search inside evolutionary optimisers that may visit infeasible points.

Note: the surrogate's training loss has no geometry term because every
training design is, by construction, a valid geometry. L_geometry in the
total loss (Eq. L1) is therefore applied at the optimisation level; see
docs/methodology.md, Sec. 11.
"""
from __future__ import annotations

import numpy as np

from metasurface_optimizer.geometry.geometry_validator import GeometryValidator


def constraint_penalty(validator: GeometryValidator, X: np.ndarray) -> np.ndarray:
    return np.array([validator.penalty(x) for x in np.atleast_2d(X)])


def feasible_mask(validator: GeometryValidator, X: np.ndarray, generic: bool = True) -> np.ndarray:
    return validator.filter_valid(X, generic)
