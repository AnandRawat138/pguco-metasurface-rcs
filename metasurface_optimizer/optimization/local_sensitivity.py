"""Local response sensitivities around the existing design (Eq. S1-S2).

    S_i     = [ J(x0 + d_i e_i) - J(x0 - d_i e_i) ] / (2 d_i)      (central difference)
    S_hat_i = |S_i| / (sum_j |S_j| + eps)

When x0 +/- d_i e_i leaves the bounds or violates geometry constraints a
one-sided difference is used and flagged. Sensitivities may be estimated from
high-fidelity evaluations (``source='high_fidelity'``) or from the surrogate
mean (``source='surrogate'``); the source is always recorded.

These are *local response sensitivities* of the chosen objective at x0 with
step d_i; they are not causal physical effects unless backed by controlled
perturbation experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from metasurface_optimizer.geometry.parameter_schema import ParameterSchema


@dataclass
class SensitivityResult:
    names: List[str]
    raw: np.ndarray  # S_i per objective column (d x m) or (d, 1) for scalar
    normalized: np.ndarray  # S_hat_i for the composite / first column (d,)
    deltas: np.ndarray
    scheme: List[str]  # central | forward | backward | skipped
    source: str
    objective_names: List[str] = field(default_factory=list)
    evaluations: List[Dict] = field(default_factory=list)

    def ranking(self) -> List[Dict[str, object]]:
        o = np.argsort(-self.normalized)
        return [{"parameter": self.names[i], "sensitivity": float(self.normalized[i]), "raw": self.raw[i].tolist(),
                 "delta": float(self.deltas[i]), "scheme": self.scheme[i]} for i in o]

    def table(self) -> str:
        lines = [f"{'Parameter':<22}{'Sensitivity':>12}   ({self.source}; local response sensitivity)"]
        for r in self.ranking():
            lines.append(f"{r['parameter']:<22}{r['sensitivity']:>12.3f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"source": self.source, "ranking": self.ranking(), "objective_names": self.objective_names,
                "label": "local response sensitivity (not a causal effect)"}


def central_difference_points(x0: np.ndarray, schema: ParameterSchema, indices: Sequence[int], deltas: np.ndarray,
                              feasible: Optional[Callable[[np.ndarray], bool]] = None) -> List[Dict]:
    """Plan the perturbation designs (used both for HF experiments and surrogate queries)."""
    plan = []
    for i in indices:
        for sign in (+1, -1):
            x = x0.copy()
            x[i] = x0[i] + sign * deltas[i]
            ok = schema.lower[i] - 1e-12 <= x[i] <= schema.upper[i] + 1e-12 and (feasible is None or feasible(x))
            plan.append({"index": int(i), "sign": sign, "x": x, "ok": bool(ok)})
    return plan


def local_sensitivity(
    f: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, schema: ParameterSchema,
    deltas: Optional[np.ndarray] = None, indices: Optional[Sequence[int]] = None,
    feasible: Optional[Callable[[np.ndarray], bool]] = None, f0: Optional[np.ndarray] = None,
    scalarize: Optional[Callable[[np.ndarray], float]] = None, source: str = "surrogate",
    objective_names: Optional[List[str]] = None, cache: Optional[Dict[str, np.ndarray]] = None, eps: float = 1e-12,
    weighting: str = "step",
) -> SensitivityResult:
    """Estimate S_i with ``f`` (x -> objective vector). ``cache`` maps design hashes to known J (reuse HF results).

    ``weighting='raw'`` gives S_hat_i = |S_i| / sum_j |S_j| exactly (Eq. S2); ``'step'`` (default) uses
    |S_i| d_i, i.e. the objective change produced by each parameter's own perturbation step (Eq. S2'), which is
    unit-independent when parameters have different units (mm, degrees, dimensionless).
    """
    x0 = np.asarray(x0, float)
    deltas = schema.steps if deltas is None else np.asarray(deltas, float)
    indices = list(np.where(schema.free_mask)[0]) if indices is None else list(indices)
    from metasurface_optimizer.data.dataset import design_hash

    def call(x):
        h = design_hash(x)
        if cache is not None and h in cache:
            return np.atleast_1d(cache[h])
        v = np.atleast_1d(f(x))
        if cache is not None:
            cache[h] = v
        return v

    J0 = np.atleast_1d(f0) if f0 is not None else call(x0)
    m = len(J0)
    raw = np.zeros((schema.dim, m))
    scheme = ["skipped"] * schema.dim
    evals = []
    plan = central_difference_points(x0, schema, indices, deltas, feasible)
    by_i: Dict[int, Dict[int, Dict]] = {}
    for p in plan:
        by_i.setdefault(p["index"], {})[p["sign"]] = p
    for i, d in by_i.items():
        up, dn = d[1], d[-1]
        if up["ok"] and dn["ok"]:
            Ju, Jd = call(up["x"]), call(dn["x"])
            raw[i] = (Ju - Jd) / (2 * deltas[i]); scheme[i] = "central"
            evals += [{"x": up["x"].tolist(), "J": Ju.tolist()}, {"x": dn["x"].tolist(), "J": Jd.tolist()}]
        elif up["ok"]:
            Ju = call(up["x"]); raw[i] = (Ju - J0) / deltas[i]; scheme[i] = "forward"
            evals.append({"x": up["x"].tolist(), "J": Ju.tolist()})
        elif dn["ok"]:
            Jd = call(dn["x"]); raw[i] = (J0 - Jd) / deltas[i]; scheme[i] = "backward"
            evals.append({"x": dn["x"].tolist(), "J": Jd.tolist()})
    # sensitivity of the scalar objective: either via scalarize(J) or the first column.
    if scalarize is not None:
        s0 = scalarize(J0)
        scal = np.zeros(schema.dim)
        for i in by_i:
            if scheme[i] == "central":
                scal[i] = (scalarize(J0 + raw[i] * deltas[i]) - scalarize(J0 - raw[i] * deltas[i])) / (2 * deltas[i])
            elif scheme[i] in ("forward", "backward"):
                scal[i] = (scalarize(J0 + raw[i] * deltas[i]) - s0) / deltas[i]
    else:
        scal = raw[:, 0]
    contrib = np.abs(scal) * deltas if weighting == "step" else np.abs(scal)
    norm = contrib / (contrib.sum() + eps)
    res = SensitivityResult(schema.names, raw, norm, deltas, scheme, source, objective_names or [], evals)
    res.scalar = scal
    res.weighting = weighting
    return res


def regression_sensitivity(
    X: np.ndarray, J: np.ndarray, x0: np.ndarray, schema: ParameterSchema,
    scalarize: Optional[Callable[[np.ndarray], float]] = None, source: str = "high_fidelity",
    objective_names: Optional[List[str]] = None, ridge: float = 1e-3, eps: float = 1e-12, weighting: str = "step",
) -> SensitivityResult:
    """Local sensitivities from scattered evaluations around x0 (Eq. S1''): a ridge-regularised local linear fit

        J(x) ~ J(x0) + sum_i S_i (x_i - x0_i),   fitted on the evaluated designs near x0,

    used when the initial design is space-filling (LHS) rather than one-at-a-time, so no central-difference
    pairs exist. Inputs are scaled by the step d_i before fitting. With few points relative to the number of
    free parameters the estimate is rough; the result records n and d so this is visible."""
    X = np.atleast_2d(np.asarray(X, float)); J = np.atleast_2d(np.asarray(J, float))
    deltas = schema.steps
    free = np.where(schema.free_mask)[0]
    Z = (X[:, free] - x0[free]) / deltas[free]
    A = np.hstack([np.ones((len(Z), 1)), Z])
    reg = ridge * np.eye(A.shape[1]); reg[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ J)  # (1 + d_free, m)
    raw = np.zeros((schema.dim, J.shape[1]))
    raw[free] = coef[1:] / deltas[free, None]
    scheme = ["skipped"] * schema.dim
    for i in free:
        scheme[i] = "linear_fit"
    J0 = coef[0]
    if scalarize is not None:
        scal = np.zeros(schema.dim)
        for i in free:
            scal[i] = (scalarize(J0 + raw[i] * deltas[i]) - scalarize(J0 - raw[i] * deltas[i])) / (2 * deltas[i])
    else:
        scal = raw[:, 0]
    contrib = np.abs(scal) * deltas if weighting == "step" else np.abs(scal)
    res = SensitivityResult(schema.names, raw, contrib / (contrib.sum() + eps), deltas, scheme, source, objective_names or [],
                            [{"n_points": int(len(X)), "n_free_parameters": int(len(free)), "ridge": ridge}])
    res.scalar = scal
    res.weighting = weighting
    return res
