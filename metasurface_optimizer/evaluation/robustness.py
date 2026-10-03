"""Manufacturing-tolerance robustness of a candidate, estimated with the surrogate (PREDICTED).

Samples x + e, e_i ~ U(-tol_i, tol_i) (tol_i defaults to the parameter's
manufacturing resolution or 1% of its range), discards invalid geometries,
and reports the predicted composite distribution. Clearly labeled PREDICTED.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from metasurface_optimizer.geometry.parameter_schema import ParameterSchema


def tolerance_analysis(x: np.ndarray, schema: ParameterSchema, predict_composite: Callable[[np.ndarray], np.ndarray],
                       feasible: Callable[[np.ndarray], bool], n: int = 200, tol: Optional[np.ndarray] = None, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if tol is None:
        tol = np.array([p.resolution if p.resolution else 0.01 * p.span for p in schema.parameters])
    tol = np.where(schema.free_mask, tol, 0.0)
    X = np.asarray(x, float)[None] + rng.uniform(-1, 1, (n, schema.dim)) * tol
    X = schema.clip(X)
    ok = np.array([feasible(xi) for xi in X])
    if not ok.any():
        return {"n_valid": 0, "label": "PREDICTED"}
    c = predict_composite(X[ok])
    return {"n_valid": int(ok.sum()), "fraction_valid": float(ok.mean()), "mean": float(c.mean()), "std": float(c.std()),
            "p95": float(np.quantile(c, 0.95)), "worst": float(c.max()), "tolerance": tol.tolist(), "label": "PREDICTED"}
