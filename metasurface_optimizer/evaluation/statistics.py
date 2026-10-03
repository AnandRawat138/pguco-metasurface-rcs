"""Multi-seed statistics (Sec. 30): mean, std, median, best, worst, bootstrap CI. No seed is dropped."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def summarize(values: Sequence[float], lower_is_better: bool = True, ci: float = 0.95, n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    out = {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "median": float(np.median(v)),
           "best": float(v.min() if lower_is_better else v.max()), "worst": float(v.max() if lower_is_better else v.min())}
    if len(v) >= 3:
        boots = rng.choice(v, (n_boot, len(v))).mean(1)
        a = (1 - ci) / 2
        out["ci_low"], out["ci_high"] = float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))
        out["ci_level"] = ci
    return out


def convergence_band(curves: Sequence[Sequence[Dict[str, float]]], budget: int) -> Dict[str, list]:
    """Best-so-far composite vs N_HF across seeds (step-interpolated): mean, std, min, max."""
    grid = np.arange(1, budget + 1)
    M = []
    for c in curves:
        if not c:
            continue
        n = np.array([p["n_hf"] for p in c]); b = np.array([p["best"] for p in c])
        row = np.full(len(grid), np.nan)
        for gi, g in enumerate(grid):
            k = np.where(n <= g)[0]
            if len(k):
                row[gi] = b[k[-1]]
        M.append(row)
    if not M:
        return {"n_hf": grid.tolist()}
    M = np.array(M)
    return {"n_hf": grid.tolist(), "mean": np.nanmean(M, 0).tolist(), "std": np.nanstd(M, 0).tolist(),
            "min": np.nanmin(M, 0).tolist(), "max": np.nanmax(M, 0).tolist()}
