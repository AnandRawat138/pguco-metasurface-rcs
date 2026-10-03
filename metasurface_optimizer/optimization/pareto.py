"""Pareto utilities: non-dominated sorting, crowding distance, hypervolume (minimisation)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray, cv: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Deb et al. (2002) with constraint domination: feasible beats infeasible; lower CV wins among infeasible."""
    F = np.atleast_2d(F)
    n = len(F)
    cv = np.zeros(n) if cv is None else np.asarray(cv, float)
    S = [[] for _ in range(n)]
    counts = np.zeros(n, int)
    fronts: List[List[int]] = [[]]

    def dom(i, j):
        if cv[i] <= 0 and cv[j] > 0:
            return True
        if cv[i] > 0 and cv[j] > 0:
            return cv[i] < cv[j]
        if cv[i] > 0 and cv[j] <= 0:
            return False
        return dominates(F[i], F[j])

    for i in range(n):
        for j in range(i + 1, n):
            if dom(i, j):
                S[i].append(j); counts[j] += 1
            elif dom(j, i):
                S[j].append(i); counts[i] += 1
    fronts[0] = [i for i in range(n) if counts[i] == 0]
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [np.array(f, int) for f in fronts if f]


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        o = np.argsort(F[:, k])
        span = F[o[-1], k] - F[o[0], k]
        d[o[0]] = d[o[-1]] = np.inf
        if span <= 0:
            continue
        d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return d


def pareto_mask(F: np.ndarray) -> np.ndarray:
    F = np.atleast_2d(F)
    mask = np.ones(len(F), bool)
    for i in range(len(F)):
        if mask[i]:
            dominated_by_i = np.all(F[i] <= F, 1) & np.any(F[i] < F, 1)
            mask[dominated_by_i] = False
            if np.any(np.all(F <= F[i], 1) & np.any(F < F[i], 1)):
                mask[i] = False
    return mask


def hypervolume(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume dominated by F and bounded by ref (minimisation), via recursive slicing (HSO).

    Suitable for the small sets that arise here (tens of points, m <= 5).
    """
    F = np.atleast_2d(np.asarray(F, float))
    ref = np.asarray(ref, float)
    F = F[np.all(F < ref, 1)]
    if len(F) == 0:
        return 0.0
    F = F[pareto_mask(F)]
    return _hso(F, ref)


def _hso(F: np.ndarray, ref: np.ndarray) -> float:
    m = F.shape[1]
    if m == 1:
        return float(ref[0] - F[:, 0].min())
    if m == 2:
        o = np.argsort(F[:, 0])
        P = F[o]
        hv = 0.0
        best_y = ref[1]
        for i in range(len(P)):
            x_r = P[i + 1, 0] if i + 1 < len(P) else ref[0]
            best_y = min(best_y, P[i, 1])
            hv += (x_r - P[i, 0]) * (ref[1] - best_y)
        return float(hv)
    o = np.argsort(F[:, -1])
    P = F[o]
    hv = 0.0
    for i in range(len(P)):
        z_hi = P[i + 1, -1] if i + 1 < len(P) else ref[-1]
        if z_hi <= P[i, -1]:
            continue
        sub = P[: i + 1, :-1]
        sub = sub[pareto_mask(sub)]
        hv += _hso(sub, ref[:-1]) * (z_hi - P[i, -1])
    return float(hv)
