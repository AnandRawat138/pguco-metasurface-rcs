"""Objective functions J_k(x) computed from an electromagnetic response.

All objectives are expressed so that **lower is better**. Quantities that
should be maximised (entropy, bandwidth, manufacturing margin) are negated
and the negation is recorded in :attr:`ObjectiveSpec.sense`.

Implemented kinds (equation numbers refer to docs/methodology.md):

``specular_rcs``          (Eq. O1)  J_spec  = sum_f w_f sigma_spec(f) / sum_f w_f
``broadband_rcs``         (Eq. O2)  J_broad = int w(f) sigma(f) df / int w(f) df   (trapezoid rule)
``angular_robustness``    (Eq. O3)  J_ang   = sum_theta w_theta sigma_bar(theta) / sum w_theta   (or max)
``polarization_robustness`` (Eq. O4) J_pol  = (1/N_p) sum_p J_p                    (or max)
``angular_entropy``       (Eq. O5)  J_diff  = -mean_f H_ang(f)
``total_scattered``       (Eq. O6)  mean over band of (1/4pi) int sigma dOmega
``reduction_bandwidth``   (Eq. O7)  J_bw    = -FBW{ f : 10log10(sigma_spec/sigma_ref) <= threshold }
``peak_rcs``              (Eq. O10) J_peak  = max_f s(sigma(f)) per (pol, inc), aggregated  (worst frequency)
``sparam_level``          (Eq. O8)  mean_f |S_ij(f)|_dB (sense='max' negates)
``manufacturing_margin``  (Eq. O9)  J_mfg   = max_i g_i(x)  (geometry constraint closest to violation)

``scale`` selects how sigma enters the average: ``linear`` (m^2), ``dbsm``
(average of dBsm) or ``reduction_db`` (dB relative to the reference, e.g. a
PEC plate). Diffusion via maximum entropy is *one* configurable choice; the
framework never assumes it is the correct stealth objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from metasurface_optimizer.electromagnetics.rcs import RCSData, to_dbsm
from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.electromagnetics.response_features import angular_entropy, band_mask, threshold_bandwidth


@dataclass
class ObjectiveSpec:
    name: str
    kind: str
    weight: float = 1.0
    band: Optional[Tuple[float, float]] = None  # Hz
    polarizations: Optional[List[str]] = None  # None = all available
    incidence: Optional[List[float]] = None  # deg; None = all available
    scale: str = "linear"  # linear | dbsm | reduction_db
    aggregate: str = "mean"  # mean | max (worst case)
    freq_weights: Optional[List[float]] = None  # w_f on the band samples; None = uniform
    angle_weights: Optional[List[float]] = None  # w_theta
    rcs_kind: str = "specular"  # specular | monostatic (used by broadband/angular/polarization)
    normalize_entropy: bool = True
    threshold_db: float = -10.0
    sparam: Optional[Tuple[int, int]] = None
    sense: str = "min"  # "max" means the *physical* quantity is maximised and J is its negation
    norm_floor: float = 0.0  # lower bound on |J0| in baseline-relative normalisation (guards J0 ~ 0)
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _pol_idx(rcs: RCSData, spec: ObjectiveSpec) -> List[int]:
    pols = spec.polarizations or rcs.polarizations
    missing = [p for p in pols if p not in rcs.polarizations]
    if missing:
        raise ValueError(f"Objective {spec.name}: polarizations {missing} not in response")
    return [rcs.polarizations.index(p) for p in pols]


def _inc_idx(rcs: RCSData, spec: ObjectiveSpec) -> List[int]:
    if spec.incidence is None:
        return list(range(len(rcs.theta_inc)))
    idx = []
    for th in spec.incidence:
        k = int(np.argmin(np.abs(rcs.theta_inc - th)))
        if abs(rcs.theta_inc[k] - th) > 0.5:
            raise ValueError(f"Objective {spec.name}: incidence {th} deg not available in response")
        idx.append(k)
    return idx


def _scaled(rcs: RCSData, spec: ObjectiveSpec, which: str) -> np.ndarray:
    """Return (P, I, F) array in the requested scale."""
    s = rcs.specular() if which == "specular" else rcs.monostatic
    if s is None:
        raise ValueError(f"Objective {spec.name}: {which} RCS not available")
    if spec.scale == "linear":
        return s
    if spec.scale == "dbsm":
        return to_dbsm(s)
    if spec.scale == "reduction_db":
        if rcs.reference is None:
            raise ValueError(f"Objective {spec.name}: reduction_db needs reference RCS")
        return to_dbsm(s) - to_dbsm(rcs.reference)
    raise ValueError(f"Unknown scale {spec.scale}")


def _fw(spec: ObjectiveSpec, n: int) -> np.ndarray:
    if spec.freq_weights is None:
        return np.ones(n)
    w = np.asarray(spec.freq_weights, float)
    if len(w) != n:
        raise ValueError(f"Objective {spec.name}: {len(w)} frequency weights for {n} band samples")
    return w


def _band_mean(f: np.ndarray, y: np.ndarray, w: np.ndarray, integral: bool) -> float:
    if integral and len(f) > 1:
        return float(np.trapezoid(w * y, f) / np.trapezoid(w, f))
    return float(np.sum(w * y) / np.sum(w))


def _agg(vals: Sequence[float], weights: Optional[Sequence[float]], how: str) -> float:
    v = np.asarray(vals, float)
    if how == "max":
        return float(v.max())
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    return float(np.sum(w * v) / np.sum(w))


def _per_pol_inc(rcs: RCSData, spec: ObjectiveSpec, which: str, integral: bool) -> np.ndarray:
    """Band average for each (pol, inc), shape (len(pols), len(incs))."""
    s = _scaled(rcs, spec, which)
    m = band_mask(rcs.frequency, spec.band)
    w = _fw(spec, int(m.sum()))
    P, I = _pol_idx(rcs, spec), _inc_idx(rcs, spec)
    return np.array([[_band_mean(rcs.frequency[m], s[p, i, m], w, integral) for i in I] for p in P])


# --------------------------------------------------------------- kinds
def _specular(resp: EMResponse, spec: ObjectiveSpec, x, ctx) -> float:
    return _agg(_per_pol_inc(resp.rcs, spec, "specular", False).ravel(), None, spec.aggregate)


def _broadband(resp: EMResponse, spec: ObjectiveSpec, x, ctx) -> float:
    return _agg(_per_pol_inc(resp.rcs, spec, spec.rcs_kind, True).ravel(), None, spec.aggregate)


def _angular(resp: EMResponse, spec: ObjectiveSpec, x, ctx) -> float:
    a = _per_pol_inc(resp.rcs, spec, spec.rcs_kind, False).mean(axis=0)  # over pols -> per angle
    return _agg(a, spec.angle_weights, spec.aggregate)


def _polarization(resp: EMResponse, spec: ObjectiveSpec, x, ctx) -> float:
    jp = _per_pol_inc(resp.rcs, spec, spec.rcs_kind, False).mean(axis=1)  # J_p
    return _agg(jp, None, spec.aggregate)


def _entropy(resp: EMResponse, spec: ObjectiveSpec, x, ctx) -> float:
    rcs = resp.rcs
    m = np.where(band_mask(rcs.frequency, spec.band))[0]
    H = [angular_entropy(rcs, p, i, k, normalized=spec.normalize_entropy) for p in _pol_idx(rcs, spec) for i in _inc_idx(rcs, spec) for k in m]
    return -float(np.min(H) if spec.aggregate == "max" else np.mean(H))


def _total(resp: EMResponse, spec: ObjectiveSpec, x, ctx) -> float:
    rcs = resp.rcs
    ts = rcs.total_scattered()
    m = band_mask(rcs.frequency, spec.band)
    vals = [ts[p, i, m].mean() for p in _pol_idx(rcs, spec) for i in _inc_idx(rcs, spec)]
    if spec.scale == "dbsm":
        vals = list(to_dbsm(np.asarray(vals)))
    return _agg(vals, None, spec.aggregate)


def _peak(resp: EMResponse, spec: ObjectiveSpec, x, ctx) -> float:
    """Worst frequency in the band: max_f s(sigma(f)) per (pol, inc), then aggregated (e.g. the '> 10 dB reduction everywhere' criterion)."""
    rcs = resp.rcs
    s = _scaled(rcs, spec, spec.rcs_kind)
    m = band_mask(rcs.frequency, spec.band)
    return _agg([float(s[p, i, m].max()) for p in _pol_idx(rcs, spec) for i in _inc_idx(rcs, spec)], None, spec.aggregate)


def _reduction_bw(resp: EMResponse, spec: ObjectiveSpec, x, ctx) -> float:
    rcs = resp.rcs
    red = to_dbsm(rcs.specular()) - to_dbsm(rcs.reference)
    m = band_mask(rcs.frequency, spec.band)
    fbw = [threshold_bandwidth(rcs.frequency[m], red[p, i, m], spec.threshold_db)["fractional_bandwidth"]
           for p in _pol_idx(rcs, spec) for i in _inc_idx(rcs, spec)]
    return -float(np.min(fbw) if spec.aggregate == "max" else np.mean(fbw))


def _sparam(resp: EMResponse, spec: ObjectiveSpec, x, ctx) -> float:
    s = resp.sparams
    i, j = spec.sparam or (1, 1)
    pols = spec.polarizations or s.polarizations
    m = band_mask(s.frequency, spec.band)
    vals = [float(np.mean(s.db(i, j, p)[m])) for p in pols]
    v = _agg(vals, None, spec.aggregate)
    return -v if spec.sense == "max" else v


def _mfg(resp: EMResponse, spec: ObjectiveSpec, x, ctx) -> float:
    validator = ctx.get("validator") if ctx else None
    if validator is None or x is None:
        raise ValueError("manufacturing_margin needs a geometry validator and design vector")
    g = validator.constraint_values(np.asarray(x, float), generic=False)
    return float(max(g.values()))


OBJECTIVE_KINDS: Dict[str, Callable[..., float]] = {
    "specular_rcs": _specular,
    "broadband_rcs": _broadband,
    "angular_robustness": _angular,
    "polarization_robustness": _polarization,
    "angular_entropy": _entropy,
    "total_scattered": _total,
    "reduction_bandwidth": _reduction_bw,
    "peak_rcs": _peak,
    "sparam_level": _sparam,
    "manufacturing_margin": _mfg,
}

#: kinds that depend only on x (not on the EM response)
DESIGN_ONLY_KINDS = {"manufacturing_margin"}


@dataclass
class ObjectiveSet:
    """Ordered set of objectives producing the vector J = [J_1..J_m]."""

    objectives: List[ObjectiveSpec]
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Sequence[Mapping[str, Any]], context: Optional[dict] = None) -> "ObjectiveSet":
        objs = []
        for c in cfg:
            c = dict(c)
            if c.get("band") is not None:
                c["band"] = tuple(float(b) for b in c["band"])
            if c.get("sparam") is not None:
                c["sparam"] = tuple(c["sparam"])
            objs.append(ObjectiveSpec(**c))
        for o in objs:
            if o.kind not in OBJECTIVE_KINDS:
                raise ValueError(f"Unknown objective kind {o.kind!r}")
        return cls(objs, context or {})

    @property
    def names(self) -> List[str]:
        return [o.name for o in self.objectives]

    @property
    def weights(self) -> np.ndarray:
        return np.array([o.weight for o in self.objectives], float)

    @property
    def m(self) -> int:
        return len(self.objectives)

    @property
    def norm_floors(self) -> np.ndarray:
        return np.array([o.norm_floor for o in self.objectives], float)

    def response_objective_mask(self) -> np.ndarray:
        return np.array([o.kind not in DESIGN_ONLY_KINDS for o in self.objectives])

    def evaluate(self, resp: Optional[EMResponse], x: Optional[np.ndarray] = None) -> np.ndarray:
        """J vector. Response-based objectives require ``resp``; design-only ones require ``x``."""
        out = np.empty(self.m)
        for k, o in enumerate(self.objectives):
            if o.kind in DESIGN_ONLY_KINDS:
                out[k] = OBJECTIVE_KINDS[o.kind](None, o, x, self.context)
            else:
                if resp is None:
                    raise ValueError(f"Objective {o.name} needs a response")
                out[k] = OBJECTIVE_KINDS[o.kind](resp, o, x, self.context)
        return out

    def evaluate_design_only(self, x: np.ndarray) -> np.ndarray:
        """Values of design-only objectives (NaN for response-based ones)."""
        out = np.full(self.m, np.nan)
        for k, o in enumerate(self.objectives):
            if o.kind in DESIGN_ONLY_KINDS:
                out[k] = OBJECTIVE_KINDS[o.kind](None, o, x, self.context)
        return out

    def to_config(self) -> List[dict]:
        return [o.to_dict() for o in self.objectives]


# ------------------------------------------------------------- normalisation
@dataclass
class ObjectiveNormalizer:
    """Maps raw J to dimensionless J~ before any weighted combination (Eq. N1-N3).

    * ``minmax``:            J~ = (J - J_min) / (J_max - J_min)
    * ``zscore``:            J~ = (J - mean) / std
    * ``baseline_relative``: J~ = (J - J_0) / max(|J_0|, f_k)   with a per-objective floor f_k >= 0
      (default 0) that prevents division by a near-zero baseline value

    Statistics are fitted once (on the baseline / initial data) and then frozen
    so the scalarisation does not drift during optimisation; the method and the
    fitted statistics are written to the experiment configuration.
    """

    method: str = "baseline_relative"
    eps: float = 1e-12
    stats: Dict[str, List[float]] = field(default_factory=dict)
    floors: Optional[List[float]] = None

    def fit(self, J: np.ndarray, J0: Optional[np.ndarray] = None) -> "ObjectiveNormalizer":
        J = np.atleast_2d(np.asarray(J, float))
        if self.method == "minmax":
            lo, hi = J.min(0), J.max(0)
            self.stats = {"min": lo.tolist(), "max": hi.tolist()}
        elif self.method == "zscore":
            self.stats = {"mean": J.mean(0).tolist(), "std": J.std(0).tolist()}
        elif self.method == "baseline_relative":
            if J0 is None:
                raise ValueError("baseline_relative normalisation needs the baseline objective J0")
            self.stats = {"J0": np.asarray(J0, float).tolist()}
        else:
            raise ValueError(f"Unknown normalisation {self.method}")
        return self

    def transform(self, J: np.ndarray) -> np.ndarray:
        J = np.asarray(J, float)
        if self.method == "minmax":
            lo, hi = np.asarray(self.stats["min"]), np.asarray(self.stats["max"])
            span = np.where(hi - lo > self.eps, hi - lo, 1.0)
            return (J - lo) / span
        if self.method == "zscore":
            mu, sd = np.asarray(self.stats["mean"]), np.asarray(self.stats["std"])
            return (J - mu) / np.where(sd > self.eps, sd, 1.0)
        J0 = np.asarray(self.stats["J0"])
        return (J - J0) / self._denom(J0)

    def scale(self) -> np.ndarray:
        """d J~ / d J (used to transform standard deviations)."""
        if self.method == "minmax":
            lo, hi = np.asarray(self.stats["min"]), np.asarray(self.stats["max"])
            return 1.0 / np.where(hi - lo > self.eps, hi - lo, 1.0)
        if self.method == "zscore":
            sd = np.asarray(self.stats["std"])
            return 1.0 / np.where(sd > self.eps, sd, 1.0)
        return 1.0 / self._denom(np.asarray(self.stats["J0"]))

    def _denom(self, J0: np.ndarray) -> np.ndarray:
        fl = np.zeros_like(J0) if self.floors is None else np.asarray(self.floors, float)
        return np.maximum(np.maximum(np.abs(J0), fl), self.eps)

    def to_dict(self) -> dict:
        return {"method": self.method, "stats": self.stats, "floors": self.floors}


def composite(J_tilde: np.ndarray, weights: np.ndarray, constraint_penalty: np.ndarray | float = 0.0, lambda_c: float = 0.0) -> np.ndarray:
    """J(x) = sum_k lambda_k J~_k(x) + lambda_c J_constraint(x)   (Eq. C1)."""
    J_tilde = np.atleast_2d(J_tilde)
    w = np.asarray(weights, float)
    return J_tilde @ w + lambda_c * np.asarray(constraint_penalty, float)


def relative_improvement(J_new: np.ndarray, J_old: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Delta J = (J_new - J_old) / |J_old| * 100  (percent; negative = improvement for minimisation)."""
    return (np.asarray(J_new, float) - np.asarray(J_old, float)) / np.maximum(np.abs(np.asarray(J_old, float)), eps) * 100.0
