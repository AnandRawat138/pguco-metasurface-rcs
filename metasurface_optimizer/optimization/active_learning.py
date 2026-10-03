"""Physics-Guided Uncertainty-Aware Counterfactual Local Optimisation (PGUCO).

This is the proposed algorithm (docs/methodology.md, Algorithm 1):

    x0, y0 -> F_phys -> S_local -> C -> A -> x*

Input: existing design x0 with its (validated / measured / reported) response
y0; parameter schema; objectives; physics configuration; high-fidelity budget
N_max; batch size K.

    0. J0 = J(y0); fit normaliser; D = {(x0, y0)}; centre c = x0; radius r = r_0.
    1. Initial local design (charged to N_HF): one-at-a-time central
       perturbations x0 +/- delta_i e_i (gives high-fidelity local
       sensitivities S_i, Eq. S1) or a Latin hypercube inside the trust region.
    2. repeat
       a. train surrogate ensemble on D (physics-aware loss on the response block,
          physics terms also on unlabeled collocation designs in the trust region);
       b. local sensitivities and gradient of mu_J at c (surrogate, Eq. S1);
       c. counterfactual pool C = CF1..CF5(c, r, S) U NSGA-II-on-surrogate(c, r) (CF6);
          reject invalid geometries;
       d. predict mu, sigma for every candidate; composite mean/std (Eq. C1);
          physics audit of the predicted response;
       e. select K candidates by acquisition (Eq. A2 default; A1/A3/A4/A5 switchable),
          lexicographically after Pareto rank in multi-objective mode;
       f. [Mode B] evaluate them with the high-fidelity source; record the
          prediction error of each (out-of-sample, measured before training on it);
       g. D <- D U new; update incumbent and trust region
          (success: c <- best, r <- min(r * expand, r_max); failure streak: r <- max(r * shrink, r_min));
       h. check stopping rules (Eq. T1-T4).
    3. return best validated design, validated Pareto set, best predicted
       (unvalidated) candidates with explanations.

Mode A (prediction-only) runs steps 2a-2e once on the data available and
returns ranked PREDICTED candidates; nothing is labeled validated.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.stats import qmc

from metasurface_optimizer.core.logging_utils import get_logger
from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.data.dataset import DesignDataset, DesignRecord, Partition, design_hash
from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.geometry.geometry_generator import GeometryGenerator
from metasurface_optimizer.models.features import FeatureExtractor
from metasurface_optimizer.models.registry import build_surrogate
from metasurface_optimizer.models.target_encoding import TargetEncoder
from metasurface_optimizer.models.torch_common import TrainConfig
from metasurface_optimizer.optimization.acquisition import AcquisitionConfig, select_batch
from metasurface_optimizer.optimization.counterfactual import Candidate, CounterfactualConfig, CounterfactualGenerator
from metasurface_optimizer.optimization.local_sensitivity import SensitivityResult, central_difference_points, local_sensitivity, regression_sensitivity
from metasurface_optimizer.optimization.nsga2 import NSGA2Config, nsga2
from metasurface_optimizer.optimization.pareto import fast_non_dominated_sort, hypervolume, pareto_mask
from metasurface_optimizer.optimization.problem import OptimizationProblem
from metasurface_optimizer.optimization.stopping import StoppingConfig, StoppingMonitor
from metasurface_optimizer.physics.passivity import passivity_violation
from metasurface_optimizer.solvers.base import BudgetedEvaluator, BudgetExceeded

log = get_logger(__name__)


@dataclass
class TrustRegionConfig:
    init_radius: float = 0.06
    min_radius: float = 0.02
    max_radius: float = 0.3
    expand: float = 1.5
    shrink: float = 0.5
    failure_tolerance: int = 2
    success_rel_tol: float = 1e-3


@dataclass
class PGUCOConfig:
    surrogate: str = "physics_ensemble"
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    trust_region: TrustRegionConfig = field(default_factory=TrustRegionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    batch_size: int = 4
    init_strategy: str = "lhs"  # lhs | oat | none   (oat costs 2 d evaluations but yields high-fidelity sensitivities)
    init_top_k: Optional[int] = None
    n_init_lhs: int = 8
    objective_mode: str = "composite"  # composite | pareto
    use_surrogate_nsga: bool = True
    nsga: NSGA2Config = field(default_factory=lambda: NSGA2Config(pop_size=40, generations=20))
    feature_mode: str = "physics"  # physics | raw
    use_collocation: bool = True
    hv_margin: float = 0.1
    response_n_freq: int = 21
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class CounterfactualOptimizer:
    """Stateful implementation of PGUCO supporting automatic (solver) and human-in-the-loop operation."""

    def __init__(self, problem: OptimizationProblem, cfg: Optional[PGUCOConfig] = None,
                 prior_feature_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 reference_frequency: Optional[float] = None, conditions: Optional[dict] = None):
        self.problem = problem
        self.cfg = cfg or PGUCOConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.schema = problem.schema
        self.dataset = DesignDataset()
        self.generator = GeometryGenerator(self.schema) if self.schema.template else None
        band = next((o.band for o in problem.objectives.objectives if o.band), None)
        fref = reference_frequency or (0.5 * (band[0] + band[1]) if band else None)
        self.features = FeatureExtractor(self.schema, fref, self.cfg.feature_mode, prior_feature_fn)
        self.band = band
        self.conditions = conditions
        self.encoder: Optional[TargetEncoder] = None
        self.surrogate = None
        self.center = problem.x0.copy()
        self.radius = np.full(self.schema.dim, self.cfg.trust_region.init_radius)
        self.failures = 0
        self.iteration = 0
        self.history: List[Dict[str, Any]] = []
        self.pending: List[Dict[str, Any]] = []
        self.hf_sensitivity: Optional[SensitivityResult] = None
        self.surrogate_sensitivity: Optional[SensitivityResult] = None
        self.monitor = StoppingMonitor(self.cfg.stopping)
        self.timing: Dict[str, float] = {"train": 0.0, "propose": 0.0, "evaluate": 0.0}
        self.last_proposals: List[Dict[str, Any]] = []
        self.prediction_log: List[Dict[str, Any]] = []
        self.counterfactual = CounterfactualGenerator(self.schema, self._is_feasible_generic, self.cfg.counterfactual)

    def _is_feasible_generic(self, x: np.ndarray) -> bool:
        return self.problem.validator.is_valid(x, generic=True)

    # ================================================================ data
    def set_baseline(self, response: EMResponse, partition: Partition = Partition.TRAINING, counted: bool = False) -> np.ndarray:
        """Register the existing design's response y0 (not charged to N_HF unless ``counted``)."""
        if response.fidelity.is_prediction:
            raise ValueError("The baseline response must be validated/measured/reported, not PREDICTED")
        J0 = self.problem.evaluate_response(response, self.problem.x0)
        self.problem.set_baseline(J0)
        self.encoder = TargetEncoder.from_response(self.problem.objectives.names, response, self.cfg.response_n_freq, self.band)
        self.dataset.add(DesignRecord(self.problem.x0.copy(), response, J0, partition, "baseline", tag="baseline",
                                      meta={"counted": counted, "is_external_baseline": partition == Partition.TRAINING}))
        self.best_composite_hist: List[float] = []
        return J0

    def finish_initial_design(self) -> None:
        """After the initial local design is observed: HF sensitivities, frozen normaliser, incumbent."""
        self.compute_hf_sensitivity()
        self.problem.fit_normalizer(np.vstack([r.J for r in self.hf_records()]))
        self.best_composite_hist = [self.composite_of(self.problem.J0, self.problem.x0),
                                    min(self.composite_of(r.J, r.x) for r in self.hf_records())]
        best = self.best_record()
        if self.best_composite_hist[-1] < self.best_composite_hist[0]:
            self.center = best.x.copy()

    def composite_of(self, J: np.ndarray, x: np.ndarray) -> float:
        return float(self.problem.scalarize(np.atleast_2d(J), np.atleast_2d(x))[0])

    def observe(self, x: np.ndarray, response: EMResponse, tag: str = "", meta: Optional[dict] = None) -> DesignRecord:
        """Add a high-fidelity (or measured) result for design x (Mode B upload or solver call)."""
        if response.fidelity.is_prediction:
            raise ValueError("Cannot observe a PREDICTED response")
        J = self.problem.evaluate_response(response, x)
        rec = DesignRecord(np.asarray(x, float), response, J, Partition.TRAINING, response.provenance.origin, iteration=self.iteration, tag=tag,
                           meta={"counted": True, **(meta or {})})
        self.dataset.add(rec)
        # out-of-sample prediction error, if this design was proposed with a prediction
        for p in self.pending:
            if design_hash(p["x"]) == rec.hash:
                self.prediction_log.append({"iteration": self.iteration, "predicted_J": p["mu_J"], "sigma_J": p["sigma_J"], "validated_J": J.tolist(),
                                            "predicted_composite": p["mu"], "sigma_composite": p["sigma"], "validated_composite": self.composite_of(J, x)})
        return rec

    def hf_records(self) -> List[DesignRecord]:
        return [r for r in self.dataset.records if r.partition == Partition.TRAINING and r.J is not None]

    @property
    def n_hf(self) -> int:
        return sum(1 for r in self.hf_records() if r.meta.get("counted", True))

    def best_record(self) -> DesignRecord:
        recs = self.hf_records()
        comp = [self.composite_of(r.J, r.x) for r in recs]
        return recs[int(np.argmin(comp))]

    # ================================================================ initial design
    def initial_design(self) -> List[Dict[str, Any]]:
        """Designs to evaluate first (returned so they can be simulated externally in Mode B)."""
        cfg, s = self.cfg, self.schema
        feas = lambda x: self.problem.validator.is_valid(x, generic=True)
        if cfg.init_strategy == "oat":
            idx = list(np.where(s.free_mask)[0])
            if cfg.init_top_k is not None and self.surrogate_sensitivity is not None:
                idx = [i for i in np.argsort(-self.surrogate_sensitivity.normalized) if i in idx][: cfg.init_top_k]
            plan = central_difference_points(self.problem.x0, s, idx, s.steps, feas)
            return [{"x": s.snap(p["x"]), "tag": f"init_oat_{s.names[p['index']]}_{'+' if p['sign'] > 0 else '-'}"} for p in plan if p["ok"]]
        if cfg.init_strategy == "lhs":
            u0 = s.normalize(self.problem.x0)
            pts = qmc.LatinHypercube(d=s.dim, seed=self.rng).random(cfg.n_init_lhs * 3)
            out = []
            for p in pts:
                u = np.clip(u0 + (2 * p - 1) * self.radius, 0, 1)
                x = s.denormalize(u); x[~s.free_mask] = self.problem.x0[~s.free_mask]; x = s.snap(x)
                if feas(x):
                    out.append({"x": x, "tag": "init_lhs"})
                if len(out) >= cfg.n_init_lhs:
                    break
            return out
        return []

    def compute_hf_sensitivity(self) -> Optional[SensitivityResult]:
        """Eq. S1 from the high-fidelity OAT data (no extra evaluations: results are looked up)."""
        cache = {r.hash: r.J for r in self.hf_records()}
        if len(cache) < 3:
            return None

        def lookup(x):
            h = design_hash(self.schema.snap(x))
            if h not in cache:
                raise KeyError("missing")
            return cache[h]

        plan_ok = lambda x: design_hash(self.schema.snap(x)) in cache
        try:
            self.hf_sensitivity = local_sensitivity(lookup, self.problem.x0, self.schema, feasible=plan_ok, f0=cache[design_hash(self.problem.x0)],
                                                    scalarize=lambda J: self.composite_of(J, self.problem.x0), source="high_fidelity",
                                                    objective_names=self.problem.objectives.names)
        except KeyError:
            self.hf_sensitivity = None
        if self.hf_sensitivity is None or all(s == "skipped" for s in self.hf_sensitivity.scheme):
            # Space-filling initial design: no one-at-a-time pairs, so fit a local linear model instead (Eq. S1'').
            recs = self.hf_records()
            self.hf_sensitivity = regression_sensitivity(np.vstack([r.x for r in recs]), np.vstack([r.J for r in recs]), self.problem.x0,
                                                         self.schema, scalarize=lambda J: self.composite_of(J, self.problem.x0),
                                                         objective_names=self.problem.objectives.names)
        return self.hf_sensitivity

    # ================================================================ surrogate
    def _training_arrays(self):
        recs = self.hf_records()
        X = np.vstack([r.x for r in recs])
        Y = np.vstack([self.encoder.encode(r.response, r.J) for r in recs])
        return X, Y

    def fit_surrogate(self) -> None:
        t = time.time()
        X, Y = self._training_arrays()
        if len(X) < 2:
            raise RuntimeError("At least two evaluated designs are needed to train a surrogate (run the initial design first)")
        self.problem.fit_normalizer(np.vstack([r.J for r in self.hf_records()]))
        self.surrogate = build_surrogate(self.cfg.surrogate, self.features, self.encoder, self.cfg.seed + self.iteration,
                                         self.problem.physics, self.generator, self.cfg.train)
        Xc = None
        if self.cfg.use_collocation and self.problem.physics.any_enabled:
            Xc = self._trust_region_samples(self.problem.physics.collocation_points)
        self.surrogate.fit(X, Y, Xc)
        self.timing["train"] += time.time() - t

    def _trust_region_samples(self, n: int) -> np.ndarray:
        s = self.schema
        u = np.clip(s.normalize(self.center) + self.rng.uniform(-1, 1, (n, s.dim)) * self.radius, 0, 1)
        X = s.denormalize(u)
        X[:, ~s.free_mask] = self.center[~s.free_mask]
        return s.snap(X)

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        Y, S = self.surrogate.predict(np.atleast_2d(X))
        if S is None:
            S = np.zeros_like(Y)
        m = self.encoder.m
        mu_J, sd_J = Y[:, :m], S[:, :m]
        mu, sd = self.problem.scalarize_with_std(mu_J, sd_J, X)
        return {"Y": Y, "S": S, "mu_J": mu_J, "sigma_J": sd_J, "mu": mu, "sigma": sd}

    # ================================================================ proposal
    def _surrogate_sensitivity(self) -> SensitivityResult:
        f = lambda x: self.predict(x[None])["mu_J"][0]
        return local_sensitivity(f, self.center, self.schema, feasible=lambda x: self.problem.validator.is_valid(x, generic=False),
                                 scalarize=lambda J: self.composite_of(J, self.center), source="surrogate",
                                 objective_names=self.problem.objectives.names)

    def _nsga_candidates(self) -> List[Candidate]:
        s = self.schema
        free = np.where(s.free_mask)[0]
        uc = s.normalize(self.center)[free]
        lo, hi = np.clip(uc - self.radius[free], 0, 1), np.clip(uc + self.radius[free], 0, 1)

        def to_x(Z):
            X = np.repeat(self.center[None], len(Z), 0)
            X[:, free] = s.lower[free] + Z * (s.upper[free] - s.lower[free])
            return s.snap(X)

        kappa = self.cfg.acquisition.kappa if self.cfg.acquisition.kind not in ("mean", "random") else 0.0

        def f(Z):
            X = to_x(Z)
            p = self.predict(X)
            F = self.problem.normalized(p["mu_J"]) - kappa * p["sigma_J"] * self.problem.normalizer.scale()
            CV = np.array([max(0.0, max(self.problem.validator.constraint_values(x, generic=False).values())) for x in X])
            return F, CV

        Z, F, CV, _ = nsga2(f, len(free), self.cfg.nsga, self.rng, lo, hi, init=uc[None])
        front = fast_non_dominated_sort(F, CV)[0] if len(F) else []
        return [Candidate(x, "surrogate_nsga2", self.center.copy(), "predicted Pareto-optimal (NSGA-II on surrogate)") for x in to_x(Z[front])]

    def propose(self, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Steps 2a-2e. Returns K selected candidates with predictions and explanations (all PREDICTED)."""
        t = time.time()
        k = k or self.cfg.batch_size
        self.fit_surrogate()
        self.surrogate_sensitivity = self._surrogate_sensitivity()
        sens = self.hf_sensitivity.normalized if (self.hf_sensitivity is not None and self.iteration == 0) else self.surrogate_sensitivity.normalized
        grad = getattr(self.surrogate_sensitivity, "scalar", None)
        evaluated = np.vstack([r.x for r in self.hf_records()])
        pool = self.counterfactual.generate(self.center, self.radius, self.rng, sens, grad, evaluated)
        if self.cfg.use_surrogate_nsga:
            extra = self.counterfactual.finalize(self._nsga_candidates(), self.center, self.radius, np.vstack([evaluated] + [c.x[None] for c in pool]) if pool else evaluated)
            pool += extra
        if not pool:
            log.warning("Empty candidate pool; shrinking trust region")
            self.radius = np.maximum(self.radius * 0.5, self.cfg.trust_region.min_radius)
            return []
        X = np.vstack([c.x for c in pool])
        pred = self.predict(X)
        best = min(self.composite_of(r.J, r.x) for r in self.hf_records())
        U = self.schema.normalize(X)[:, self.schema.free_mask]
        Uref = self.schema.normalize(evaluated)[:, self.schema.free_mask]
        acq = copy.deepcopy(self.cfg.acquisition)
        priority = None
        if self.cfg.objective_mode == "pareto":
            Fopt = self.problem.normalized(pred["mu_J"]) - (acq.kappa if acq.kind not in ("mean", "random") else 0) * pred["sigma_J"] * self.problem.normalizer.scale()
            priority = np.empty(len(X))
            for r_, fr in enumerate(fast_non_dominated_sort(Fopt)):
                priority[fr] = r_
        chosen = select_batch(acq, pred["mu"], pred["sigma"], best, U, Uref, k, self.rng, priority)
        out = []
        base_pred = self.predict(self.center[None])
        for c in chosen:
            i = c["index"]
            out.append(self._explain(pool[i], {kk: v[i] for kk, v in pred.items()}, c, base_pred))
        # keep the full predicted ranking (Mode A output)
        order = np.argsort(pred["mu"])
        self.last_pool = {"X": X, "mu": pred["mu"], "sigma": pred["sigma"], "mu_J": pred["mu_J"], "sigma_J": pred["sigma_J"],
                          "kinds": [c.kind for c in pool], "order": order}
        self.pending = out
        self.last_proposals = out
        self.timing["propose"] += time.time() - t
        return out

    # ================================================================ explanation
    def _predicted_effect(self, y_new: np.ndarray, y_old: np.ndarray) -> str:
        enc = self.encoder
        f = enc.frequency
        names = [n for n in enc.slices if n.startswith("RCSred_")]
        if names:
            s = enc.slices[names[0]]
            a, b = y_old[s], y_new[s]

            def centroid(r):
                w = np.maximum(-r, 0) + 1e-9
                return float((w * f).sum() / w.sum())

            ca, cb = centroid(a), centroid(b)
            msg = f"predicted mean in-band RCS reduction {a.mean():.1f} dB -> {b.mean():.1f} dB ({names[0].replace('RCSred_', '')}); "
            if self.band:
                fc = 0.5 * (self.band[0] + self.band[1])
                toward = abs(cb - fc) < abs(ca - fc)
                msg += f"reduction-weighted centroid shifts {ca / 1e9:.2f} -> {cb / 1e9:.2f} GHz ({'toward' if toward else 'away from'} the target-band centre)"
            return msg
        pairs = enc.complex_pairs()
        if pairs:
            sre, sim, key = pairs[0]
            pa = np.degrees(np.angle(y_old[sre] + 1j * y_old[sim])).mean()
            pb = np.degrees(np.angle(y_new[sre] + 1j * y_new[sim])).mean()
            return f"predicted mean reflection phase S{key[0]}{key[1]} {pa:.0f} -> {pb:.0f} deg"
        return "no response block available"

    def _explain(self, cand: Candidate, p: Dict[str, Any], acq: Dict[str, float], base_pred: Dict[str, np.ndarray]) -> Dict[str, Any]:
        sens = self.surrogate_sensitivity
        mods = cand.modifications(self.schema)
        for mmod in mods:
            mmod["local_sensitivity"] = float(sens.normalized[self.schema.index(mmod["parameter"])])
        base_mu = float(base_pred["mu"][0])
        ref_J = self.composite_of(self.best_record().J, self.best_record().x)
        dJ = (p["mu"] - ref_J) / max(abs(ref_J), 1e-12) * 100 if abs(ref_J) > 1e-9 else p["mu"] - ref_J
        reasons = []
        if mods and max(m_["local_sensitivity"] for m_ in mods) >= np.sort(sens.normalized)[-3:].min():
            reasons.append("high local sensitivity")
        if p["mu"] < ref_J:
            reasons.append("predicted improvement")
        if "beta_sigma" in acq and acq.get("beta_sigma", 0) > 0.5 * max(acq.get("EI", 0), 1e-12):
            reasons.append("exploration (high uncertainty)")
        if acq.get("D", 0) > 0.1:
            reasons.append("diversity from tested designs")
        resp = self.encoder.decode(p["Y"], p["S"], origin=self.surrogate.name)["response"]
        phys = {}
        if resp.sparams is not None and self.problem.physics.passive:
            v = [passivity_violation(sv)["max"] for kk, sv in resp.sparams.data.items() if kk[0] == kk[1]]
            phys["predicted_passivity_violation_max"] = float(max(v)) if v else 0.0
        return {
            "x": cand.x, "values": self.schema.to_dict_values(cand.x), "kind": cand.kind, "detail": cand.detail, "modifications": mods,
            "mu": float(p["mu"]), "sigma": float(p["sigma"]), "mu_J": p["mu_J"].tolist(), "sigma_J": p["sigma_J"].tolist(),
            "predicted_composite_change_pct_vs_incumbent": float(dJ),
            "predicted_effect": self._predicted_effect(p["Y"], base_pred["Y"][0]),
            "acquisition": acq, "reason": " + ".join(reasons) or "acquisition ranking", "physics_audit": phys,
            "label": "PREDICTED (model-associated sensitivity / predicted effect; not validated)",
            "valid_geometry": True,
        }

    # ================================================================ update
    def update_after_batch(self) -> Optional[str]:
        """Steps 2g-2h after the pending batch has been observed."""
        recs = self.hf_records()
        comps = np.array([self.composite_of(r.J, r.x) for r in recs])
        best_i = int(np.argmin(comps))
        best = comps[best_i]
        prev_best = self.best_composite_hist[-1]
        tr = self.cfg.trust_region
        if best < prev_best - tr.success_rel_tol * max(abs(prev_best), 1e-9):
            self.center = recs[best_i].x.copy()
            self.radius = np.minimum(self.radius * tr.expand, tr.max_radius)
            self.failures = 0
            status = "success"
        else:
            self.failures += 1
            status = "failure"
            if self.failures >= tr.failure_tolerance:
                self.radius = np.maximum(self.radius * tr.shrink, tr.min_radius)
                self.failures = 0
        self.best_composite_hist.append(float(best))
        hv = self.validated_hypervolume()
        sig = max((p["sigma"] for p in self.pending), default=None)
        self.history.append({"iteration": self.iteration, "n_hf": self.n_hf, "best_composite": float(best), "hypervolume": hv, "status": status,
                             "radius": float(self.radius.mean()), "selected": [{"values": p["values"], "mu": p["mu"], "sigma": p["sigma"], "kind": p["kind"]} for p in self.pending]})
        self.pending = []
        self.iteration += 1
        return self.monitor.update(float(best), hv, self.n_hf, sig, self.iteration)

    def validated_hypervolume(self) -> float:
        recs = self.hf_records()
        F = self.problem.normalized(np.vstack([r.J for r in recs]))
        ref = self.problem.normalized(self.problem.J0[None])[0] + self.cfg.hv_margin
        return hypervolume(F, ref)

    # ================================================================ drivers
    def run(self, evaluator: BudgetedEvaluator, baseline_response: Optional[EMResponse] = None) -> Dict[str, Any]:
        """Mode B with an automatic high-fidelity source (solver adapter or synthetic oracle)."""
        t0 = time.time()
        if baseline_response is None:
            baseline_response = evaluator(self.schema.to_dict_values(self.problem.x0), self.conditions, "baseline")
            counted = True
        else:
            counted = False
        self.set_baseline(baseline_response, counted=counted)
        if evaluator.max_evaluations is not None:
            self.cfg.stopping.max_hf_evaluations = evaluator.max_evaluations
        reason = None
        try:
            for d in self.initial_design():
                t = time.time()
                self.observe(d["x"], evaluator(self.schema.to_dict_values(d["x"]), self.conditions, d["tag"]), d["tag"])
                self.timing["evaluate"] += time.time() - t
            self.finish_initial_design()
            while True:
                remaining = evaluator.remaining
                if remaining is not None and remaining <= 0:
                    reason = "budget exhausted"
                    break
                props = self.propose(min(self.cfg.batch_size, remaining) if remaining is not None else None)
                if not props:
                    reason = self.update_after_batch() or None
                    if reason:
                        break
                    continue
                for p in props:
                    t = time.time()
                    self.observe(p["x"], evaluator(p["values"], self.conditions, f"iter{self.iteration}_{p['kind']}"), f"iter{self.iteration}")
                    self.timing["evaluate"] += time.time() - t
                reason = self.update_after_batch()
                if reason:
                    break
        except BudgetExceeded:
            reason = "budget exhausted"
        self.stop_reason = reason
        self.timing["total"] = time.time() - t0
        return self.results()

    def predict_only(self, k: int = 10) -> Dict[str, Any]:
        """Mode A: rank candidates from the data available; no high-fidelity evaluation."""
        props = self.propose(k)
        return {"mode": "A (prediction-only)", "candidates": props, "label": "PREDICTED - requires high-fidelity validation"}

    # ================================================================ results
    def results(self) -> Dict[str, Any]:
        recs = self.hf_records()
        J = np.vstack([r.J for r in recs])
        comps = np.array([self.composite_of(r.J, r.x) for r in recs])
        b = int(np.argmin(comps))
        J0 = self.problem.J0
        c0 = self.composite_of(J0, self.problem.x0)
        Fn = self.problem.normalized(J)
        pm = pareto_mask(Fn)
        n_hf = self.n_hf
        improvement = c0 - comps[b]
        fid = recs[b].fidelity
        pred_err = None
        if self.prediction_log:
            pc = np.array([p["predicted_composite"] for p in self.prediction_log]); vc = np.array([p["validated_composite"] for p in self.prediction_log])
            pj = np.array([p["predicted_J"] for p in self.prediction_log]); vj = np.array([p["validated_J"] for p in self.prediction_log])
            pred_err = {"n": len(pc), "composite_rmse": float(np.sqrt(np.mean((pc - vc) ** 2))), "composite_mae": float(np.mean(np.abs(pc - vc))),
                        "J_rmse": np.sqrt(np.mean((pj - vj) ** 2, 0)).tolist(), "J_mae": np.mean(np.abs(pj - vj), 0).tolist(),
                        "note": "out-of-sample: each prediction was made before the design was evaluated"}
        return {
            "algorithm": "PGUCO (physics-guided uncertainty-aware counterfactual local optimisation)",
            "config": self.cfg.to_dict(),
            "objective_names": self.problem.objectives.names,
            "baseline": {"values": self.schema.to_dict_values(self.problem.x0), "J": J0.tolist(), "composite": c0},
            "best_validated": {"values": self.schema.to_dict_values(recs[b].x), "x": recs[b].x.tolist(), "J": recs[b].J.tolist(), "composite": float(comps[b]),
                               "fidelity": fid.value if fid else None, "label": fid.display if fid else None,
                               "delta_J_pct": ((recs[b].J - J0) / np.maximum(np.abs(J0), 1e-12) * 100).tolist(), "record_id": recs[b].record_id},
            "pareto_validated": [{"values": self.schema.to_dict_values(recs[i].x), "J": recs[i].J.tolist(), "composite": float(comps[i])} for i in np.where(pm)[0]],
            "n_hf": n_hf,
            "composite_improvement": float(improvement),
            "improvement_per_hf": float(improvement / max(n_hf, 1)),
            "history": self.history,
            "stop_reason": getattr(self, "stop_reason", None),
            "hf_sensitivity": self.hf_sensitivity.to_dict() if self.hf_sensitivity else None,
            "surrogate_sensitivity": self.surrogate_sensitivity.to_dict() if self.surrogate_sensitivity else None,
            "prediction_error": pred_err,
            "timing": self.timing,
            "surrogate": self.surrogate.describe() if self.surrogate is not None else None,
            "features": self.features.describe(),
            "encoder": self.encoder.to_dict() if self.encoder else None,
        }
