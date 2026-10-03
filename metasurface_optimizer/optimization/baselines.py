"""Conventional comparators, all charged against the same high-fidelity budget N_HF.

* random_perturbation : x0 + Gaussian perturbations (sigma = r0 of the range), keep best
* local_search        : compass/pattern search from x0 with step halving
* pso, differential_evolution, bayesian_optimization, nsga2 (HF black-box)
* surrogate_no_physics: the PGUCO loop with a physics-free ensemble and no physics terms

Every runner returns the same result structure so that the comparison and
ablation tables are generated automatically.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from metasurface_optimizer.optimization.bayesian_optimizer import BOConfig, bayesian_optimization
from metasurface_optimizer.optimization.blackbox import HFObjective
from metasurface_optimizer.optimization.differential_evolution import DEConfig, differential_evolution
from metasurface_optimizer.optimization.nsga2 import NSGA2Config, nsga2
from metasurface_optimizer.optimization.pareto import pareto_mask
from metasurface_optimizer.optimization.problem import OptimizationProblem
from metasurface_optimizer.optimization.pso import PSOConfig, pso
from metasurface_optimizer.solvers.base import BudgetedEvaluator, BudgetExceeded


def _summarize(name: str, obj: HFObjective, t0: float, base_comp: float) -> Dict[str, Any]:
    b = obj.best()
    conv = obj.convergence()
    n_hf = obj.evaluator.n_evaluations
    invalid = sum(1 for r in obj.history if not r.feasible)
    out = {"algorithm": name, "n_hf": n_hf, "time_s": time.time() - t0, "convergence": conv, "invalid_proposals": invalid}
    if b is None:
        out.update({"best_composite": base_comp, "best_J": None, "best_values": None})
    else:
        out.update({"best_composite": b.composite, "best_J": b.J.tolist(), "best_values": obj.problem.schema.to_dict_values(b.x), "best_x": b.x.tolist()})
    feas = [r for r in obj.history if r.feasible]
    if feas:
        F = obj.problem.normalized(np.vstack([r.J for r in feas]))
        out["pareto_J"] = [feas[i].J.tolist() for i in np.where(pareto_mask(F))[0]]
    out["composite_improvement"] = base_comp - out["best_composite"]
    out["improvement_per_hf"] = out["composite_improvement"] / max(n_hf, 1)
    return out


def _with_baseline(obj: HFObjective, baseline_response) -> float:
    x0 = obj.problem.x0
    J0 = obj.problem.evaluate_response(baseline_response, x0)
    return float(obj.problem.scalarize(J0[None], x0[None])[0])


def run_baseline(name: str, problem: OptimizationProblem, evaluator: BudgetedEvaluator, baseline_response, seed: int,
                 conditions: Optional[dict] = None, **kw) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    obj = HFObjective(problem, evaluator, conditions)
    t0 = time.time()
    base = _with_baseline(obj, baseline_response)
    sp = obj.space
    z0 = sp.to_z(problem.x0)[0]
    f = lambda Z: obj.scalar_z(Z, name)
    budget = evaluator.max_evaluations or 50
    try:
        if name == "random_perturbation":
            sigma = kw.get("sigma", 0.15)
            while not obj.exhausted:
                obj.scalar_z(np.clip(z0 + rng.normal(0, sigma, sp.dim), 0, 1)[None], name)
        elif name == "local_search":
            step = kw.get("step", 0.1)
            z, fz = z0.copy(), base
            while not obj.exhausted and step > 1e-3:
                improved = False
                for i in rng.permutation(sp.dim):
                    for sg in (1, -1):
                        zt = z.copy(); zt[i] = np.clip(zt[i] + sg * step, 0, 1)
                        v = f(zt[None])[0]
                        if v < fz:
                            z, fz, improved = zt, v, True
                            break
                    if improved:
                        break
                if not improved:
                    step /= 2
        elif name == "pso":
            pso(f, sp.dim, PSOConfig(n_particles=kw.get("n_particles", 8)), rng, init=z0[None])
        elif name == "differential_evolution":
            differential_evolution(f, sp.dim, DEConfig(pop_size=kw.get("pop_size", 8)), rng, init=z0[None])
        elif name == "bayesian_optimization":
            bayesian_optimization(f, sp.dim, BOConfig(n_init=kw.get("n_init", min(10, max(4, budget // 4)))), rng, z0=z0)
        elif name == "nsga2":
            pop = kw.get("pop_size", 8)
            nsga2(lambda Z: obj.vector_z(Z, name), sp.dim, NSGA2Config(pop_size=pop, generations=10_000), rng, init=z0[None])
        else:
            raise ValueError(f"Unknown baseline {name}")
    except BudgetExceeded:
        pass
    return _summarize(name, obj, t0, base)


BASELINES = ["random_perturbation", "local_search", "pso", "differential_evolution", "bayesian_optimization", "nsga2"]
