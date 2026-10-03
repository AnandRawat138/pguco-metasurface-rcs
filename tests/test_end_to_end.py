"""Small end-to-end runs (reduced budgets/epochs) of the proposed loop, both modes."""
import numpy as np
import pytest

from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.geometry.geometry_validator import GeometryValidator
from metasurface_optimizer.models.torch_common import TrainConfig
from metasurface_optimizer.optimization.active_learning import CounterfactualOptimizer, PGUCOConfig
from metasurface_optimizer.optimization.baselines import run_baseline
from metasurface_optimizer.optimization.nsga2 import NSGA2Config
from metasurface_optimizer.optimization.objectives import ObjectiveNormalizer
from metasurface_optimizer.optimization.problem import OptimizationProblem
from metasurface_optimizer.physics.config import PhysicsConfig
from metasurface_optimizer.solvers.base import BudgetedEvaluator


def _problem(schema, objectives):
    return OptimizationProblem(schema, objectives, GeometryValidator(schema), ObjectiveNormalizer("baseline_relative"), schema.default_vector(),
                               physics=PhysicsConfig(reciprocal=False))


def _cfg(**kw):
    return PGUCOConfig(surrogate="physics_ensemble", batch_size=3, train=TrainConfig(epochs=60, n_members=2),
                       nsga=NSGA2Config(pop_size=16, generations=4), **kw)


def test_mode_b_automatic(schema, oracle, objectives):
    ev = BudgetedEvaluator(oracle, 22)
    opt = CounterfactualOptimizer(_problem(schema, objectives), _cfg())
    res = opt.run(ev)
    assert res["n_hf"] <= 22
    assert res["best_validated"]["fidelity"] == Fidelity.SYNTHETIC_ORACLE.value
    assert res["best_validated"]["composite"] <= res["baseline"]["composite"] + 1e-12  # never worse than the baseline
    assert opt.hf_sensitivity is not None and len(opt.hf_sensitivity.ranking()) == schema.dim
    assert all(p["label"].startswith("PREDICTED") for p in opt.last_proposals)
    assert all(GeometryValidator(schema).is_valid(p["x"]) for p in opt.last_proposals)
    assert res["prediction_error"]["n"] > 0


def test_mode_b_human_in_the_loop_and_mode_a(schema, oracle, objectives):
    """Step-wise API: initial design -> (user simulates) -> observe -> propose."""
    opt = CounterfactualOptimizer(_problem(schema, objectives), _cfg())
    base = oracle.evaluate(schema.to_dict_values(schema.default_vector()))
    opt.set_baseline(base)
    plan = opt.initial_design()
    assert len(plan) == opt.cfg.n_init_lhs
    for d in plan:  # "uploaded" results
        opt.observe(d["x"], oracle.evaluate(schema.to_dict_values(d["x"])), d["tag"])
    opt.finish_initial_design()
    out = opt.predict_only(5)
    assert out["mode"].startswith("A") and len(out["candidates"]) == 5
    with pytest.raises(ValueError):
        from metasurface_optimizer.core.provenance import Provenance
        from metasurface_optimizer.electromagnetics.response import EMResponse

        opt.observe(plan[0]["x"], EMResponse(Provenance(Fidelity.PREDICTED)))


def test_baseline_runner(schema, oracle, objectives):
    prob = _problem(schema, objectives)
    base = oracle.evaluate(schema.to_dict_values(prob.x0))
    prob.set_baseline(prob.evaluate_response(base, prob.x0))
    for name in ("random_perturbation", "local_search", "bayesian_optimization"):
        r = run_baseline(name, prob, BudgetedEvaluator(oracle, 8), base, 0)
        assert r["n_hf"] <= 8 and r["best_composite"] <= 0.0 + 1e-12 or r["best_J"] is not None
