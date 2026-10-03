import numpy as np
import pytest

from metasurface_optimizer.core.provenance import Fidelity, Provenance
from metasurface_optimizer.data.dataset import DataLeakageError, DesignDataset, DesignRecord, Partition, assert_no_leakage
from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.geometry.geometry_validator import GeometryValidator
from metasurface_optimizer.models.features import FeatureExtractor
from metasurface_optimizer.models.registry import build_surrogate
from metasurface_optimizer.models.target_encoding import TargetEncoder
from metasurface_optimizer.models.torch_common import TrainConfig
from metasurface_optimizer.optimization.acquisition import AcquisitionConfig, expected_improvement, select_batch
from metasurface_optimizer.optimization.counterfactual import CounterfactualGenerator
from metasurface_optimizer.optimization.differential_evolution import DEConfig, differential_evolution
from metasurface_optimizer.optimization.local_sensitivity import local_sensitivity
from metasurface_optimizer.optimization.nsga2 import NSGA2Config, nsga2
from metasurface_optimizer.optimization.pareto import fast_non_dominated_sort, hypervolume, pareto_mask
from metasurface_optimizer.optimization.pso import PSOConfig, pso
from metasurface_optimizer.optimization.stopping import StoppingConfig, StoppingMonitor
from metasurface_optimizer.physics.config import PhysicsConfig
from metasurface_optimizer.solvers.base import BudgetedEvaluator, BudgetExceeded


@pytest.fixture
def small_data(schema, oracle, objectives, rng):
    X = schema.sample_uniform(400, rng, fix_to=schema.default_vector())
    v = GeometryValidator(schema)
    X = np.array([x for x in X if v.is_valid(x)][:14])
    resps = [oracle.evaluate(schema.to_dict_values(x)) for x in X]
    J = np.array([objectives.evaluate(r, x) for r, x in zip(resps, X)])
    enc = TargetEncoder.from_response(objectives.names, resps[0], n_freq=11, band=(8e9, 12e9))
    Y = np.vstack([enc.encode(r, j) for r, j in zip(resps, J)])
    return X, Y, J, enc


@pytest.mark.parametrize("name", ["random_forest", "xgboost", "gp", "mlp", "physics_mlp", "transformer", "cnn", "physics_ensemble"])
def test_surrogates_fit_predict(name, schema, small_data):
    X, Y, J, enc = small_data
    fe = FeatureExtractor(schema, 10e9)
    m = build_surrogate(name, fe, enc, 0, PhysicsConfig(reciprocal=False), train=TrainConfig(epochs=60, n_members=2))
    m.fit(X[:10], Y[:10], X[10:] if name == "physics_mlp" else None)
    mu, sd = m.predict(X[10:])
    assert mu.shape == (4, Y.shape[1])
    assert sd is not None and sd.shape == mu.shape and np.all(sd >= 0)


def test_target_encoder_roundtrip(small_data):
    X, Y, J, enc = small_data
    d = enc.decode(Y[0])
    assert d["response"].fidelity is Fidelity.PREDICTED
    assert np.allclose(d["J"], J[0])


def test_ensemble_uncertainty_formula(schema, small_data):
    from metasurface_optimizer.models.ensemble_uncertainty import EnsembleSurrogate

    X, Y, J, enc = small_data
    fe = FeatureExtractor(schema, 10e9)
    ens = EnsembleSurrogate([build_surrogate("random_forest", fe, enc, 0), build_surrogate("gp", fe, enc, 0)]).fit(X, Y)
    P, _ = ens.member_predictions(X[:3])
    mu, sd = ens.predict(X[:3])
    assert np.allclose(mu, P.mean(0)) and np.allclose(sd, np.sqrt(P.var(0)))


def test_dataset_partitions_and_leakage(schema):
    ds = DesignDataset()
    x = schema.default_vector()
    resp = EMResponse(Provenance(Fidelity.REPORTED))
    ds.add(DesignRecord(x + 1, resp, partition=Partition.REFERENCE_ONLY))
    with pytest.raises(DataLeakageError):
        ds.add(DesignRecord(x + 1, resp, partition=Partition.TRAINING))
    with pytest.raises(DataLeakageError):
        ds.add(DesignRecord(x, EMResponse(Provenance(Fidelity.PREDICTED)), partition=Partition.TRAINING))
    with pytest.raises(DataLeakageError):
        assert_no_leakage([x], [x])


def test_local_sensitivity_quadratic(schema):
    w = np.zeros(schema.dim); w[schema.index("length_a")] = 3.0; w[schema.index("period")] = 1.0
    x0 = schema.default_vector()
    f = lambda x: np.array([np.sum(w * (x - x0)) + 0.5 * np.sum((x - x0) ** 2)])
    res = local_sensitivity(f, x0, schema, weighting="raw")
    i = schema.index("length_a")
    assert np.isclose(res.raw[i, 0], 3.0) and res.scheme[i] == "central"
    assert res.ranking()[0]["parameter"] == "length_a"
    assert np.isclose(res.normalized.sum(), 1.0)


def test_counterfactuals_valid_and_traceable(schema, rng):
    v = GeometryValidator(schema)
    gen = CounterfactualGenerator(schema, lambda x: v.is_valid(x))
    sens = np.ones(schema.dim) / schema.dim
    c = gen.generate(schema.default_vector(), np.full(schema.dim, 0.1), rng, sens, np.ones(schema.dim), None)
    kinds = {k.kind for k in c}
    assert {"single", "pairwise", "correlated", "descent", "trust_region_random"} <= kinds
    assert all(v.is_valid(k.x) for k in c)
    assert all(k.modifications(schema) for k in c)
    i = schema.index("tan_delta")
    assert all(np.isclose(k.x[i], schema.default_vector()[i]) for k in c)  # fixed parameter never modified


def test_acquisition():
    mu = np.array([0.0, -1.0, 0.5]); sd = np.array([0.1, 0.1, 2.0])
    ei = expected_improvement(mu, sd, best=0.0)
    assert ei[1] > ei[0] and np.all(ei >= 0)
    U = np.array([[0.0], [0.01], [1.0]])
    for kind in ("ei_sigma_diversity", "ei", "lcb", "mean", "random"):
        b = select_batch(AcquisitionConfig(kind=kind), mu, sd, 0.0, U, np.empty((0, 1)), 2)
        assert len({x["index"] for x in b}) == 2
    assert select_batch(AcquisitionConfig(kind="mean"), mu, sd, 0.0, U, np.empty((0, 1)), 1)[0]["index"] == 1
    assert select_batch(AcquisitionConfig(kind="lcb", kappa=3.0), mu, sd, 0.0, U, np.empty((0, 1)), 1)[0]["index"] == 2


def test_pareto_and_hypervolume():
    F = np.array([[1, 3], [2, 2], [3, 1], [3, 3.0]])
    assert pareto_mask(F).tolist() == [True, True, True, False]
    assert np.isclose(hypervolume(F, np.array([4, 4.0])), 6.0)
    assert len(fast_non_dominated_sort(F)[0]) == 3


def test_nsga2_zdt1(rng):
    def zdt1(Z):
        f1 = Z[:, 0]; g = 1 + 9 * Z[:, 1:].mean(1); f2 = g * (1 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2]), np.zeros(len(Z))

    Z, F, CV, _ = nsga2(zdt1, 5, NSGA2Config(pop_size=40, generations=60), rng)
    assert hypervolume(F, np.array([1.1, 1.1])) > 0.6  # true front HV ~ 0.87


def test_de_pso_budget(rng):
    calls = {"n": 0}

    def f(Z):
        calls["n"] += 1
        if calls["n"] > 60:
            raise BudgetExceeded()
        return ((Z - 0.3) ** 2).sum(1)

    z, v = differential_evolution(f, 3, DEConfig(pop_size=8), rng)
    assert calls["n"] == 61 and v < 0.1
    calls["n"] = 0
    z, v = pso(f, 3, PSOConfig(n_particles=8), rng)
    assert v < 0.1


def test_budgeted_evaluator(oracle, schema):
    ev = BudgetedEvaluator(oracle, 1)
    ev(schema.to_dict_values(schema.default_vector()))
    with pytest.raises(BudgetExceeded):
        ev(schema.to_dict_values(schema.default_vector()))


def test_stopping_rules():
    m = StoppingMonitor(StoppingConfig(rules=["objective"], objective_window=2, objective_eps=1e-3))
    assert m.update(1.0, None, 1, None, 1) is None
    m.update(0.5, None, 2, None, 2)
    m.update(0.5, None, 3, None, 3)
    assert m.update(0.5, None, 4, None, 4).startswith("objective")
    m2 = StoppingMonitor(StoppingConfig(rules=["budget"], max_hf_evaluations=5))
    assert m2.update(0, None, 5, None, 1).startswith("budget")
    m3 = StoppingMonitor(StoppingConfig(rules=["uncertainty"], sigma_min=0.1))
    assert m3.update(0, None, 1, 0.01, 1).startswith("uncertainty")
