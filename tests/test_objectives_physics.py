import numpy as np
import pytest
import torch

from metasurface_optimizer.optimization.objectives import ObjectiveNormalizer, ObjectiveSet, composite, relative_improvement
from metasurface_optimizer.physics.config import PhysicsConfig
from metasurface_optimizer.physics.passivity import passivity_loss_torch, passivity_violation, project_passive
from metasurface_optimizer.physics.reciprocity import reciprocity_violation
from metasurface_optimizer.physics.spectral_smoothness import resonance_weights, roughness, smoothness_loss_torch


def test_objectives_evaluate(oracle, schema, objectives):
    r = oracle.evaluate(schema.to_dict_values(schema.default_vector()))
    J = objectives.evaluate(r, schema.default_vector())
    assert J.shape == (2,) and np.all(np.isfinite(J))
    assert -1.0 <= J[1] <= 0.0  # -H normalised


def test_unknown_objective_rejected():
    with pytest.raises(ValueError):
        ObjectiveSet.from_config([{"name": "x", "kind": "nope"}])


def test_normalizers():
    J = np.array([[1.0, 10.0], [2.0, 30.0], [3.0, 20.0]])
    mm = ObjectiveNormalizer("minmax").fit(J)
    assert np.allclose(mm.transform(J).min(0), 0) and np.allclose(mm.transform(J).max(0), 1)
    z = ObjectiveNormalizer("zscore").fit(J)
    assert np.allclose(z.transform(J).mean(0), 0)
    br = ObjectiveNormalizer("baseline_relative").fit(J[:1], J[0])
    assert np.allclose(br.transform(J[0]), 0) and np.allclose(br.transform(J[1]), [1.0, 2.0])
    br2 = ObjectiveNormalizer("baseline_relative", floors=[0, 100.0]).fit(J[:1], J[0])
    assert np.isclose(br2.transform(J[1])[1], 0.2)
    with pytest.raises(ValueError):
        ObjectiveNormalizer("baseline_relative").fit(J)


def test_composite_and_improvement():
    assert np.allclose(composite(np.array([[1.0, -1.0]]), np.array([2.0, 1.0]), 0.5, 2.0), [2.0])
    assert np.isclose(relative_improvement(-12.0, -10.0), -20.0)


def test_passivity():
    s = np.array([0.5, 0.9, 1.1]); t = np.array([0.5, 0.5, 0.0])
    v = passivity_violation(s, t)
    assert v["violated"] and v["max"] > 0
    s2, t2 = project_passive(s, t)
    assert np.all(np.abs(s2) ** 2 + np.abs(t2) ** 2 <= 1 + 1e-12)
    re = torch.tensor([[0.9, 1.2]]); im = torch.zeros_like(re)
    assert passivity_loss_torch(re, im) > 0 and passivity_loss_torch(re * 0.5, im) == 0


def test_reciprocity():
    a = np.array([0.1 + 0.2j]); assert reciprocity_violation(a, a)["max_abs"] == 0


def test_smoothness_resonance_preserving():
    f = np.linspace(0, 1, 50)
    smooth, rough = np.sin(2 * np.pi * f), np.sin(2 * np.pi * f) + 0.05 * (-1) ** np.arange(50)
    assert roughness(rough) > roughness(smooth)
    res = 1 / (1 + ((f - 0.5) / 0.01) ** 2)  # sharp resonance in the reference data
    w = resonance_weights(res[None])
    assert w[24] < w[5]  # curvature is penalised less at the resonance
    yt = torch.tensor(rough[None], dtype=torch.float32)
    assert smoothness_loss_torch(yt) > smoothness_loss_torch(torch.tensor(smooth[None], dtype=torch.float32))


def test_physics_config_conditional():
    c = PhysicsConfig(passive=False, reciprocal=False)
    assert not c.passivity_active and not c.reciprocity_active
    assert not PhysicsConfig.none().any_enabled
