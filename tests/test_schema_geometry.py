import numpy as np
import pytest

from metasurface_optimizer.geometry.geometry_generator import GeometryGenerator, available_templates, rasterize_geometry
from metasurface_optimizer.geometry.geometry_validator import GeometryValidator
from metasurface_optimizer.geometry.parameter_schema import ParameterSchema


def test_schema_roundtrip(schema, tmp_path):
    x = schema.default_vector()
    assert np.allclose(schema.denormalize(schema.normalize(x)), x)
    p = tmp_path / "s.yaml"
    schema.save(p)
    s2 = ParameterSchema.load(p)
    assert s2.names == schema.names and s2.fingerprint() == schema.fingerprint()


def test_schema_snap_and_fixed(schema, rng):
    X = schema.sample_uniform(20, rng)
    assert np.all(schema.in_bounds(X))
    i = schema.index("tan_delta")
    assert np.allclose(X[:, i], schema["tan_delta"].default)
    j = schema.index("period")
    assert np.allclose(np.round((X[:, j] - schema["period"].min) / 0.01), (X[:, j] - schema["period"].min) / 0.01, atol=1e-6)


def test_schema_rejects_bad_spec():
    with pytest.raises(ValueError):
        ParameterSchema.from_dict({"parameters": [{"name": "a", "min": 1, "max": 0}]})
    with pytest.raises(ValueError):
        ParameterSchema.from_dict({"parameters": [{"name": "a", "min": 0, "max": 1}, {"name": "a", "min": 0, "max": 1}]})


def test_categorical_parameter():
    s = ParameterSchema.from_dict({"parameters": [{"name": "mat", "type": "categorical", "choices": ["FR4", "RO4003"]}]})
    x = s.to_vector({"mat": "RO4003"})
    assert s.to_dict_values(x)["mat"] == "RO4003"


def test_templates_build_and_validate(schema):
    assert {"rect_patch", "checkerboard_rect_patch", "square_loop", "slotted_patch", "cross_dipole"} <= set(available_templates())
    v = GeometryValidator(schema)
    x = schema.default_vector()
    assert v.validate(x).valid
    bad = x.copy(); bad[schema.index("length_a")] = 9.95  # gap 0.05 < min_gap 0.2
    rep = v.validate(bad)
    assert not rep.valid and any("gap" in k for k in rep.violations)
    rot = x.copy(); rot[schema.index("rotation")] = 30.0  # rotated patch no longer fits
    assert not v.validate(rot).valid


def test_min_feature_and_penalty(schema):
    v = GeometryValidator(schema)
    x = schema.default_vector(); x[schema.index("width_b")] = 0.1
    rep = v.validate(x)
    assert "template:min_width_b" in rep.violations
    assert v.penalty(x) > 0 and v.penalty(schema.default_vector()) == 0


def test_rasterize(schema):
    g = GeometryGenerator(schema)
    m = g.rasterize(schema.default_vector(), 64)
    assert m.shape == (64, 64) and 0.3 < m.mean() < 0.6  # two 8.5^2 and two 5^2 patches in a 20x20 super-cell


def test_other_templates_validate():
    s = ParameterSchema.from_dict({
        "template": "square_loop", "manufacturing": {"min_gap": 0.2, "min_feature": 0.2},
        "parameters": [
            {"name": "period", "min": 5, "max": 15, "default": 10, "role": "period"},
            {"name": "loop_length", "min": 2, "max": 14, "default": 8, "role": "length"},
            {"name": "trace_width", "min": 0.1, "max": 2, "default": 0.5, "role": "width"},
            {"name": "substrate_thickness", "min": 0.5, "max": 3, "default": 1.6, "role": "thickness"},
        ]})
    v = GeometryValidator(s)
    assert v.validate(s.default_vector()).valid
    x = s.default_vector(); x[2] = 0.15
    assert not v.validate(x).valid
