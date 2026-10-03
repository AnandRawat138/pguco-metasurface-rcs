import numpy as np
import pytest

from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.electromagnetics.rcs import RCSData, to_dbsm
from metasurface_optimizer.electromagnetics.response_features import angular_entropy, threshold_bandwidth
from metasurface_optimizer.electromagnetics.sparameter import SParameterSet, from_columns, read_touchstone, write_touchstone
from metasurface_optimizer.solvers.analytic_oracle import cell_reflection


def test_touchstone_roundtrip(tmp_path):
    f = np.linspace(8e9, 12e9, 11)
    s11 = 0.5 * np.exp(1j * np.linspace(0, 3, 11)); s21 = 0.3 * np.exp(-1j * np.linspace(0, 2, 11))
    s = SParameterSet(f, {(1, 1, "TE"): s11, (2, 1, "TE"): s21, (1, 2, "TE"): s21, (2, 2, "TE"): s11})
    p = tmp_path / "x.s2p"
    write_touchstone(s, p)
    r = read_touchstone(p)
    assert np.allclose(r.get(1, 1), s11) and np.allclose(r.get(2, 1), s21)
    assert np.allclose(r.frequency, f)


def test_from_columns_complex_and_magnitude_only():
    f = [8, 9, 10]
    s = from_columns(f, {"S11_re": [0.1, 0.2, 0.3], "S11_im": [0, 0.1, 0], "S21_TM_db": [-3, -4, -5]})
    assert s.has(1, 1, "TE") and np.isclose(s.frequency[0], 8e9)
    assert (2, 1, "TM") in s.magnitude_only_db and not s.has(2, 1, "TM")  # phase is never invented
    assert np.allclose(s.db(2, 1, "TM"), [-3, -4, -5])


def test_resample_refuses_extrapolation():
    s = SParameterSet(np.linspace(8e9, 12e9, 5), {(1, 1, "TE"): np.ones(5)})
    with pytest.raises(ValueError):
        s.resample(np.linspace(7e9, 12e9, 5))


def test_cell_reflection_is_passive():
    f = np.linspace(2e9, 20e9, 200)
    for pol in ("TE", "TM"):
        for th in (0, 30, 60):
            g = cell_reflection(f, 10e-3, 8e-3, 6e-3, 10, 2e-3, 4.4, 0.02, th, pol)
            assert np.all(np.abs(g) <= 1 + 1e-9)


def test_cell_reflection_lossless_unit_magnitude():
    f = np.linspace(5e9, 15e9, 50)
    g = cell_reflection(f, 10e-3, 8e-3, 8e-3, 0, 2e-3, 3.0, 0.0, 0, "TE", rs=0.0)
    assert np.allclose(np.abs(g), 1.0, atol=1e-9)


def test_oracle_pec_reference_and_specular(oracle, schema):
    r = oracle.evaluate(schema.to_dict_values(schema.default_vector()))
    assert r.fidelity is Fidelity.SYNTHETIC_ORACLE and not r.fidelity.is_prediction
    rcs = r.rcs
    spec = rcs.specular()
    assert spec.shape == rcs.reference.shape
    # a passive surface cannot reflect more specular power than an equal PEC plate (scalar PO)
    assert np.all(to_dbsm(spec[:, 0]) <= to_dbsm(rcs.reference[:, 0]) + 1e-6)
    # identical states -> checkerboard = uniform surface -> specular ~ |Gamma|^2 sigma_ref
    v = schema.to_dict_values(schema.default_vector()); v["length_b"] = v["length_a"]; v["width_b"] = v["width_a"]
    r2 = oracle.evaluate(v)
    g = r2.sparams.get(1, 1, "TE")
    assert np.allclose(r2.rcs.specular()[0, 0], np.abs(g) ** 2 * r2.rcs.reference[0, 0], rtol=1e-6)


def test_entropy_bounds():
    th = np.arange(0, 90, 10.0); ph = np.arange(0, 360, 30.0)
    uni = np.ones((1, 1, 1, len(th), len(ph)))
    r = RCSData([10e9], [0.0], ["TE"], 0.0, th, ph, uni)
    h_uni = angular_entropy(r, 0, 0, 0)
    peak = np.full_like(uni, 1e-12); peak[..., 3, 0] = 1.0  # theta=0 has zero solid angle
    h_peak = angular_entropy(RCSData([10e9], [0.0], ["TE"], 0.0, th, ph, peak), 0, 0, 0)
    assert 0 <= h_peak < 0.1 < h_uni <= 1.0 + 1e-9


def test_threshold_bandwidth():
    f = np.linspace(8, 12, 41)
    lvl = np.where((f >= 9) & (f <= 10), -15.0, -5.0)
    bw = threshold_bandwidth(f, lvl, -10)
    assert np.isclose(bw["f_low"], 9) and np.isclose(bw["f_high"], 10)
    assert np.isclose(bw["fractional_bandwidth"], 2 * 1 / 19)
