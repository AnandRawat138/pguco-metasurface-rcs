"""openEMS unit-cell solver: PEC reference and one ring+disc cell against the paper's Fig. 2 (skipped without openEMS)."""
import numpy as np
import pytest

oec = pytest.importorskip("metasurface_optimizer.solvers.openems_unit_cell")
try:
    oec._import()
except Exception as e:  # pragma: no cover - depends on the local installation
    pytest.skip(f"openEMS not available: {e}", allow_module_level=True)


def test_pec_ground_reference_phase():
    r = oec.reference_run(res=0.25, n_freq=5)
    ph = np.degrees(np.angle(-r["gamma"]))  # Gamma = -1 -> 0 deg
    assert np.abs(ph).max() < 1.0
    assert np.all(np.abs(np.abs(r["gamma"]) - 1) < 0.06)  # uncalibrated port: ~4 % magnitude offset


def test_small_ring_matches_fig2():
    from metasurface_optimizer.solvers.ring_disc_array import FIG2_F_GHZ, FIG2_PHASE_DEG

    r = oec.simulate_cell(1.5, res=0.1, n_freq=71)
    assert np.all(np.abs(r["gamma"]) <= 1.0 + 1e-3)  # passive after calibration
    ph = np.degrees(np.angle(np.interp(FIG2_F_GHZ * 1e9, r["f"], r["gamma"].real) + 1j * np.interp(FIG2_F_GHZ * 1e9, r["f"], r["gamma"].imag)))
    err = np.degrees(np.angle(np.exp(1j * np.radians(ph - FIG2_PHASE_DEG[:, 0]))))
    assert np.abs(err).max() < 10.0  # R1 = 1.5 mm column of Fig. 2, read by hand (+-10 deg)
