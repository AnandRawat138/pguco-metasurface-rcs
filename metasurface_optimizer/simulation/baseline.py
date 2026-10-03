"""Baseline reproduction: does the HFSS template reproduce the existing design's known response?

Before any optimization run on a real template, the baseline design x0 is
simulated through the same pipeline and compared with the reference response
(the user's own earlier simulation, a measurement, or values from the paper).
Above the thresholds the framework warns and blocks automatic optimization:
optimizing a model that does not reproduce the starting design is meaningless.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.electromagnetics.rcs import to_dbsm
from metasurface_optimizer.electromagnetics.response import EMResponse

DEFAULT_THRESHOLDS = {"response_rmse_db": 3.0, "resonance_shift_rel": 0.03, "peak_difference_db": 5.0}


def _db(z):
    return 20 * np.log10(np.maximum(np.abs(z), 1e-12))


def _curve_metrics(f: np.ndarray, ref: np.ndarray, rep: np.ndarray) -> Dict[str, float]:
    i, j = int(np.argmin(ref)), int(np.argmin(rep))
    return {"rmse_db": float(np.sqrt(np.mean((ref - rep) ** 2))), "peak_ref_db": float(ref[i]), "peak_rep_db": float(rep[j]),
            "peak_difference_db": float(abs(ref[i] - rep[j])), "resonance_ref_hz": float(f[i]), "resonance_rep_hz": float(f[j]),
            "resonance_shift_hz": float(f[j] - f[i]), "resonance_shift_rel": float(abs(f[j] - f[i]) / max(f[i], 1.0))}


def compare_responses(reference: EMResponse, reproduced: EMResponse, thresholds: Optional[Dict[str, float]] = None,
                      n_points: int = 201) -> Dict[str, Any]:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    curves: Dict[str, Dict[str, float]] = {}
    if reference.sparams is not None and reproduced.sparams is not None:
        for key in sorted(set(reference.sparams.data) & set(reproduced.sparams.data)):
            fa, fb = reference.sparams.frequency, reproduced.sparams.frequency
            lo, hi = max(fa.min(), fb.min()), min(fa.max(), fb.max())
            if hi <= lo:
                continue
            f = np.linspace(lo, hi, n_points)
            ra = _db(np.interp(f, fa, reference.sparams.data[key].real) + 1j * np.interp(f, fa, reference.sparams.data[key].imag))
            rb = _db(np.interp(f, fb, reproduced.sparams.data[key].real) + 1j * np.interp(f, fb, reproduced.sparams.data[key].imag))
            curves[f"S{key[0]}{key[1]}_{key[2]}"] = _curve_metrics(f, ra, rb)
    if reference.rcs is not None and reproduced.rcs is not None:
        A, B = reference.rcs, reproduced.rcs
        try:
            sa, sb = A.specular(), B.specular()
        except ValueError:
            sa = sb = None
        if sa is not None:
            for pol in sorted(set(A.polarizations) & set(B.polarizations)):
                for th_ in A.theta_inc:
                    if np.min(np.abs(B.theta_inc - th_)) > 1e-6:
                        continue
                    lo, hi = max(A.frequency.min(), B.frequency.min()), min(A.frequency.max(), B.frequency.max())
                    if hi <= lo:
                        continue
                    f = np.linspace(lo, hi, n_points)
                    ra = np.interp(f, A.frequency, to_dbsm(sa[A.pol_index(pol), A.inc_index(th_)]))
                    rb = np.interp(f, B.frequency, to_dbsm(sb[B.pol_index(pol), B.inc_index(th_)]))
                    curves[f"RCS_{pol}_{th_:g}deg"] = _curve_metrics(f, ra, rb)
    if not curves:
        return {"passed": False, "comparable": False, "reason": "reference and reproduction share no S-parameter key or RCS polarization/angle/band",
                "thresholds": th}
    worst = {"response_rmse_db": max(c["rmse_db"] for c in curves.values()),
             "resonance_shift_rel": max(c["resonance_shift_rel"] for c in curves.values()),
             "peak_difference_db": max(c["peak_difference_db"] for c in curves.values())}
    failed = [k for k, v in worst.items() if v > th[k]]
    physical = reference.fidelity.is_physical and reproduced.fidelity is Fidelity.SIMULATED_HF
    return {"passed": not failed, "comparable": True, "failed_metrics": failed, "worst": worst, "curves": curves, "thresholds": th,
            "reference_fidelity": reference.fidelity.value, "reproduced_fidelity": reproduced.fidelity.value,
            "physically_meaningful": physical,
            "message": ("Baseline reproduced within thresholds." if not failed else
                        "WARNING: the template does not reproduce the baseline (" + ", ".join(f"{k}={worst[k]:.3g} > {th[k]:g}" for k in failed)
                        + "). Automatic optimization is blocked until the template is fixed or the user overrides with a recorded reason.")
            + ("" if physical else " (Not a physical check: one side is not a full-wave/measured response.)")}
