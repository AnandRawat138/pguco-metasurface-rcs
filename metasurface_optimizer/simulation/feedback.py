"""Surrogate error feedback: PREDICTED vs high-fidelity, per candidate and over the run.

For each simulated candidate that had a prediction:
    e(x) = |J_HF(x) - J_ML(x)|   (per objective, absolute and relative)
plus the composite error, the RMSE between predicted and simulated |S| (dB)
and RCS-reduction spectra, and the error in the frequency of the strongest
dip (resonance / best reduction). These are out-of-sample errors: every
prediction was made before the design was simulated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from metasurface_optimizer.electromagnetics.response import EMResponse


def _db(z: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(z), 1e-12))


def candidate_feedback(opt, x: np.ndarray, response: EMResponse, prediction: Dict[str, Any]) -> Dict[str, Any]:
    x = np.asarray(x, float)
    J_hf = np.asarray(opt.problem.evaluate_response(response, x), float)
    mu = np.asarray(prediction["mu_J"], float)
    sd = np.asarray(prediction.get("sigma_J") or np.zeros_like(mu), float)
    e = np.abs(J_hf - mu)
    rel = e / np.maximum(np.abs(J_hf), 1e-12)
    comp_hf = opt.composite_of(J_hf, x)
    out: Dict[str, Any] = {
        "objective_names": list(opt.problem.objectives.names), "J_hf": J_hf.tolist(), "J_ml": mu.tolist(), "sigma_ml": sd.tolist(),
        "abs_error": e.tolist(), "rel_error": rel.tolist(), "z_score": (e / np.maximum(sd, 1e-12)).tolist(),
        "composite_ml": prediction.get("mu"), "composite_hf": comp_hf,
        "composite_abs_error": None if prediction.get("mu") is None else abs(comp_hf - float(prediction["mu"])),
    }
    if opt.surrogate is None or opt.encoder is None:
        return out
    pred = opt.predict(x[None])
    y_ml = pred["Y"][0]
    y_hf = opt.encoder.encode(response, J_hf)
    f = opt.encoder.frequency
    sp_rmse, peak_err = {}, {}
    for sre, sim, key in opt.encoder.complex_pairs():
        a, b = _db(y_ml[sre] + 1j * y_ml[sim]), _db(y_hf[sre] + 1j * y_hf[sim])
        if np.isnan(b).all():
            continue
        name = f"S{key[0]}{key[1]}_{key[2]}"
        sp_rmse[name] = float(np.sqrt(np.nanmean((a - b) ** 2)))
        peak_err[name] = float(abs(f[int(np.nanargmin(a))] - f[int(np.nanargmin(b))]))
    rcs_rmse = {}
    for name, s in opt.encoder.slices.items():
        if name.startswith("RCSred_") and not np.isnan(y_hf[s]).all():
            rcs_rmse[name] = float(np.sqrt(np.nanmean((y_ml[s] - y_hf[s]) ** 2)))
            peak_err[name] = float(abs(f[int(np.nanargmin(y_ml[s]))] - f[int(np.nanargmin(y_hf[s]))]))
    out.update({"sparam_rmse_db": sp_rmse, "rcs_reduction_rmse_db": rcs_rmse, "peak_frequency_error_hz": peak_err,
                "response_rmse_db": float(np.mean(list(sp_rmse.values()) + list(rcs_rmse.values()))) if (sp_rmse or rcs_rmse) else None})
    return out


def summarize(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = [r for r in rows if r.get("abs_error") is not None]
    if not rows:
        return None
    E = np.array([r["abs_error"] for r in rows])
    R = np.array([r["rel_error"] for r in rows])
    comp = [r["composite_abs_error"] for r in rows if r.get("composite_abs_error") is not None]
    resp = [r["response_rmse_db"] for r in rows if r.get("response_rmse_db") is not None]
    return {"n": len(rows), "objective_names": rows[0]["objective_names"], "mae": E.mean(0).tolist(), "rmse": np.sqrt((E ** 2).mean(0)).tolist(),
            "mean_rel_error": R.mean(0).tolist(), "composite_mae": float(np.mean(comp)) if comp else None,
            "response_rmse_db_mean": float(np.mean(resp)) if resp else None}
