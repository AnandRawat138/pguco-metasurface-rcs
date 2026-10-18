"""Robust counterfactual study for the AEU master manuscript (Song et al. 2016 baseline), openEMS cell database.

    PYTHONPATH=. python examples/07_robust_cf_study.py --grid outputs/openems/grid --out outputs/robust_cf [--seeds 0 1 2 3 4]

1. Calibrate f_LF (Eq. 17): circuit coefficients fitted to the openEMS sweep of the baseline cell only; report the phase
   error against openEMS and against the values read from Fig. 2 of Song et al.
2. Baseline (published rule) and best of K random layouts (seed xi varied), high fidelity.
3. Proposed method (Algorithm 1), N_max evaluations, S seeds.
4. Surrogate-free baselines at the same budget (random search, GA, PSO), S seeds, and long runs to measure how many
   evaluations they need to reach the proposed method's final worst-case reduction.
5. Ablation (Table 6), S seeds.
6. Data for the figures: calibration curves, rho(f) of baseline/optimized with the eps_r spread, bistatic cuts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from metasurface_optimizer.optimization import robust_cf as rc
from metasurface_optimizer.solvers.ring_disc_array import FIG2_F_GHZ, FIG2_PHASE_DEG, FIG2_R1_MM

HERE = Path(__file__).resolve().parents[1]


def changed(x, tol=0.01):
    d = np.abs(rc.to_u(x) - rc.to_u(rc.X0))
    return [rc.NAMES[i] for i in range(7) if d[i] > tol]


def first_reach(curve, level):
    return next((p["n_hf"] for p in curve if p["best_rho_wc"] >= level - 1e-9), None)


def bistatic_cut(model: rc.ArrayModel, x, f_ghz, theta_i=0.0, eps=4.4, pol="TE"):
    """sigma(theta_s)/sigma_plate,spec in the plane of incidence (phi = 0/180), physical optics with HF cells."""
    R1 = model.layout(np.asarray(x, float))
    k2, k3, h = x[4], x[5], x[6]
    j = int(np.argmin(np.abs(rc.F_BAND - f_ghz)))
    G = rc._apply(model.cells.table(k2, k3, h, eps), rc._weights(model.r1_grid, R1.ravel()))[:, j]
    if theta_i:
        G = G * rc._apply(model.lf.ratio(model.lf.r1, k2, k3, h, eps, theta_i, pol), rc._weights(model.lf.r1, R1.ravel()))[:, j]
    n = R1.shape[0]
    P = rc.PERIOD * 1e-3
    xs = (np.arange(n) - (n - 1) / 2) * P
    X, Y = np.meshgrid(xs, xs[::-1])
    k = 2 * np.pi * f_ghz * 1e9 / 299792458.0
    th = np.radians(np.arange(-90, 90.01, 0.5))
    ix = np.sin(np.radians(theta_i))
    q = k * (np.sin(th) + ix)  # phi_s = 0 plane, incidence from phi = 180 side so that specular is at +theta_i
    af = np.array([np.sum(G * np.exp(1j * qq * X.ravel())) for qq in q]) * P ** 2 * np.sinc(q * P / (2 * np.pi))
    sig = 4 * np.pi * (k / (2 * np.pi)) ** 2 * np.cos(th) ** 2 * np.abs(af) ** 2
    plate = 4 * np.pi * (k / (2 * np.pi)) ** 2 * (n * n * P ** 2) ** 2 * np.cos(np.radians(theta_i)) ** 2
    return np.degrees(th).tolist(), (10 * np.log10(sig / plate)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", default="outputs/robust_cf")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--nmax", type=int, default=40)
    ap.add_argument("--long", type=int, default=600)
    ap.add_argument("--k-layouts", type=int, default=40)
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    R = {}
    t0 = time.time()

    hf = rc.load_hf(HERE / a.grid)
    lf = rc.LFCells()
    # ---- 1. calibration of f_LF on the baseline cell only
    ih, ie = list(hf.h).index(3.0), int(np.argmin(np.abs(hf.eps - 4.4)))
    base_tab = hf.table(0.92, 0.80, 3.0, 4.4)  # (R1 grid, band) from openEMS
    f_all = hf.f
    g_all = np.array([[complex(v) for v in row] for row in
                      (hf.G[ih, ie][list(np.round(hf.k2, 4)).index(0.92), list(np.round(hf.gap, 4)).index(0.12)])])
    uncal = lf.gamma(f_all, hf.r1, 0.92, 0.80, 3.0, 4.4)
    e0 = np.degrees(np.abs(np.angle(uncal * np.conj(g_all))))
    fit = lf.fit(hf.r1, f_all, g_all)
    fig2_oems, fig2_lf = [], []
    for k, fg in enumerate(FIG2_F_GHZ):
        j = int(np.argmin(np.abs(f_all - fg)))
        go = np.interp(FIG2_R1_MM, hf.r1, g_all[:, j].real) + 1j * np.interp(FIG2_R1_MM, hf.r1, g_all[:, j].imag)
        gl = lf.gamma([fg], FIG2_R1_MM, 0.92, 0.80, 3.0, 4.4)[:, 0]
        ref = np.exp(1j * np.radians(FIG2_PHASE_DEG[k]))
        fig2_oems += list(np.degrees(np.abs(np.angle(go * np.conj(ref)))))
        fig2_lf += list(np.degrees(np.abs(np.angle(gl * np.conj(ref)))))
    R["calibration"] = {"lf_vs_openems_before_fit_rms_deg": float(np.sqrt(np.mean(e0 ** 2))), **{f"lf_vs_openems_{k}": v for k, v in fit.items()},
                        "openems_vs_fig2_mean_deg": float(np.mean(fig2_oems)), "openems_vs_fig2_rms_deg": float(np.sqrt(np.mean(np.square(fig2_oems)))),
                        "lf_vs_fig2_rms_deg": float(np.sqrt(np.mean(np.square(fig2_lf)))),
                        "curves": {"r1": hf.r1.tolist(), "f": f_all.tolist(), "openems_phase_deg": np.degrees(np.angle(g_all)).tolist(),
                                   "lf_phase_deg": np.degrees(np.angle(lf.gamma(f_all, hf.r1, 0.92, 0.80, 3.0, 4.4))).tolist()}}
    print("calibration", {k: v for k, v in R["calibration"].items() if k != "curves"}, flush=True)

    Mh, Ml = rc.ArrayModel(hf, lf), rc.ArrayModel(lf, lf)
    hf_obj, lf_obj = rc.Objective(Mh), rc.Objective(Ml)
    t = time.time(); hf_obj(rc.X0); t_hf = time.time() - t
    t = time.time(); lf_obj(rc.X0); t_lf = time.time() - t
    R["timing"] = {"hf_eval_s": t_hf, "lf_eval_s": t_lf, "note": "HF evaluation interpolates the precomputed openEMS database"}

    # ---- 2. baseline and best of K layouts
    R["baseline"] = rc.report(Mh, rc.X0)
    R["baseline_lf"] = rc.report(Ml, rc.X0)
    lay = []
    for s in range(1, a.k_layouts + 1):
        lay.append((s, rc.report(rc.ArrayModel(hf, lf, seed=s), rc.X0)))
    bs, br = max(lay, key=lambda v: v[1]["rho_wc"])
    R["best_of_K"] = {"K": a.k_layouts, "seed": bs, **br, "rho_wc_all": [v[1]["rho_wc"] for v in lay]}
    print("baseline", R["baseline"], "\nbest of K", {k: v for k, v in R["best_of_K"].items() if k != "rho_wc_all"}, flush=True)

    # ---- 3. proposed
    runs = []
    for s in a.seeds:
        r = rc.CounterfactualOptimizer(hf_obj, lf_obj, rc.CFConfig(n_max=a.nmax, seed=s)).run()
        r["report"] = rc.report(Mh, np.array(r["x_best"])); r["changed"] = changed(r["x_best"]); r["seed"] = s
        runs.append(r)
        print(f"proposed seed {s}: rho_wc {r['y_best']['rho_wc']:.2f} J {r['y_best']['J']:.2f} x {np.round(r['x_best'], 3)} changed {r['changed']} ({r['time_s']:.0f} s)", flush=True)
    R["proposed"] = runs

    # ---- 4. baselines at the same budget and long runs
    B = {}
    for name, fn in (("random_search", rc.random_search), ("genetic_algorithm", rc.genetic_algorithm), ("particle_swarm", rc.particle_swarm)):
        B[name] = []
        for s in a.seeds:
            r = fn(hf_obj, a.nmax, s); r["seed"] = s; r["report"] = rc.report(Mh, np.array(r["x_best"]))
            B[name].append(r)
        print(name, [round(r["y_best"]["rho_wc"], 2) for r in B[name]], flush=True)
    R["baselines"] = B
    level = float(np.mean([r["y_best"]["rho_wc"] for r in runs]))
    L = {}
    for name, fn in (("random_search", rc.random_search), ("genetic_algorithm", rc.genetic_algorithm), ("particle_swarm", rc.particle_swarm)):
        L[name] = []
        for s in a.seeds:
            r = fn(hf_obj, a.long, s)
            L[name].append({"seed": s, "reach": first_reach(r["curve"], level), "final_rho_wc": r["y_best"]["rho_wc"],
                            "curve": r["curve"][:: max(1, a.long // 200)]})
        print("long", name, [(v["reach"], round(v["final_rho_wc"], 2)) for v in L[name]], flush=True)
    R["long_runs"] = {"level_rho_wc": level, "budget": a.long, "runs": L}

    # ---- 5. ablation
    if not a.skip_ablation:
        variants = {
            "full": (dict(), dict()),
            "no_physics_prior": (dict(physics_prior=False), dict()),
            "no_phase_error_term": (dict(), dict(sigma=0.0)),
            "nominal_objective": (dict(), dict(robust=False)),
            "no_counterfactual_penalty": (dict(), dict(lam_cf=0.0)),
            "mean_only_acquisition": (dict(uncertainty=False), dict()),
        }
        A = {}
        for name, (ckw, okw) in variants.items():
            if name == "full":
                A[name] = [{"seed": r["seed"], "rho_wc": r["report"]["rho_wc"], "changed": len(r["changed"]), "x": r["x_best"]} for r in runs]
                continue
            A[name] = []
            for s in a.seeds:
                r = rc.CounterfactualOptimizer(rc.Objective(Mh, **okw), rc.Objective(Ml, **okw), rc.CFConfig(n_max=a.nmax, seed=s, **ckw)).run()
                rep = rc.report(Mh, np.array(r["x_best"]))  # reported with the full robust objective and phase-error term
                A[name].append({"seed": s, "rho_wc": rep["rho_wc"], "changed": len(changed(r["x_best"])), "x": r["x_best"]})
            print("ablation", name, [round(v["rho_wc"], 2) for v in A[name]], flush=True)
        R["ablation"] = A

    # ---- 6. figure data
    best = max(runs, key=lambda r: r["report"]["rho_wc"])
    xb = np.array(best["x_best"])
    R["best_seed"] = best["seed"]
    spec = {}
    for name, x in (("baseline", rc.X0), ("optimized", xb)):
        r = Mh.rho(x)
        spec[name] = {"nominal": r[rc.EPS_SET.index(4.4), 0, 0].tolist(), "eps_min": r[:, 0, 0].min(0).tolist(), "eps_max": r[:, 0, 0].max(0).tolist(),
                      "worst_all": r.min(axis=(0, 1, 2)).tolist()}
    R["spectra"] = {"f": rc.F_BAND.tolist(), **spec}
    cuts = {}
    for name, x in (("baseline", rc.X0), ("optimized", xb)):
        for th in (0.0, 30.0):
            for fg in (7.2, 8.0, 9.0, 10.8, 12.0):
                cuts.setdefault(name, {}).setdefault(f"{th:g}", {})[f"{fg:g}"] = bistatic_cut(Mh, x, fg, th)
    R["bistatic"] = cuts
    R["elapsed_s"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(R, default=float))
    print("done", round(R["elapsed_s"]), "s", flush=True)


if __name__ == "__main__":
    main()
