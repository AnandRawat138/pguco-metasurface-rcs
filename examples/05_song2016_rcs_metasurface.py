"""Optimise the Song et al. (2016) random-phase ring+disc metasurface on the APPROXIMATE model, then write HFSS job batches.

    PYTHONPATH=. python examples/05_song2016_rcs_metasurface.py [--seeds 10 11 12] [--out outputs/song2016]

Steps
1. Baseline = the paper's design rule (f = 7.2/8/9/10.8 GHz, R1 1.5-4.5 mm, full phase span) on the
   Fig.-2-calibrated circuit + physical-optics model (SYNTHETIC_ORACLE, not HFSS).
2. PGUCO (trust-region active learning, budget 36 model evaluations) for each seed; all runs are kept.
3. Best design over the seeds is re-evaluated and compared with the baseline (curves, layouts, tables).
4. Layout CSVs (R1 per cell) for the baseline, the best design and the uniform R1 = 4 mm array are written
   for the HFSS Template-B (finite array) batch.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metasurface_optimizer.experiment import ExperimentBuilder, load_config, run_pguco
from metasurface_optimizer.solvers.ring_disc_array import PAPER

HERE = Path(__file__).resolve().parents[1]


def reduction(resp, kind="monostatic"):
    r = resp.rcs
    s = r.monostatic if kind == "monostatic" else r.specular()
    return 10 * np.log10(s / r.reference)


def write_layout(path: Path, R1: np.ndarray, sub: np.ndarray, period: float = 10.0) -> None:
    n = R1.shape[0]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "x_mm", "y_mm", "subarray", "r1_mm"])
        for i in range(n):
            for j in range(n):
                w.writerow([i, j, round((j - (n - 1) / 2) * period, 4), round(((n - 1) / 2 - i) * period, 4), int(sub[i, j]) + 1, round(float(R1[i, j]), 4)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--out", default="outputs/song2016")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    b = ExperimentBuilder(load_config(HERE / "configs/experiments/song2016_rcs.yaml"), HERE)
    solver = b.solver()
    prob = b.problem()
    x0 = prob.x0
    base = solver.evaluate(b.schema.to_dict_values(x0))
    J0 = prob.evaluate_response(base, x0)
    prob.set_baseline(J0)
    runs = []
    for seed in a.seeds:
        opt, res = run_pguco(b, seed, base)
        best = next(r for r in opt.hf_records() if r.record_id == res["best_validated"]["record_id"])
        Jb = prob.evaluate_response(best.response, best.x)
        runs.append({"seed": seed, "x": best.x.tolist(), "values": b.schema.to_dict_values(best.x), "J": dict(zip(prob.objectives.names, map(float, Jb))),
                     "composite": float(prob.scalarize(Jb[None], best.x[None])[0]), "n_hf": res["n_hf"], "stop_reason": res.get("stop_reason"),
                     "convergence": res["convergence"], "prediction_error": res.get("prediction_error")})
        print(f"seed {seed}: composite {runs[-1]['composite']:.4f}  J {runs[-1]['J']}", flush=True)
    best_run = min(runs, key=lambda r: r["composite"])
    xb = np.asarray(best_run["x"])
    rb = solver.evaluate(b.schema.to_dict_values(xb))
    f = base.rcs.frequency / 1e9
    band = (f >= 7) & (f <= 12)
    rows = []
    for name, resp in (("baseline_paper", base), ("optimized", rb)):
        mono = reduction(resp)
        spec = reduction(resp, "specular")
        rows.append({"design": name, "worst_dB_7_12GHz_normal": float(mono[:, 0, band].max()), "mean_dB_7_12GHz_normal": float(mono[:, 0, band].mean()),
                     "band_below_minus10dB_GHz": [float(f[band][mono[0, 0, band] <= -10].min()) if (mono[0, 0, band] <= -10).any() else None,
                                                  float(f[band][mono[0, 0, band] <= -10].max()) if (mono[0, 0, band] <= -10).any() else None],
                     "fraction_of_7_12GHz_below_minus10dB": float(np.mean(mono[:, 0, band] <= -10)),
                     "specular_mean_dB_by_incidence": {f"{t:g}deg": float(spec[:, i, band].mean()) for i, t in enumerate(resp.rcs.theta_inc)}})
    unif = solver.evaluate({**b.schema.to_dict_values(x0), "r1_min": 4.0, "r1_max": 4.0})
    with (out / "reduction_curves.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_GHz", "paper_TE_0deg_dB", "optimized_TE_0deg_dB", "uniform_R1_4mm_TE_0deg_dB",
                    "paper_TE_30deg_specular_dB", "optimized_TE_30deg_specular_dB"])
        for k in range(len(f)):
            w.writerow([round(f[k], 4), reduction(base)[0, 0, k], reduction(rb)[0, 0, k], reduction(unif)[0, 0, k],
                        reduction(base, "specular")[0, 2, k], reduction(rb, "specular")[0, 2, k]])
    lay = out / "layouts"
    lay.mkdir(exist_ok=True)
    for name, v in (("baseline_paper", b.schema.to_dict_values(x0)), ("optimized", b.schema.to_dict_values(xb)),
                    ("uniform_R1_4mm", {**b.schema.to_dict_values(x0), "r1_min": 4.0, "r1_max": 4.0})):
        R1, sub, _ = solver.layout(v)
        write_layout(lay / f"{name}.csv", R1, sub)
    summary = {"label": "APPROXIMATE MODEL (Fig.-2-calibrated circuit + physical optics) - NOT HFSS; validate with the HFSS batches",
               "paper_design": {k: PAPER[k] for k in ("f1", "f2", "f3", "f4", "phase_span", "r1_min", "r1_max")},
               "optimized_design": {k: best_run["values"][k] for k in ("f1", "f2", "f3", "f4", "phase_span", "r1_min", "r1_max")},
               "best_seed": best_run["seed"], "comparison": rows, "runs": runs, "solver": solver.describe(),
               "J_baseline": dict(zip(prob.objectives.names, map(float, J0)))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(f, reduction(base)[0, 0], label="paper design (model)")
        ax[0].plot(f, reduction(rb)[0, 0], label="optimized (model)")
        ax[0].plot(f, reduction(unif)[0, 0], ":", label="uniform R1 = 4 mm (model)")
        ax[0].axhline(-10, color="k", lw=0.8, ls="--")
        ax[0].axvspan(7, 12, color="0.92")
        ax[0].set(xlabel="f (GHz)", ylabel="backscatter RCS vs PEC plate (dB)", title="Normal incidence, TE  [APPROXIMATE MODEL]")
        ax[0].legend(fontsize=8)
        for k, (name, v) in enumerate((("paper", b.schema.to_dict_values(x0)), ("optimized", b.schema.to_dict_values(xb)))):
            R1, _, _ = solver.layout(v)
            im = ax[k + 1].imshow(R1, cmap="viridis", vmin=1.5, vmax=4.5)
            ax[k + 1].set_title(f"R1 layout (mm): {name}")
            fig.colorbar(im, ax=ax[k + 1], fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / "baseline_vs_optimized.png", dpi=130)
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in runs:
            c = r["convergence"]
            ax.step([p["n_hf"] for p in c], [p["best"] for p in c], where="post", label=f"seed {r['seed']}")
        ax.set(xlabel="model evaluations (N_HF)", ylabel="best composite J (lower is better)", title="Convergence [APPROXIMATE MODEL]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "optimization_history.png", dpi=130)
    except Exception as e:  # plots are optional
        print("plots skipped:", e)
    print(json.dumps({k: summary[k] for k in ("paper_design", "optimized_design", "comparison")}, indent=1, default=float))


if __name__ == "__main__":
    main()
