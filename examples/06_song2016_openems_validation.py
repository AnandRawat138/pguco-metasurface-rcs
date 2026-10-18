"""Song et al. (2016) metasurface with openEMS full-wave unit-cell data: reproduction, optimization, baselines, ablation.

    PYTHONPATH=. python examples/06_song2016_openems_validation.py --table outputs/openems/cell_table_res035.csv

High-fidelity source in this study: openEMS FDTD reflection of the ring+disc unit cell (TEM unit cell with PEC/PMC
walls, exact for the infinite periodic array at normal incidence; ``solvers/openems_unit_cell.py``) on an R1 x f grid,
combined with physical optics over the 30 x 30 layout (local periodicity approximation; oblique incidence through the
equivalent circuit's angular ratio). No finite-array full-wave simulation is involved.

Writes outputs/song2016_openems/: fig2_reproduction.json, designs.json, benchmark/, ablation/, curves CSVs, summary.json.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np

from metasurface_optimizer.evaluation.ablation import run_ablation
from metasurface_optimizer.experiment import ExperimentBuilder, load_config, run_benchmark
from metasurface_optimizer.solvers.ring_disc_array import (FIG2_F_GHZ, FIG2_PHASE_DEG, FIG2_R1_MM, PAPER, ArrayConditions, RingDiscArrayModel,
                                                            RingDiscCell, array_rcs, layout_r1)

HERE = Path(__file__).resolve().parents[1]


def fig2_reproduction(cell: RingDiscCell):
    """Circular phase error of the openEMS cell against the values read from the paper's Fig. 2."""
    rows, err = [], []
    for k, f in enumerate(FIG2_F_GHZ):
        g = cell.gamma([f * 1e9], FIG2_R1_MM)[:, 0]
        for r, gg, ref in zip(FIG2_R1_MM, g, FIG2_PHASE_DEG[k]):
            e = float(np.degrees(np.angle(gg * np.exp(-1j * np.radians(ref)))))
            rows.append({"f_GHz": f, "r1_mm": r, "paper_deg": ref, "openems_deg_wrapped": float(np.degrees(np.angle(gg))), "error_deg": e})
            err.append(abs(e))
    per_f = {f"{f:g}GHz": float(np.mean([abs(r["error_deg"]) for r in rows if r["f_GHz"] == f])) for f in FIG2_F_GHZ}
    small = [abs(r["error_deg"]) for r in rows if r["r1_mm"] <= 2.5]
    return {"mean_abs_deg": float(np.mean(err)), "median_abs_deg": float(np.median(err)), "max_abs_deg": float(np.max(err)),
            "mean_abs_deg_by_frequency": per_f, "mean_abs_deg_r1_le_2p5": float(np.mean(small)), "points": rows}


def reduction(resp, kind="monostatic"):
    r = resp.rcs
    s = r.monostatic if kind == "monostatic" else r.specular()
    return 10 * np.log10(s / r.reference)


def metrics(resp, band=(7.0, 12.0)):
    f = resp.rcs.frequency / 1e9
    m = (f >= band[0] - 1e-9) & (f <= band[1] + 1e-9)
    mono = reduction(resp)
    spec = reduction(resp, "specular") if resp.rcs.bistatic is not None else None
    below = mono[0, 0, m] <= -10
    out = {"worst_dB": float(mono[:, 0, m].max()), "mean_dB": float(mono[:, 0, m].mean()), "fraction_below_-10dB": float(np.mean(mono[:, 0, m] <= -10)),
           "band_below_-10dB_GHz": [float(f[m][below].min()), float(f[m][below].max())] if below.any() else None}
    if spec is not None:
        out["specular_mean_dB"] = {f"{t:g}deg": float(spec[:, i, m].mean()) for i, t in enumerate(resp.rcs.theta_inc)}
        out["specular_worst_dB"] = {f"{t:g}deg": float(spec[:, i, m].max()) for i, t in enumerate(resp.rcs.theta_inc)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--out", default="outputs/song2016_openems")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    table = str((HERE / a.table).resolve())
    cell = RingDiscCell.from_table(table)
    rep = fig2_reproduction(cell)
    (out / "fig2_reproduction.json").write_text(json.dumps(rep, indent=1))
    print("Fig. 2 reproduction: mean |err| %.1f deg, median %.1f, max %.1f" % (rep["mean_abs_deg"], rep["median_abs_deg"], rep["max_abs_deg"]), flush=True)

    cfg = load_config(HERE / "configs/experiments/song2016_rcs.yaml")
    cfg["high_fidelity"]["conditions"].update({"cell_table": table, "bistatic_grid": "specular"})
    cfg["name"] = "song2016_rcs_openems_cells"
    b = ExperimentBuilder(cfg, HERE)
    solver = b.solver()
    prob = b.problem()
    x0 = prob.x0
    base = solver.evaluate(b.schema.to_dict_values(x0))
    J0 = prob.evaluate_response(base, x0)
    print("baseline J", dict(zip(prob.objectives.names, np.round(J0, 3))), flush=True)

    t0 = time.time()
    bench = None if a.skip_benchmark else run_benchmark(b, a.seeds, ["random_perturbation", "local_search", "differential_evolution",
                                                                   "bayesian_optimization", "surrogate_no_physics"], out / "benchmark")
    print("benchmark %.0f s" % (time.time() - t0), flush=True)
    t0 = time.time()
    abl = None if a.skip_ablation else run_ablation(b, {"only": ["reference", "A3_no_physics", "A5_mean", "A7_random", "A9_single", "A11_gp",
                                                                 "A11_random_forest"], "seeds": a.seeds}, out / "ablation", a.seeds)
    print("ablation %.0f s" % (time.time() - t0), flush=True)

    # best PGUCO design over the seeds (from the benchmark runs)
    designs = {"baseline_paper_rule": b.schema.to_dict_values(x0)}
    if bench is not None:
        best = min(bench["runs"]["PGUCO (proposed)"], key=lambda r: r["best_composite"])
        designs["pguco_best_seed"] = best["seed"]
    # re-run the best seed to recover x (benchmark rows keep J, not x)
    from metasurface_optimizer.experiment import run_pguco
    seed = designs.get("pguco_best_seed", a.seeds[0])
    opt, res = run_pguco(b, seed, base)
    rec = next(r for r in opt.hf_records() if r.record_id == res["best_validated"]["record_id"])
    designs["optimized"] = b.schema.to_dict_values(rec.x)
    designs["optimized_J"] = dict(zip(prob.objectives.names, map(float, prob.evaluate_response(rec.response, rec.x))))
    designs["baseline_J"] = dict(zip(prob.objectives.names, map(float, J0)))

    # full-grid evaluations for the figures
    full = RingDiscArrayModel({**cfg["high_fidelity"]["conditions"], "bistatic_grid": "full", "theta_s_step": 1.0, "phi_s_step": 15.0}, HERE, bistatic=True)
    resp = {"baseline_paper_rule": full.evaluate(designs["baseline_paper_rule"]), "optimized": full.evaluate(designs["optimized"]),
            "uniform_R1_4mm": full.evaluate({**designs["baseline_paper_rule"], "r1_min": 4.0, "r1_max": 4.0})}
    # the published layout (R1 from the paper's own Fig. 2 phase curve) evaluated with the openEMS cells
    fig2_layout, _ = layout_r1(RingDiscCell(), PAPER, 30, 1)
    resp["published_layout_fig2_mapping"] = None
    c = ArrayConditions.from_mapping({**cfg["high_fidelity"]["conditions"], "theta_s_step": 1.0})
    from metasurface_optimizer.electromagnetics.response import EMResponse
    from metasurface_optimizer.core.provenance import Fidelity, Provenance
    resp["published_layout_fig2_mapping"] = EMResponse(Provenance(Fidelity.SYNTHETIC_ORACLE, "openEMS cells + PO", "published layout"), None,
                                                      array_rcs(cell, fig2_layout, 10.0, c, bistatic=True))
    summ = {k: metrics(v) for k, v in resp.items()}
    f = resp["baseline_paper_rule"].rcs.frequency / 1e9
    with (out / "reduction_curves.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        keys = list(resp)
        w.writerow(["f_GHz"] + [f"{k}_TE_0deg_dB" for k in keys] + [f"{k}_TE_spec30_dB" for k in keys] + [f"{k}_TM_spec30_dB" for k in keys])
        R = {k: reduction(v) for k, v in resp.items()}
        S = {k: reduction(v, "specular") for k, v in resp.items()}
        for i in range(len(f)):
            w.writerow([round(f[i], 4)] + [R[k][0, 0, i] for k in keys] + [S[k][0, 2, i] for k in keys] + [S[k][1, 2, i] for k in keys])
    # bistatic cuts (phi = 0/180 plane, normal incidence, TE)
    cuts = {}
    for k in ("baseline_paper_rule", "optimized"):
        r = resp[k].rcs
        ts, ps = r.theta_s, r.phi_s
        j0, j180 = int(np.argmin(np.abs(ps - 0))), int(np.argmin(np.abs(ps - 180)))
        for fg in (7.2, 8.0, 9.0, 10.8, 12.0):
            kf = int(np.argmin(np.abs(f - fg)))
            th = np.concatenate([-ts[::-1], ts[1:]])
            sig = np.concatenate([r.bistatic[0, 0, kf, ::-1, j180], r.bistatic[0, 0, kf, 1:, j0]])
            ref = r.reference[0, 0, kf]
            cuts.setdefault(k, {})[f"{fg:g}"] = {"theta": th.tolist(), "rel_dB": (10 * np.log10(sig / ref)).tolist()}
    (out / "bistatic_cuts.json").write_text(json.dumps(cuts))
    designs["metrics"] = summ
    designs["fig2_reproduction"] = {k: rep[k] for k in rep if k != "points"}
    designs["hf_source"] = "openEMS FDTD unit cell (R1 x f table) + physical optics over the 30x30 layout (local periodicity)"
    if bench is not None:
        designs["benchmark_table"] = bench["table"]
    if abl is not None:
        designs["ablation_summary"] = abl["summary"]
    (out / "summary.json").write_text(json.dumps(designs, indent=1, default=float))
    print(json.dumps({k: designs[k] for k in ("optimized", "metrics")}, indent=1, default=float))


if __name__ == "__main__":
    main()
