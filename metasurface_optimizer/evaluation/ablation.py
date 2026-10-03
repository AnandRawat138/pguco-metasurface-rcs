"""Ablation engine (Sec. 18): A1-A12 as configuration variants of the proposed method.

Every variant is run for every seed with the same high-fidelity budget and
the same baseline response. The best validated design of each run is
re-scored with the *full* objective set (so single-objective runs are
comparable), and all requested metrics are recorded. All runs are kept.

| id  | variant                                        |
|-----|------------------------------------------------|
| A1  | no transfer learning (prior features off)      |
| A2  | with transfer learning (prior features on)     |
| A3  | no physics constraints                         |
| A4  | physics constraints (configured set)           |
| A5  | no uncertainty (acquisition = predicted mean)  |
| A6  | uncertainty-aware acquisition (proposed)       |
| A7  | random candidate selection                     |
| A8  | proposed acquisition                           |
| A9  | single-objective (first objective only)        |
| A10 | multi-objective (Pareto-ranked acquisition)    |
| A11 | different surrogate models                     |
| A12 | different optimisation algorithms (baselines)  |

A1/A4/A6/A8 share the reference configuration and are therefore one run.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from metasurface_optimizer.core.logging_utils import get_logger
from metasurface_optimizer.evaluation.statistics import summarize
from metasurface_optimizer.physics.config import PhysicsConfig
from metasurface_optimizer.tracking.experiment import dumps

log = get_logger(__name__)


def _variants(builder, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = [
        {"id": "A1/A4/A6/A8 (reference: physics + uncertainty + proposed acquisition, no transfer)", "short": "reference"},
        {"id": "A2 transfer learning", "short": "A2_transfer", "transfer": True},
        {"id": "A3 no physics constraints", "short": "A3_no_physics", "physics": PhysicsConfig.none(), "overrides": {"surrogate": "ensemble_no_physics"}},
        {"id": "A5 no uncertainty (mean acquisition)", "short": "A5_mean", "overrides": {"acquisition": {"kind": "mean"}}},
        {"id": "A7 random candidate selection", "short": "A7_random", "overrides": {"acquisition": {"kind": "random"}}},
        {"id": "A9 single-objective", "short": "A9_single", "objectives": [builder.cfg["objectives"][0]]},
        {"id": "A10 multi-objective (Pareto)", "short": "A10_pareto", "overrides": {"objective_mode": "pareto"}},
    ]
    for s in cfg.get("surrogates", ["random_forest", "xgboost", "mlp", "gp", "cnn", "transformer"]):
        v.append({"id": f"A11 surrogate={s}", "short": f"A11_{s}", "overrides": {"surrogate": s}})
    only = cfg.get("only")
    return [x for x in v if not only or x["short"] in only]


def run_ablation(builder, cfg: Dict[str, Any], out_dir: Path, seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    from metasurface_optimizer.experiment import pguco_convergence, run_pguco, _write_csv

    seeds = seeds or list(cfg.get("seeds", (builder.cfg.get("benchmark") or {}).get("seeds", [0, 1, 2])))
    budget = int(cfg.get("budget", builder.budget()))
    solver = builder.solver()
    full = builder.problem()
    base = solver.evaluate(builder.schema.to_dict_values(full.x0))
    J0 = full.evaluate_response(base, full.x0)
    full.set_baseline(J0)
    names = full.objectives.names
    rows, runs = [], {}
    for var in _variants(builder, cfg):
        for seed in seeds:
            t = time.time()
            opt, res = run_pguco(builder, seed, base, overrides=var.get("overrides"), physics=var.get("physics"), transfer=var.get("transfer"),
                                 budget=budget, objectives_cfg=var.get("objectives"))
            best = next(r for r in opt.hf_records() if r.record_id == res["best_validated"]["record_id"])
            Jf = full.evaluate_response(best.response, best.x)
            comp = float(full.scalarize(Jf[None], best.x[None])[0])
            conv = pguco_convergence(opt)
            final = conv[-1]["best"] if conv else 0.0
            start = opt.best_composite_hist[0] if opt.best_composite_hist else 0.0
            goal = start - 0.9 * (start - final)
            n90 = next((c["n_hf"] for c in conv if c["best"] <= goal + 1e-12), None)
            pe = res.get("prediction_error") or {}
            phys_viol = sum(1 for p in opt.last_proposals if p.get("physics_audit", {}).get("predicted_passivity_violation_max", 0) > 1e-6)
            row = {"variant": var["id"], "short": var["short"], "seed": seed, "objective_value_full_composite": comp,
                   **{f"J_{n}": float(v) for n, v in zip(names, Jf)},
                   "prediction_RMSE_composite": pe.get("composite_rmse"), "prediction_MAE_composite": pe.get("composite_mae"),
                   "n_hf": res["n_hf"], "time_s": time.time() - t, "n_hf_to_90pct_of_final_improvement": n90,
                   "geometry_rejections_last_pool": opt.counterfactual.stats.get("rejected_geometry"),
                   "predicted_physics_violations_last_batch": phys_viol, "stop_reason": res.get("stop_reason")}
            rows.append(row)
            runs.setdefault(var["short"], []).append(row)
            log.info("ablation %s seed %d -> %.4f", var["short"], seed, comp)
    summary = []
    for short, rs in runs.items():
        s = summarize([r["objective_value_full_composite"] for r in rs])
        row = {"variant": rs[0]["variant"], "n_seeds": len(rs), "composite_mean": s["mean"], "composite_std": s["std"], "composite_median": s["median"],
               "composite_best": s["best"], "composite_worst": s["worst"]}
        for n in names:
            row[f"J_{n}_mean"] = float(np.mean([r[f"J_{n}"] for r in rs]))
        rm = [r["prediction_RMSE_composite"] for r in rs if r["prediction_RMSE_composite"] is not None]
        row["prediction_RMSE_mean"] = float(np.mean(rm)) if rm else None
        n90 = [r["n_hf_to_90pct_of_final_improvement"] for r in rs if r["n_hf_to_90pct_of_final_improvement"] is not None]
        row["n_hf_to_90pct_mean"] = float(np.mean(n90)) if n90 else None
        row["time_mean_s"] = float(np.mean([r["time_s"] for r in rs]))
        summary.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "ablation_runs.csv", rows)
    _write_csv(out_dir / "ablation_table.csv", summary)
    (out_dir / "ablation.json").write_text(dumps({"summary": summary, "runs": rows, "budget": budget, "seeds": seeds,
                                                  "note": "A12 (optimisers) is the benchmark comparison table; SYNTHETIC oracle"}, indent=1))
    return {"summary": summary, "budget": budget, "seeds": seeds}
