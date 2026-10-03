"""Configuration-driven experiments: build, run, benchmark and export.

Usage::

    python -m metasurface_optimizer.experiment configs/experiments/demo_checkerboard.yaml --benchmark

Everything an experiment needs is read from the YAML configuration (no
hard-coded paths, bands or constants); the complete resolved configuration
is written next to the results as ``config/experiment_config.json``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from metasurface_optimizer.core.logging_utils import get_logger
from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.core.seeding import set_global_seed
from metasurface_optimizer.electromagnetics.rcs import to_dbsm
from metasurface_optimizer.evaluation import plots
from metasurface_optimizer.evaluation.metrics import objective_deltas
from metasurface_optimizer.evaluation.report_generator import build_report, save_report
from metasurface_optimizer.evaluation.robustness import tolerance_analysis
from metasurface_optimizer.evaluation.statistics import convergence_band, summarize
from metasurface_optimizer.geometry.geometry_generator import GeometryGenerator
from metasurface_optimizer.geometry.geometry_validator import GeometryValidator
from metasurface_optimizer.geometry.parameter_schema import ParameterSchema
from metasurface_optimizer.geometry.stl_exporter import export_stl
from metasurface_optimizer.models.torch_common import TrainConfig
from metasurface_optimizer.optimization.acquisition import AcquisitionConfig
from metasurface_optimizer.optimization.active_learning import CounterfactualOptimizer, PGUCOConfig, TrustRegionConfig
from metasurface_optimizer.optimization.baselines import run_baseline
from metasurface_optimizer.optimization.counterfactual import CounterfactualConfig
from metasurface_optimizer.optimization.nsga2 import NSGA2Config
from metasurface_optimizer.optimization.objectives import ObjectiveNormalizer, ObjectiveSet
from metasurface_optimizer.optimization.pareto import pareto_mask
from metasurface_optimizer.optimization.problem import OptimizationProblem
from metasurface_optimizer.optimization.stopping import StoppingConfig
from metasurface_optimizer.physics.config import PhysicsConfig
from metasurface_optimizer.solvers.analytic_oracle import AnalyticCheckerboardOracle
from metasurface_optimizer.solvers.base import BudgetedEvaluator, HighFidelitySolver
from metasurface_optimizer.tracking.experiment import ExperimentRecord, SQLiteRepository, dumps

log = get_logger(__name__)
_PRIOR_CACHE: Dict[str, Any] = {}


_NUM = __import__("re").compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def coerce_numbers(o: Any) -> Any:
    """PyYAML (YAML 1.1) reads '6.0e9' as a string; convert numeric-looking strings back to numbers."""
    if isinstance(o, dict):
        return {k: coerce_numbers(v) for k, v in o.items()}
    if isinstance(o, list):
        return [coerce_numbers(v) for v in o]
    if isinstance(o, str) and _NUM.match(o.strip()):
        v = float(o)
        return int(v) if v.is_integer() and "." not in o and "e" not in o.lower() else v
    return o


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg = coerce_numbers(yaml.safe_load(Path(path).read_text()))
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def _dc(cls, d: Optional[Dict[str, Any]]):
    """Instantiate a dataclass from a dict, ignoring unknown keys (they are reported)."""
    d = dict(d or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        log.warning("%s: ignoring unknown config keys %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in d.items() if k in names})


class ExperimentBuilder:
    def __init__(self, cfg: Dict[str, Any], root: Optional[Path] = None):
        self.cfg = cfg
        if root is None:  # resolve relative paths against the first ancestor of the config file that contains them
            cands = [Path(cfg["_config_path"]).parent, *Path(cfg["_config_path"]).parents] if cfg.get("_config_path") else []
            root = next((c for c in cands if (c / cfg["schema"]).exists()), Path("."))
        self.root = Path(root)
        self.schema = ParameterSchema.load(self.root / cfg["schema"])
        self.generator = GeometryGenerator(self.schema) if self.schema.template else None

    # --------------------------------------------------------------- pieces
    def x0(self) -> np.ndarray:
        vals = (self.cfg.get("baseline") or {}).get("values")
        if not vals:
            return self.schema.default_vector()
        full = {p.name: p.default for p in self.schema.parameters}
        full.update(vals)
        return self.schema.to_vector(full)

    def physics(self) -> PhysicsConfig:
        return _dc(PhysicsConfig, self.cfg.get("physics"))

    def problem(self, physics: Optional[PhysicsConfig] = None, objectives_cfg=None) -> OptimizationProblem:
        objs = ObjectiveSet.from_config(objectives_cfg or self.cfg["objectives"])
        return OptimizationProblem(self.schema, objs, GeometryValidator(self.schema, self.generator), ObjectiveNormalizer(self.cfg.get("normalization", "baseline_relative")),
                                   self.x0(), physics=physics or self.physics(), lambda_c=float(self.cfg.get("lambda_c", 10.0)))

    def solver(self) -> HighFidelitySolver:
        hf = self.cfg["high_fidelity"]
        if hf["kind"] == "synthetic_oracle":
            return AnalyticCheckerboardOracle(hf.get("conditions"), self.schema.metadata.get("template_map"))
        raise ValueError(f"High-fidelity kind {hf['kind']!r} cannot run automatically; use the human-in-the-loop API")

    def pguco_config(self, seed: int, overrides: Optional[Dict[str, Any]] = None) -> PGUCOConfig:
        o = copy.deepcopy(self.cfg.get("optimizer") or {})
        if overrides:
            for k, v in overrides.items():
                if isinstance(v, dict) and isinstance(o.get(k), dict):
                    o[k].update(v)
                else:
                    o[k] = v
        nested = {"acquisition": AcquisitionConfig, "counterfactual": CounterfactualConfig, "stopping": StoppingConfig,
                  "trust_region": TrustRegionConfig, "train": TrainConfig, "nsga": NSGA2Config}
        kw = {k: _dc(cls, o.pop(k)) for k, cls in nested.items() if k in o}
        cfg = _dc(PGUCOConfig, {**o, **kw})
        cfg.seed = seed
        return cfg

    def prior_feature_fn(self, enabled: Optional[bool] = None):
        t = self.cfg.get("transfer") or {}
        if not (t.get("enabled") if enabled is None else enabled):
            return None
        from metasurface_optimizer.models.transfer import CellPriorModel, PriorFeatureMap

        key = json.dumps(t, sort_keys=True)
        if key not in _PRIOR_CACHE:
            if t.get("prior", "synthetic_cell_prior") == "synthetic_cell_prior":
                from metasurface_optimizer.data.public_datasets.synthetic_prior import SyntheticCellPrior

                data = SyntheticCellPrior(n_cells=int(t.get("n_cells", 600)), seed=int(t.get("seed", 123))).load()
            else:
                raise ValueError("Only parameterised cell priors can feed the cell prior model; mask datasets (FSSDataBase) need the mask encoder path")
            _PRIOR_CACHE[key] = CellPriorModel(seed=int(t.get("seed", 123))).pretrain(data, epochs=int(t.get("epochs", 60)))
        band = next(o["band"] for o in self.cfg["objectives"] if o.get("band"))
        return PriorFeatureMap(_PRIOR_CACHE[key], self.schema, (float(band[0]), float(band[1])), int(t.get("n_freq", 5)))

    def budget(self) -> int:
        return int(self.cfg["high_fidelity"]["budget"])


# ------------------------------------------------------------------ PGUCO run
def run_pguco(builder: ExperimentBuilder, seed: int, baseline_response=None, overrides: Optional[Dict[str, Any]] = None,
              physics: Optional[PhysicsConfig] = None, transfer: Optional[bool] = None, budget: Optional[int] = None,
              objectives_cfg=None) -> Tuple[CounterfactualOptimizer, Dict[str, Any]]:
    set_global_seed(seed)
    prob = builder.problem(physics, objectives_cfg)
    solver = builder.solver()
    ev = BudgetedEvaluator(solver, budget or builder.budget())
    opt = CounterfactualOptimizer(prob, builder.pguco_config(seed, overrides), builder.prior_feature_fn(transfer))
    res = opt.run(ev, baseline_response)
    res["convergence"] = pguco_convergence(opt)
    res["solver"] = solver.describe()
    return opt, res


def pguco_convergence(opt: CounterfactualOptimizer) -> List[Dict[str, float]]:
    out, best, n = [], np.inf, 0
    for r in opt.hf_records():
        c = opt.composite_of(r.J, r.x)
        if r.meta.get("counted", True):
            n += 1
        best = min(best, c)
        if r.meta.get("counted", True):
            out.append({"n_hf": n, "best": best})
    return out


# ------------------------------------------------------------------ artifacts
def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def _response_rows(resp, label: str) -> List[Dict[str, Any]]:
    rows = []
    if resp is None:
        return rows
    if resp.sparams is not None:
        tab = resp.sparams.to_table()
        for i in range(len(resp.sparams.frequency)):
            rows.append({"label": label, "fidelity": resp.fidelity.value, **{k: float(v[i]) for k, v in tab.items()}})
    return rows


def _rcs_rows(resp, label: str) -> List[Dict[str, Any]]:
    rows = []
    if resp is None or resp.rcs is None:
        return rows
    r = resp.rcs
    spec = r.specular()
    for pi, p in enumerate(r.polarizations):
        for ii, th in enumerate(r.theta_inc):
            for k, f in enumerate(r.frequency):
                row = {"label": label, "fidelity": resp.fidelity.value, "pol": p, "theta_inc": float(th), "frequency_hz": float(f),
                       "specular_dbsm": float(to_dbsm(spec[pi, ii, k]))}
                if r.reference is not None:
                    row["reference_dbsm"] = float(to_dbsm(r.reference[pi, ii, k]))
                    row["reduction_db"] = row["specular_dbsm"] - row["reference_dbsm"]
                rows.append(row)
    return rows


def write_artifacts(builder: ExperimentBuilder, opt: CounterfactualOptimizer, res: Dict[str, Any], out_dir: Path,
                    baseline_comparison: Optional[Dict[str, Any]] = None, ablation: Optional[Dict[str, Any]] = None,
                    extra_figures: Optional[List[str]] = None) -> Dict[str, Any]:
    cfg = builder.cfg
    schema = builder.schema
    dirs = {k: out_dir / k for k in ("baseline", "optimized", "candidates", "plots", "metrics", "config")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    prob = opt.problem
    recs = opt.hf_records()
    base_rec = recs[0]
    best_rec = next(r for r in recs if r.record_id == res["best_validated"]["record_id"])
    synthetic = best_rec.fidelity is Fidelity.SYNTHETIC_ORACLE

    # ---- parameters
    def param_doc(rec, role):
        return {"role": role, "values": schema.to_dict_values(rec.x), "units": {p.name: p.unit for p in schema.parameters},
                "response_fidelity": rec.fidelity.value, "response_label": rec.fidelity.display,
                "objectives": dict(zip(prob.objectives.names, rec.J.tolist())), "composite": opt.composite_of(rec.J, rec.x),
                "geometry_validation": prob.validator.validate(rec.x).to_dict(), "schema": schema.name, "schema_fingerprint": schema.fingerprint()}

    (dirs["baseline"] / "baseline_parameters.json").write_text(dumps(param_doc(base_rec, "baseline (existing design x0)"), indent=2))
    (dirs["optimized"] / "optimized_parameters.json").write_text(dumps(param_doc(best_rec, "best validated design x*"), indent=2))

    # ---- geometry images + STL
    figs: List[str] = []
    if builder.generator is not None:
        for rec, d, name, col in ((base_rec, dirs["baseline"], "baseline", plots.SERIES[0]), (best_rec, dirs["optimized"], "optimized", plots.SERIES[1])):
            geom = builder.generator.build(rec.x)
            f = plots.plot_geometry(geom, title=f"{name} geometry ({schema.length_unit})", color=col)
            f.savefig(d / f"{name}_geometry.png"); plots.plt.close(f)
            val = export_stl(geom, d / f"{name}.stl", unit=schema.length_unit)
            (d / f"{name}_stl_validation.json").write_text(dumps(val.to_dict(), indent=2))
        figs += [str(dirs["baseline"] / "baseline_geometry.png"), str(dirs["optimized"] / "optimized_geometry.png")]

    # ---- responses
    pred_best = opt.predict(best_rec.x[None])
    dec = opt.encoder.decode(pred_best["Y"][0], pred_best["S"][0], origin=opt.surrogate.name)
    pred_rows = []
    for i, f in enumerate(dec["frequency"]):
        row = {"label": "optimized (best validated design) - surrogate prediction", "fidelity": "PREDICTED", "frequency_hz": float(f)}
        for name, s in opt.encoder.slices.items():
            if name != "J":
                row[name] = float(pred_best["Y"][0][s][i]); row[name + "_std"] = float(pred_best["S"][0][s][i])
        pred_rows.append(row)
    _write_csv(dirs["metrics"] / "predicted_response.csv", pred_rows)
    _write_csv(dirs["metrics"] / "validated_response.csv", _response_rows(base_rec.response, "baseline") + _response_rows(best_rec.response, "optimized"))
    _write_csv(dirs["metrics"] / "validated_rcs.csv", _rcs_rows(base_rec.response, "baseline") + _rcs_rows(best_rec.response, "optimized"))

    band = next((o.band for o in prob.objectives.objectives if o.band), None)
    f = plots.plot_baseline_vs_optimized(base_rec.response, best_rec.response, dec, band)
    f.savefig(dirs["plots"] / "baseline_vs_optimized.png"); plots.plt.close(f)
    figs.append(str(dirs["plots"] / "baseline_vs_optimized.png"))

    # ---- objective table
    deltas = objective_deltas(base_rec.J, best_rec.J, prob.objectives.names, [o.kind for o in prob.objectives.objectives])
    best_pred = None
    if opt.last_proposals:
        bp = min(opt.last_proposals, key=lambda p: p["mu"])
        best_pred = {k: bp[k] for k in ("values", "mu", "sigma", "mu_J", "sigma_J", "kind", "reason", "label")}
    rows = []
    for k, n in enumerate(prob.objectives.names):
        rows.append({"objective": n, "kind": prob.objectives.objectives[k].kind, "baseline": float(base_rec.J[k]), "baseline_label": base_rec.fidelity.display,
                     "optimized_validated": float(best_rec.J[k]), "optimized_label": best_rec.fidelity.display,
                     "delta_pct": deltas[n]["delta_pct"], "optimized_predicted_mu": float(pred_best["mu_J"][0][k]),
                     "optimized_predicted_sigma": float(pred_best["sigma_J"][0][k]), "predicted_label": "PREDICTED",
                     "best_unvalidated_candidate_mu": None if best_pred is None else best_pred["mu_J"][k]})
    rows.append({"objective": "composite", "kind": "weighted normalised", "baseline": opt.composite_of(base_rec.J, base_rec.x),
                 "optimized_validated": opt.composite_of(best_rec.J, best_rec.x), "optimized_predicted_mu": float(pred_best["mu"][0]),
                 "optimized_predicted_sigma": float(pred_best["sigma"][0]), "best_unvalidated_candidate_mu": None if best_pred is None else best_pred["mu"]})
    _write_csv(out_dir / "baseline_vs_optimized.csv", rows)

    # ---- candidates
    cand_rows = []
    for rank, p in enumerate(sorted(opt.last_proposals, key=lambda p: -p["acquisition"]["score"]), 1):
        main = max(p["modifications"], key=lambda m: abs(m["change"]) / max(schema[m["parameter"]].span, 1e-12)) if p["modifications"] else None
        cand_rows.append({"rank": rank, "candidate": f"C{rank}", "label": "PREDICTED", "predicted_objective": p["mu"], "uncertainty": p["sigma"],
                          "main_modification": f"{main['parameter']}: {main['original']:.4g} -> {main['new']:.4g} {main['unit']}" if main else "",
                          "validity": "valid" if p["valid_geometry"] else "invalid", "kind": p["kind"], "reason": p["reason"],
                          "predicted_effect": p["predicted_effect"], **{f"x_{k}": v for k, v in p["values"].items()}})
    pool = getattr(opt, "last_pool", None)
    if pool is not None:
        for j, i in enumerate(pool["order"][:50]):
            cand_rows.append({"rank": f"pool-{j + 1}", "candidate": f"P{j + 1}", "label": "PREDICTED", "predicted_objective": float(pool["mu"][i]),
                              "uncertainty": float(pool["sigma"][i]), "kind": pool["kinds"][i], "validity": "valid",
                              **{f"x_{k}": v for k, v in schema.to_dict_values(pool["X"][i]).items()}})
    _write_csv(dirs["candidates"] / "candidate_rankings.csv", cand_rows)
    (dirs["candidates"] / "candidate_explanations.json").write_text(dumps(opt.last_proposals, indent=2))

    # ---- sensitivity / pareto / uncertainty / convergence plots
    sens = {}
    if res.get("hf_sensitivity"):
        sens["high-fidelity (initial design, at x0)"] = res["hf_sensitivity"]["ranking"]
    if res.get("surrogate_sensitivity"):
        sens["surrogate (final incumbent)"] = res["surrogate_sensitivity"]["ranking"]
    if sens:
        f = plots.plot_sensitivity(sens); f.savefig(dirs["plots"] / "sensitivity.png"); plots.plt.close(f)
        figs.append(str(dirs["plots"] / "sensitivity.png"))
    Fv = prob.normalized(np.vstack([r.J for r in recs]))
    Fp = prob.normalized(pool["mu_J"]) if pool is not None else None
    f = plots.plot_pareto(Fv, prob.objectives.names, Fp, prob.normalized(prob.J0[None])[0], np.where(pareto_mask(Fv))[0])
    f.savefig(dirs["plots"] / "pareto_front.png"); plots.plt.close(f)
    f = plots.plot_uncertainty(opt.prediction_log); f.savefig(dirs["plots"] / "uncertainty.png"); plots.plt.close(f)
    curves = {"PGUCO (proposed)": convergence_band([res["convergence"]], builder.budget())}
    if baseline_comparison:
        curves = baseline_comparison["curves"]
    f = plots.plot_convergence(curves); f.savefig(dirs["plots"] / "convergence.png"); plots.plt.close(f)
    figs += [str(dirs["plots"] / n) for n in ("pareto_front.png", "uncertainty.png", "convergence.png")] + (extra_figures or [])

    # ---- robustness of the optimised design (PREDICTED)
    rob = tolerance_analysis(best_rec.x, schema, lambda X: opt.predict(X)["mu"], lambda x: prob.validator.is_valid(x, generic=False))
    (dirs["metrics"] / "tolerance_robustness_PREDICTED.json").write_text(dumps(rob, indent=2))

    # ---- extrapolation distance of the best predicted candidate
    ext = None
    if best_pred is not None:
        U = schema.normalize(np.vstack([r.x for r in recs]))[:, schema.free_mask]
        u = schema.normalize(schema.to_vector(best_pred["values"]))[schema.free_mask]
        ext = float(np.sqrt(((U - u) ** 2).sum(1)).min() / np.sqrt(len(u)))

    # ---- metrics + report
    (dirs["metrics"] / "results.json").write_text(dumps(res, indent=2))
    (dirs["metrics"] / "dataset.json").write_text(dumps([r.to_dict() for r in opt.dataset.records], indent=1))
    missing = [p.name for p in schema.parameters if p.role in ("permittivity", "loss_tangent") and p.fixed and cfg.get("baseline", {}).get(f"{p.name}_source") == "default"]
    ctx = {"results": res, "objectives": prob.objectives.to_config(), "normalization": prob.normalizer.to_dict(), "schema_name": schema.name,
           "d_free": int(schema.free_mask.sum()), "physics": prob.physics.to_dict(), "acquisition": dataclasses.asdict(opt.cfg.acquisition),
           "objective_mode": opt.cfg.objective_mode, "counterfactual_config": dataclasses.asdict(opt.cfg.counterfactual),
           "surrogate_summary": opt.cfg.surrogate, "synthetic": synthetic, "geometry_confidence": (cfg.get("baseline") or {}).get("geometry_confidence"),
           "final_proposals": opt.last_proposals, "extrapolation_distance": ext, "missing_materials": missing,
           "objective_deltas": deltas, "best_predicted": best_pred, "best_predicted_unvalidated": best_pred is not None,
           "baseline_comparison": (baseline_comparison or {}).get("table"), "ablation": ablation,
           "transfer_synthetic": bool((cfg.get("transfer") or {}).get("enabled")) and (cfg.get("transfer") or {}).get("prior") == "synthetic_cell_prior"}
    rep = build_report(ctx)
    paths = save_report(out_dir, rep, figs)

    # ---- experiment record (Sec. 31) + registry
    rec = ExperimentRecord(
        name=cfg.get("name", "experiment"), random_seed=int(opt.cfg.seed), parameter_schema=schema.to_dict(),
        objective_definition=prob.objectives.to_config(), normalization=prob.normalizer.to_dict(),
        optimizer={"name": "PGUCO", **opt.cfg.to_dict()}, acquisition_function=dataclasses.asdict(opt.cfg.acquisition),
        physics_constraints=prob.physics.to_dict(), training_configuration={"surrogate": res.get("surrogate"), "features": res.get("features"), "train": dataclasses.asdict(opt.cfg.train)},
        candidate_pool_size=int(len(pool["X"])) if pool is not None else None, number_of_HFSS_evaluations=int(res["n_hf"]),
        high_fidelity_source=res.get("solver", {}), dataset_version=f"target:{len(opt.dataset)}records;prior:{(cfg.get('transfer') or {}).get('prior') if (cfg.get('transfer') or {}).get('enabled') else 'none'}",
        stopping_rule=dataclasses.asdict(opt.cfg.stopping),
        results={k: res[k] for k in ("baseline", "best_validated", "n_hf", "composite_improvement", "improvement_per_hf", "stop_reason", "prediction_error")},
        notes="SYNTHETIC oracle data; not full-wave" if synthetic else "")
    rec.save(dirs["config"] / "experiment_config.json")
    (dirs["config"] / "input_config.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=False))
    SQLiteRepository(out_dir.parent / "experiments.sqlite").add(rec, str(out_dir))
    return {"report": paths, "experiment_id": rec.experiment_id, "out_dir": str(out_dir)}


# ------------------------------------------------------------------ benchmark
def run_benchmark(builder: ExperimentBuilder, seeds: List[int], baselines: List[str], out_dir: Path) -> Dict[str, Any]:
    """Proposed method vs baselines under an identical HF budget, multiple seeds, every run kept."""
    solver = builder.solver()
    prob0 = builder.problem()
    base_resp = solver.evaluate(builder.schema.to_dict_values(prob0.x0))  # the existing design's response (given, not charged)
    budget = builder.budget()
    runs: Dict[str, List[Dict[str, Any]]] = {}
    for seed in seeds:
        t = time.time()
        opt, res = run_pguco(builder, seed, base_resp)
        runs.setdefault("PGUCO (proposed)", []).append({"seed": seed, "best_composite": res["best_validated"]["composite"], "n_hf": res["n_hf"],
                                                         "convergence": res["convergence"], "best_J": res["best_validated"]["J"], "time_s": time.time() - t,
                                                         "improvement_per_hf": res["improvement_per_hf"], "prediction_error": res["prediction_error"]})
        log.info("seed %d PGUCO best %.4f", seed, res["best_validated"]["composite"])
        for b in baselines:
            t = time.time()
            if b == "surrogate_no_physics":
                o2, r2 = run_pguco(builder, seed, base_resp, overrides={"surrogate": "ensemble_no_physics"}, physics=PhysicsConfig.none())
                row = {"seed": seed, "best_composite": r2["best_validated"]["composite"], "n_hf": r2["n_hf"], "convergence": r2["convergence"],
                       "best_J": r2["best_validated"]["J"], "improvement_per_hf": r2["improvement_per_hf"]}
            else:
                prob = builder.problem()
                prob.set_baseline(prob.evaluate_response(base_resp, prob.x0))
                r2 = run_baseline(b, prob, BudgetedEvaluator(builder.solver(), budget), base_resp, seed)
                row = {"seed": seed, "best_composite": r2["best_composite"], "n_hf": r2["n_hf"], "convergence": r2["convergence"],
                       "best_J": r2["best_J"], "improvement_per_hf": r2["improvement_per_hf"], "invalid_proposals": r2["invalid_proposals"]}
            row["time_s"] = time.time() - t
            runs.setdefault(b, []).append(row)
            log.info("seed %d %s best %.4f", seed, b, row["best_composite"])
    table, curves = [], {}
    for name, rs in runs.items():
        st = summarize([r["best_composite"] for r in rs])
        e = summarize([r["improvement_per_hf"] for r in rs], lower_is_better=False)
        # N_HF needed to first reach the median final PGUCO value (sample-efficiency)
        table.append({"algorithm": name, "n_hf": int(np.max([r["n_hf"] for r in rs])), "best_mean": st["mean"], "best_std": st["std"],
                      "best_median": st["median"], "best_best": st["best"], "best_worst": st["worst"], "ci_low": st.get("ci_low"), "ci_high": st.get("ci_high"),
                      "E_mean": e["mean"], "E_std": e["std"], "time_mean_s": float(np.mean([r["time_s"] for r in rs])), "n_seeds": len(rs)})
        curves[name] = convergence_band([r["convergence"] for r in rs], budget)
    target = float(np.median([r["best_composite"] for r in runs["PGUCO (proposed)"]]))
    for row in table:
        hits = []
        for r in runs[row["algorithm"]]:
            k = next((p["n_hf"] for p in r["convergence"] if p["best"] <= target), None)
            hits.append(k)
        row["n_hf_to_reach_PGUCO_median"] = None if any(h is None for h in hits) else float(np.mean(hits))
        row["runs_reaching_PGUCO_median"] = f"{sum(h is not None for h in hits)}/{len(hits)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "comparison_table.csv", table)
    (out_dir / "benchmark_runs.json").write_text(dumps({"runs": runs, "curves": curves, "budget": budget, "seeds": seeds,
                                                        "note": "All seeds reported; validated values from the SYNTHETIC oracle"}, indent=1))
    f = plots.plot_convergence(curves, "Best validated composite vs N_HF (mean +/- std over seeds; SYNTHETIC oracle)")
    f.savefig(out_dir / "convergence_comparison.png"); plots.plt.close(f)
    return {"table": table, "curves": curves, "runs": runs}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run a PGUCO experiment from a YAML configuration")
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--benchmark", action="store_true", help="also run baselines over the configured seeds")
    ap.add_argument("--ablation", action="store_true", help="also run the ablation suite (A1-A12)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    b = ExperimentBuilder(cfg)
    seed = a.seed if a.seed is not None else int(cfg.get("seed", 0))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(a.out or Path(cfg.get("output_dir", "outputs")) / f"{cfg.get('name', 'experiment')}_{stamp}")
    comp, abl, extra = None, None, []
    if a.benchmark:
        bm = cfg.get("benchmark") or {}
        comp = run_benchmark(b, list(bm.get("seeds", [0, 1, 2])), list(bm.get("baselines", [])), out / "benchmark")
        extra.append(str(out / "benchmark" / "convergence_comparison.png"))
    if a.ablation:
        from metasurface_optimizer.evaluation.ablation import run_ablation

        abl = run_ablation(b, cfg.get("ablation") or {}, out / "ablation")
    solver = b.solver()
    base_resp = solver.evaluate(b.schema.to_dict_values(b.x0()))
    opt, res = run_pguco(b, seed, base_resp)
    info = write_artifacts(b, opt, res, out, comp, abl, extra)
    print(json.dumps({"out_dir": info["out_dir"], "experiment_id": info["experiment_id"], "best_validated": res["best_validated"]["values"],
                      "label": res["best_validated"]["label"], "n_hf": res["n_hf"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
