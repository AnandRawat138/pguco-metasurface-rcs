"""Machine-readable research report + Markdown + PDF (Sec. 47).

Sections: Problem, Method, Optimization, Physics constraints, Results
(baseline vs predicted-optimised vs validated-optimised), Ablation,
Simulation efficiency, Limitations (detected automatically).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from metasurface_optimizer.tracking.experiment import dumps


def detect_limitations(ctx: Dict[str, Any]) -> List[str]:
    """Automatic limitation statements; each is triggered by a concrete check."""
    out: List[str] = []
    res = ctx.get("results", {})
    d_free = ctx.get("d_free", 0)
    n_hf = res.get("n_hf", 0)
    if ctx.get("synthetic"):
        out.append("All 'validated' numbers come from the SYNTHETIC analytic oracle (quasi-static HIS model + scalar physical optics), "
                   "not from full-wave simulation or measurement. They demonstrate the method only; no physical performance is claimed.")
    if ctx.get("geometry_confidence") in ("LOW", "MEDIUM"):
        out.append(f"Baseline geometry reconstruction confidence is {ctx['geometry_confidence']}; optimised dimensions inherit this uncertainty.")
    if n_hf < 2 * d_free + 2:
        out.append(f"Only {n_hf} high-fidelity evaluations for {d_free} free parameters (< 2d+2): surrogate is weakly constrained.")
    pe = res.get("prediction_error")
    if pe and abs(res.get("composite_improvement", 0)) > 0 and pe.get("composite_rmse", 0) > 0.5 * abs(res["composite_improvement"]):
        out.append(f"Out-of-sample composite RMSE ({pe['composite_rmse']:.3g}) exceeds half of the achieved improvement "
                   f"({res['composite_improvement']:.3g}): predicted rankings are unreliable; rely on validated values only.")
    props = ctx.get("final_proposals") or []
    if props:
        sig = np.median([p["sigma"] for p in props])
        if sig > 0.5 * max(abs(res.get("composite_improvement", 0)), 1e-9):
            out.append(f"High surrogate uncertainty on the final proposals (median sigma {sig:.3g}).")
    if ctx.get("extrapolation_distance") is not None and ctx["extrapolation_distance"] > 0.15:
        out.append(f"Best predicted candidate lies {ctx['extrapolation_distance']:.2f} (normalised units) from the nearest evaluated design: extrapolation.")
    for m in ctx.get("missing_materials", []):
        out.append(f"Material parameter '{m}' was not supplied and uses a default value.")
    worse = [n for n, d in ctx.get("objective_deltas", {}).items() if d["delta"] > 1e-9]
    if worse:
        out.append("The best validated design is NOT better on every objective: worse on " + ", ".join(worse) +
                   ". 'Better' is relative to the user-selected weighting/Pareto preference.")
    if ctx.get("best_predicted_unvalidated"):
        out.append("Best PREDICTED candidates beyond the evaluated set are unvalidated and require high-fidelity simulation.")
    if ctx.get("transfer_synthetic"):
        out.append("Transfer-learning prior is synthetic and shares the oracle's physics family; any transfer gain is optimistic.")
    out.append("Local sensitivities are local response sensitivities at the stated step sizes, not causal physical effects.")
    return out


def build_report(ctx: Dict[str, Any]) -> Dict[str, Any]:
    res = ctx["results"]
    return {
        "problem": {"baseline_design": res["baseline"], "objectives": ctx["objectives"], "normalization": ctx["normalization"],
                    "schema": ctx.get("schema_name"), "free_parameters": ctx.get("d_free")},
        "method": {"algorithm": res["algorithm"], "formulation": "see docs/methodology.md (Algorithm 1, Eqs. O1-O9, C1, S1-S2, CF1-CF6, E1-E2, A1-A5, L1, T1-T4)",
                   "surrogate": ctx.get("surrogate_summary"), "features": res.get("features")},
        "optimization": {"candidate_generation": ctx.get("counterfactual_config"), "acquisition": ctx.get("acquisition"),
                         "objective_mode": ctx.get("objective_mode"), "stop_reason": res.get("stop_reason"), "iterations": len(res.get("history", []))},
        "physics_constraints": ctx.get("physics"),
        "results": {
            "baseline": res["baseline"], "best_validated": res["best_validated"], "best_predicted": ctx.get("best_predicted"),
            "objective_deltas": ctx.get("objective_deltas"), "pareto_validated_count": len(res.get("pareto_validated", [])),
            "labels": {"best_validated": res["best_validated"].get("label"), "best_predicted": "PREDICTED"},
            "prediction_error": res.get("prediction_error"),
        },
        "ablation": ctx.get("ablation"),
        "simulation_efficiency": {"n_hf": res["n_hf"], "composite_improvement": res["composite_improvement"],
                                  "improvement_per_hf": res["improvement_per_hf"], "baselines": ctx.get("baseline_comparison")},
        "limitations": detect_limitations(ctx),
    }


def report_markdown(rep: Dict[str, Any]) -> str:
    r = rep["results"]
    bv = r["best_validated"]
    lines = ["# Optimization report", ""]
    lines += ["## Problem", f"Baseline design: `{json.dumps(rep['problem']['baseline_design']['values'])}`", "",
              f"Objectives (all minimised): " + ", ".join(o["name"] + f" ({o['kind']}, w={o['weight']})" for o in rep["problem"]["objectives"]),
              f"Normalisation: {rep['problem']['normalization']}", ""]
    lines += ["## Method", rep["method"]["algorithm"], f"Formulation: {rep['method']['formulation']}", ""]
    lines += ["## Optimization", f"Objective mode: {rep['optimization']['objective_mode']}; acquisition: {rep['optimization']['acquisition']}",
              f"Iterations: {rep['optimization']['iterations']}; stop reason: {rep['optimization']['stop_reason']}", ""]
    lines += ["## Physics constraints", "```", json.dumps(rep["physics_constraints"], indent=1), "```", ""]
    lines += ["## Results", f"Best validated design [{bv.get('label')}]: `{json.dumps(bv['values'])}`", "",
              "| Objective | Baseline | Best validated | Delta % |", "|---|---:|---:|---:|"]
    for n, d in (r.get("objective_deltas") or {}).items():
        lines.append(f"| {n} | {d['old']:.4g} | {d['new']:.4g} | {d['delta_pct']:+.1f} |")
    if r.get("best_predicted"):
        lines += ["", f"Best PREDICTED (unvalidated) candidate: `{json.dumps(r['best_predicted'].get('values'))}` "
                      f"composite mu = {r['best_predicted'].get('mu'):.4g} +/- {r['best_predicted'].get('sigma'):.3g}"]
    if r.get("prediction_error"):
        pe = r["prediction_error"]
        lines += ["", f"Out-of-sample prediction error (n={pe['n']}): composite RMSE {pe['composite_rmse']:.4g}, MAE {pe['composite_mae']:.4g}"]
    se = rep["simulation_efficiency"]
    lines += ["", "## Simulation efficiency", f"N_HF = {se['n_hf']}; composite improvement Delta J = {se['composite_improvement']:.4g}; "
              f"improvement per high-fidelity simulation E = {se['improvement_per_hf']:.4g}", ""]
    if se.get("baselines"):
        lines += ["| Algorithm | N_HF | best composite (mean +/- std over seeds) | E |", "|---|---:|---:|---:|"]
        for row in se["baselines"]:
            lines.append(f"| {row['algorithm']} | {row['n_hf']} | {row['best_mean']:.4g} +/- {row['best_std']:.3g} | {row['E_mean']:.4g} |")
        lines.append("")
    if rep.get("ablation"):
        lines += ["## Ablation", "See ablation_table.csv for every recorded metric.", ""]
    lines += ["## Limitations"] + [f"- {l}" for l in rep["limitations"]]
    return "\n".join(lines) + "\n"


def write_pdf(path: str | Path, markdown: str, figures: List[str]) -> None:
    """Simple PDF: wrapped text pages followed by the figures (matplotlib PdfPages, no extra dependency)."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image

    lines: List[str] = []
    for raw in markdown.splitlines():
        lines += textwrap.wrap(raw, 110) or [""]
    with PdfPages(path) as pdf:
        for i in range(0, len(lines), 60):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.97, "\n".join(lines[i:i + 60]), va="top", family="monospace", fontsize=7.2)
            pdf.savefig(fig); plt.close(fig)
        for f in figures:
            if Path(f).exists():
                img = Image.open(f)
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0.05, 0.05, 0.9, 0.9]); ax.imshow(img); ax.axis("off")
                ax.set_title(Path(f).name, fontsize=8)
                pdf.savefig(fig); plt.close(fig)


def save_report(out_dir: str | Path, rep: Dict[str, Any], figures: List[str]) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(dumps(rep, indent=2))
    md = report_markdown(rep)
    (out / "report.md").write_text(md)
    write_pdf(out / "optimization_report.pdf", md, figures)
    return {"json": str(out / "report.json"), "markdown": str(out / "report.md"), "pdf": str(out / "optimization_report.pdf")}
