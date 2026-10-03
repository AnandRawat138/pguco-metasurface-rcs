"""Final outputs of a simulation-driven study, under the names the workflow promises.

The best design is the best *simulated* design. Its surrogate prediction is
exported separately and labeled PREDICTED. With MOCK results nothing is named
"hfss_validated": the files say MOCK instead.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from hfss_worker.protocol import write_json_atomic
from metasurface_optimizer.core.provenance import Fidelity


def export_final_outputs(builder, manager, out_dir: Path) -> Dict[str, Any]:
    from metasurface_optimizer.experiment import pguco_convergence, write_artifacts

    opt = manager.opt
    out_dir = Path(out_dir)
    full = out_dir / "full_artifacts"
    res = opt.results()
    res["simulation"] = manager.summary()
    res["convergence"] = pguco_convergence(opt)
    write_artifacts(builder, opt, res, full)
    best = next(r for r in opt.hf_records() if r.record_id == res["best_validated"]["record_id"])
    mock = best.fidelity is Fidelity.MOCK
    hf_name = "MOCK_response_NOT_PHYSICS" if mock else ("hfss_validated" if best.fidelity is Fidelity.SIMULATED_HF else f"{best.fidelity.value.lower()}")
    files = {
        "optimized_parameters.json": full / "optimized" / "optimized_parameters.json",
        "optimized_geometry.stl": full / "optimized" / "optimized.stl",
        "predicted_response.csv": full / "metrics" / "predicted_response.csv",
        f"{hf_name}_response.csv": full / "metrics" / "validated_response.csv",
        f"{hf_name}_rcs.csv": full / "metrics" / "validated_rcs.csv",
        "baseline_vs_optimized.png": full / "plots" / "baseline_vs_optimized.png",
        "pareto_front.png": full / "plots" / "pareto_front.png",
        "sensitivity_analysis.png": full / "plots" / "sensitivity.png",
        "experiment_report.pdf": full / "optimization_report.pdf",
    }
    written, missing = {}, []
    for name, src in files.items():
        if src.is_file() and src.stat().st_size > 0:
            shutil.copy2(src, out_dir / name)
            written[name] = str(out_dir / name)
        else:
            missing.append(name)
    p = manager.plot_convergence(out_dir / "optimization_history.png")
    if p:
        written["optimization_history.png"] = str(p)
    rep = {**manager.budget_report(), "feedback_summary": manager.summary()["feedback_summary"], "convergence": manager.convergence,
           "best_design_fidelity": best.fidelity.display, "baseline_check": manager.baseline_check, "override": manager.override}
    write_json_atomic(out_dir / "simulation_budget_report.json", rep)
    written["simulation_budget_report.json"] = str(out_dir / "simulation_budget_report.json")
    if mock:
        (out_dir / "README_MOCK.txt").write_text("Every simulated number in this folder comes from the MOCK solver (synthetic analytic oracle).\n"
                                                 "It tests the software pipeline only. It is NOT physics and must not be used as a result.\n")
    return {"written": written, "missing": missing, "fidelity": best.fidelity.display}
