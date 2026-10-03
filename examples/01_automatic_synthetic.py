"""Example 1: the whole PGUCO loop against the SYNTHETIC analytic oracle (demo mode).

Every "high-fidelity" value here comes from an analytic model (Luukkonen
surface impedance + physical-optics array factor), NOT from HFSS/CST, and is
labeled SYNTHETIC_ANALYTIC_ORACLE throughout.

    PYTHONPATH=. python examples/01_automatic_synthetic.py
"""
from pathlib import Path

from metasurface_optimizer.experiment import ExperimentBuilder, load_config, run_pguco, write_artifacts

ROOT = Path(__file__).resolve().parents[1]

cfg = load_config(ROOT / "configs/experiments/demo_checkerboard.yaml")
builder = ExperimentBuilder(cfg, ROOT)
opt, res = run_pguco(builder, seed=10, budget=20)  # small budget so the example runs in about a minute

print(f"high-fidelity evaluations used: {res['n_hf']}  (stop: {res['stop_reason']})")
print(f"best evaluated composite: {res['best_validated']['composite']:.3f}  [{res['best_validated']['label']}]")
print("parameters of the best evaluated design:")
for k, v in res["best_validated"]["values"].items():
    print(f"  {k:>20} = {v:.4g}")
info = write_artifacts(builder, opt, res, ROOT / "outputs" / "example_01")
print("artifacts written to", info.get("out_dir", ROOT / "outputs" / "example_01"))
