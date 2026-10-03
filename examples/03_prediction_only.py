"""Example 3: Mode A, prediction only, from data you already have.

Loads previously evaluated designs (here: the files written by example 2),
trains the uncertainty-aware surrogate, and ranks counterfactual candidates.
Nothing is simulated, so every output is PREDICTED.

    PYTHONPATH=. python examples/02_human_in_the_loop.py   # creates the data
    PYTHONPATH=. python examples/03_prediction_only.py
"""
import json
from pathlib import Path

from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.data.loaders.target_loader import load_response
from metasurface_optimizer.experiment import ExperimentBuilder, load_config
from metasurface_optimizer.optimization.active_learning import CounterfactualOptimizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "example_02"
if not (DATA / "baseline").exists():
    raise SystemExit("Run examples/02_human_in_the_loop.py first (it writes the data this example reads).")

builder = ExperimentBuilder(load_config(ROOT / "configs/experiments/demo_checkerboard.yaml"), ROOT)
opt = CounterfactualOptimizer(builder.problem(), builder.pguco_config(seed=10))


def load(d: Path):
    return load_response(Fidelity.SYNTHETIC_ORACLE, d / "sparams.csv", d / "rcs.csv", origin=str(d), freq_unit="hz")


opt.set_baseline(load(DATA / "baseline"))
for d in sorted(p for p in DATA.iterdir() if p.is_dir() and p.name != "baseline" and (p / "rcs.csv").exists()):
    values = json.loads((d / "parameters.json").read_text())
    opt.observe(builder.schema.to_vector(values), load(d), d.name)
opt.finish_initial_design()

out = opt.predict_only(8)
print(out["mode"], "-", out["label"])
for i, p in enumerate(out["candidates"], 1):
    print(f"{i}. {p['kind']:<16} predicted J = {p['mu']:.3f} +/- {p['sigma']:.3f}   {p['predicted_effect']}")
