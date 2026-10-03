"""Example 2: Mode B, human in the loop, through files.

This is the workflow with a real solver:
  1. the framework writes the designs to simulate (parameters + STL),
  2. you simulate them in HFSS/CST and export S-parameters and RCS as CSV,
  3. the framework reads the files back and proposes the next counterfactuals.

To keep the example self-contained, step 2 is done by the SYNTHETIC oracle,
and its results are loaded with fidelity SYNTHETIC_ORACLE. With real
simulations, load them with Fidelity.VALIDATED_SIMULATION instead.

    PYTHONPATH=. python examples/02_human_in_the_loop.py
"""
import json
from pathlib import Path

from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.data.loaders.target_loader import load_response, write_rcs_csv, write_sparams_csv
from metasurface_optimizer.experiment import ExperimentBuilder, load_config
from metasurface_optimizer.geometry.stl_exporter import export_stl
from metasurface_optimizer.optimization.active_learning import CounterfactualOptimizer

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs" / "example_02"
WORK.mkdir(parents=True, exist_ok=True)

builder = ExperimentBuilder(load_config(ROOT / "configs/experiments/demo_checkerboard.yaml"), ROOT)
opt = CounterfactualOptimizer(builder.problem(), builder.pguco_config(seed=10))
oracle = builder.solver()  # stand-in for HFSS/CST in this example only


def simulate_externally(tag: str, values: dict) -> Path:
    """What you would do in HFSS/CST. Here: the synthetic oracle writes the same CSV files you would export."""
    d = WORK / tag
    d.mkdir(exist_ok=True)
    (d / "parameters.json").write_text(json.dumps(values, indent=2))
    resp = oracle.evaluate(values)
    write_sparams_csv(resp.sparams, d / "sparams.csv")
    write_rcs_csv(resp.rcs, d / "rcs.csv")
    return d


def read_back(d: Path):
    return load_response(Fidelity.SYNTHETIC_ORACLE, d / "sparams.csv", d / "rcs.csv", origin=str(d), freq_unit="hz")


# The existing design x0 and its response y0.
x0_values = builder.schema.to_dict_values(builder.x0())
opt.set_baseline(read_back(simulate_externally("baseline", x0_values)))

# Initial design around x0 (LHS in a small trust region + local sensitivity).
for d in opt.initial_design():
    opt.observe(d["x"], read_back(simulate_externally(d["tag"], builder.schema.to_dict_values(d["x"]))), d["tag"])
opt.finish_initial_design()
print(opt.hf_sensitivity.table())

for it in range(2):
    batch = opt.propose(4)
    print(f"\niteration {it}: {len(batch)} candidates (PREDICTED until simulated)")
    for p in batch:
        mods = ", ".join(f"{m['parameter']} {m['change']:+.3g}" for m in p["modifications"])
        print(f"  {p['kind']:<16} mu={p['mu']:.3f} sigma={p['sigma']:.3f}  {mods}  [{p['reason']}]")
        tag = f"iter{it}_{len(opt.hf_records()):02d}_{p['kind']}"
        (WORK / tag).mkdir(exist_ok=True)
        export_stl(builder.generator.build(p["x"]), WORK / tag / "design.stl")  # what you would import into HFSS/CST
        opt.observe(p["x"], read_back(simulate_externally(tag, p["values"])), tag)
    stop = opt.update_after_batch()
    print(f"  best evaluated composite now {opt.best_composite_hist[-1]:.3f}" + (f" (stop: {stop})" if stop else ""))
    if stop:
        break
print("\nfiles for each design are in", WORK)
