"""FSSDataBase mask prior: cache -> train -> test-split metrics -> ablation on the SYNTHETIC demo.

    PYTHONPATH=. python examples/04_fssdatabase_mask_prior.py --root E:/FSSDataBase [--epochs 40] [--seeds 10 11 12]

Writes outputs/cache/fssdb_mask50.npz, outputs/models/fssdb_mask_prior.pt,
outputs/models/fssdb_mask_prior_metrics.json and outputs/fssdb_mask_prior_ablation/.
The ablation target is the synthetic analytic oracle, so it says nothing about full-wave performance,
and 3 seeds cannot establish significance.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from metasurface_optimizer.evaluation.ablation import run_ablation
from metasurface_optimizer.experiment import ExperimentBuilder, load_config
from metasurface_optimizer.models.mask_prior import (MaskPriorModel, build_fssdb_cache, evaluate_split, load_cache, scalar_inputs)

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="FSSDataBase root (read-only)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()
    cache, model_path = HERE / "outputs/cache/fssdb_mask50.npz", HERE / "outputs/models/fssdb_mask_prior.pt"
    if not cache.exists():
        build_fssdb_cache(a.root, cache, seed=0)
    data = load_cache(cache)
    tr, va = data["split"] == "train", data["split"] == "val"
    sc = scalar_inputs(data)
    t = time.time()
    if model_path.exists():
        model = MaskPriorModel.load(model_path)
    else:
        model = MaskPriorModel(seed=0).fit(data["masks"][tr], sc[tr], data["S"][tr], (data["masks"][va], sc[va], data["S"][va]), epochs=a.epochs)
        model.save(model_path)
    train_s = time.time() - t
    metrics = {"val": evaluate_split(model, data, "val"), "test": evaluate_split(model, data, "test"), "train_time_s": train_s,
               "epochs": len(model.history), "best_epoch": int(np.argmin([h.get("val_loss", h["train_loss"]) for h in model.history])),
               "history": model.history, "split_counts": {k: int((data["split"] == k).sum()) for k in ("train", "val", "test")}}
    (model_path.parent / "fssdb_mask_prior_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics["test"], indent=1))
    if a.skip_ablation:
        return
    cfg = load_config(HERE / "configs/experiments/demo_checkerboard.yaml")
    out = {}
    for name, prior in (("fssdatabase_mask_prior", {"prior": "fssdatabase_mask_prior", "model_path": str(model_path)}),
                        ("synthetic_cell_prior", {"prior": "synthetic_cell_prior"})):
        c = copy.deepcopy(cfg)
        c["transfer"] = {"enabled": False, **prior}
        b = ExperimentBuilder(c, HERE)
        only = ["reference", "A2_transfer"] if name == "fssdatabase_mask_prior" else ["A2_transfer"]
        res = run_ablation(b, {"only": only, "seeds": a.seeds}, HERE / f"outputs/fssdb_mask_prior_ablation/{name}")
        fm = b.prior_feature_fn(True)
        fm(b.x0()[None])
        out[name] = {"summary": res["summary"], "prior_out_of_range_fraction_at_x0": getattr(fm, "out_of_range_fraction", None)}
    (HERE / "outputs/fssdb_mask_prior_ablation/summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
