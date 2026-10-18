"""Full-wave (openEMS) cell database for the robust counterfactual study of the Song et al. (2016) metasurface.

Grid (normal incidence, Gamma de-embedded to the substrate surface, see ``openems_unit_cell``):
  R1     1.5 ... 4.5 mm in 0.25 mm steps (13)
  kappa2 = R2/R1          {0.86, 0.92, 0.933}
  gamma  = kappa2-kappa3  {0.07, 0.12, 0.20}   (ring-disc gap ratio; the baseline has 0.12)
  h      {3.0, 3.2} mm (the published 3.0 mm and the standard 3.2 mm FR4 stock)
  eps_r  {4.2, 4.4, 4.6}  (the substrate tolerance set of the robust objective)
Gridding the gap ratio instead of kappa3 keeps every grid cell physically meaningful (no ring-disc short).

    python -m metasurface_optimizer.solvers.openems_cell_grid --out outputs/openems/grid --worker 0 --nworkers 3 --threads 5
Each worker writes grid_w<k>.csv (k2,gap,h,eps,r1_mm,f_ghz,re,im) and skips combinations already present.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

R1 = list(np.round(np.arange(1.5, 4.5001, 0.25), 3))
K2 = [0.86, 0.92, 0.933]
GAP = [0.07, 0.12, 0.20]
H = [3.0, 3.2]
EPS = [4.2, 4.4, 4.6]
COLS = ["k2", "gap", "h", "eps", "r1_mm", "f_ghz", "re", "im"]


def combos():
    return list(itertools.product(H, EPS, K2, GAP, R1))


def run_worker(out_dir: Path, worker: int, nworkers: int, res: float, threads: int) -> Path:
    from metasurface_optimizer.solvers.openems_unit_cell import simulate_cell

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"grid_w{worker}.csv"
    done = set()
    if path.exists():
        with path.open() as fh:
            done = {tuple(round(float(r[k]), 4) for k in COLS[:5]) for r in csv.DictReader(fh)}
    mine = [c for i, c in enumerate(combos()) if i % nworkers == worker]
    new = not path.exists()
    for (h, eps, k2, gap, r1) in mine:
        key = tuple(round(v, 4) for v in (k2, gap, h, eps, r1))
        if key in done:
            continue
        r = simulate_cell(r1, k2=k2, k3=k2 - gap, h=h, eps_r=eps, res=res, threads=threads)
        with path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(COLS)
                new = False
            for f, g in zip(r["f"], r["gamma"]):
                w.writerow([k2, gap, h, eps, r1, round(f / 1e9, 6), float(g.real), float(g.imag)])
        print(f"w{worker} h={h} eps={eps} k2={k2} gap={gap} R1={r1}: {r['runtime_s']:.0f} s, |G| max {np.abs(r['gamma']).max():.3f}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    run_worker(Path(a.out), a.worker, a.nworkers, a.res, a.threads)
