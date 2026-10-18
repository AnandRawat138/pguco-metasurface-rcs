# Data for the AEU manuscript

"Physics-guided, simulation-efficient counterfactual optimization of a polarization-insensitive multi-resonant
metasurface for ultra-broadband backscatter RCS reduction" (baseline: Y.-C. Song et al., IEEE AWPL 15, 329-331, 2016).

All full-wave data were computed with openEMS 0.37 (FDTD) using `metasurface_optimizer/solvers/openems_unit_cell.py`:
quarter unit cell with PMC walls normal to x and PEC walls normal to y (exact for the infinite periodic array at normal
incidence, E along y), PEC ground at z = 0, FR4 with tan delta = 0.02, zero-thickness PEC ring (outer R1, inner kappa2 R1)
and disc (kappa3 R1), TEM waveguide port, reflection coefficient de-embedded to the substrate surface and calibrated on
the empty cell. Period 10 mm. Values are complex Gamma at normal incidence.

## `openems/cell_database/` (used for the paper's results)

`grid_w0.csv`, `grid_w1.csv`, `grid_w2.csv` (one file per parallel worker), columns
`k2, gap, h, eps, r1_mm, f_ghz, re, im`: kappa2 = R2/R1, gap = kappa2 - kappa3, substrate thickness h (mm), eps_r,
outer radius R1 (mm), frequency (GHz), Re/Im of Gamma. Mesh 0.05 mm, 71 frequencies 6-13 GHz.

* **Complete slice used in the paper: h = 3.0 mm**, R1 = 1.5:0.25:4.5 mm (13) x kappa2 {0.86, 0.92, 0.933} x
  gap {0.07, 0.12, 0.20} x eps_r {4.2, 4.4, 4.6} = 351 cells (8.8 solver-hours, 49-181 s per cell).
* The files also contain incomplete slices (h = 2.4 mm, h = 3.2 mm) from runs that were stopped; they are ignored by
  `metasurface_optimizer.optimization.robust_cf.HFCells` and were not used.

## `openems/song2016_cell_table_res035.csv`

Baseline cell family only (kappa2 = 0.92, kappa3 = 0.80, h = 3 mm, eps_r = 4.4), R1 = 1.5:0.125:4.5 mm, mesh 0.035 mm,
columns `r1_mm, f_ghz, re, im` (used by the earlier study `examples/06_song2016_openems_validation.py`).

## `results/`

* `robust_cf_results.json`: everything behind Section 6 of the manuscript (calibration, baseline, best of 40 layouts,
  proposed method over 5 seeds with convergence curves and out-of-sample predictions, random search / GA / PSO,
  600-evaluation runs, ablation, spectra and bistatic cuts). Produced by
  `PYTHONPATH=. python examples/07_robust_cf_study.py --grid data/openems/cell_database --out <dir>`.
* `song2016_*`: results of the earlier, non-robust study with the 0.035 mm baseline-family table
  (`examples/06_song2016_openems_validation.py`).

## Reproduce

```
pip install -e .[dev]                      # plus openEMS 0.37 Python bindings (CSXCAD, openEMS) for new cell simulations
python -m metasurface_optimizer.solvers.openems_cell_grid --out <dir> --worker 0 --nworkers 1   # rebuild the database (hours)
PYTHONPATH=. python examples/07_robust_cf_study.py --grid data/openems/cell_database --out outputs/robust_cf
```

The study itself (given the database) runs in about 10 minutes on a desktop CPU.
