# Metasurface counterfactual optimizer

Physics-guided, simulation-efficient counterfactual optimization of **existing** electromagnetic
metasurfaces (RCS-reduction / diffusion / stealth surfaces, FSS, polarization surfaces).

Given an existing design `x0` and its response `y0`, the framework:

1. estimates which parameters the objectives are locally sensitive to,
2. generates counterfactual variants of `x0` (single, pairwise, correlated, gradient, trust-region, surrogate NSGA-II),
3. predicts them with an uncertainty-aware, physics-regularised surrogate ensemble,
4. ranks them with an acquisition function (EI + βσ + γ·diversity, or LCB) inside a trust region,
5. hands a small batch to you for HFSS/CST validation, learns from the validated results, and repeats,
6. exports the best validated design as parameters, per-layer STL, tables, plots and a report.

The method, with every implemented equation, is in [docs/methodology.md](docs/methodology.md).

## Honesty rules built into the code

- **Nothing here runs or imitates HFSS or CST.** `HFSSAdapter` / `CSTAdapter` are stubs that raise
  `NotImplementedError`. Real full-wave results enter only by upload.
- Every response carries a fidelity label: `PREDICTED`, `VALIDATED_SIMULATION`, `MEASURED`,
  `REPORTED`, `REPORTED_PLOT_DIGITIZED`, or `SYNTHETIC_ANALYTIC_ORACLE`. Loaded data can never be
  labeled `PREDICTED`, and predictions never enter the training set.
- The demo uses a **synthetic analytic oracle** (Luukkonen high-impedance-surface model + scalar
  physical-optics checkerboard array factor). It is labeled as such everywhere, in files, plots, the
  UI and the reports. Its numbers say nothing about real full-wave performance.
- Data partitions (TRAINING / VALIDATION / EXTERNAL) have leakage guards. Values extracted from
  papers keep their source (text, table, caption, plot-digitized) and a confidence level.
- Benchmarks report all seeds (mean, std, median, best, worst, bootstrap CI). Defaults were tuned on
  seeds {0, 1}; reported runs use disjoint seeds 10–14.

## Install

```bash
python -m pip install -e .[dev]        # numpy, scipy, torch, scikit-learn, xgboost, shapely, trimesh, pymupdf, fastapi, ...
cd ui && npm install && npm run build  # optional: the dashboard (served by the API from ui/dist)
```

## Quickstart (synthetic demo)

```bash
# Full study + 5-seed comparison against 7 baselines + ablation (all on the synthetic oracle)
PYTHONPATH=. python -m metasurface_optimizer.experiment configs/experiments/demo_checkerboard.yaml \
    --benchmark --ablation --out outputs/demo_checkerboard_synthetic

# Dashboard + API
PYTHONPATH=. uvicorn metasurface_optimizer.api.app:app --port 8000   # open http://localhost:8000
```

Examples:

| Script | What it shows |
|---|---|
| `examples/01_automatic_synthetic.py` | The whole loop against the synthetic oracle, then artifact export |
| `examples/02_human_in_the_loop.py` | Mode B through files: designs out (parameters + STL), CSV results in, next batch proposed |
| `examples/03_prediction_only.py` | Mode A: rank candidates from existing data, no new simulations |

## Using it with your own design

1. **Describe the parameters** in a YAML schema (bounds, units, steps, fixed flags, manufacturing
   limits, correlated directions). See `configs/schemas/checkerboard_demo.yaml`. Nothing about the
   geometry is hard-coded; templates are `rect_patch`, `checkerboard_rect_patch`, `square_loop`,
   `slotted_patch`, `cross_dipole`.
2. **Describe the objectives** in an experiment YAML (`configs/experiments/demo_checkerboard.yaml`):
   specular RCS reduction, bandwidth, angular and polarization robustness, angular-entropy diffusion,
   S-parameter targets; weights, bands, normalization (min-max, z-score, baseline-relative),
   physics flags, optimizer and stopping rules. Any `high_fidelity.kind` other than
   `synthetic_oracle` (for example `manual`) means results arrive only by upload.
3. **Load `y0`**: Touchstone `.sNp` or CSV S-parameters (`S11_re/S11_im`, `S11_db/S11_phase_deg`,
   optional `_TE`/`_TM` and state suffix such as `_TE_B`), and RCS as long CSV
   (`pol, theta_inc, frequency, rcs_dbsm|rcs_m2[, ref_dbsm|ref_m2]`, plus `theta_s, phi_s` for bistatic).
   `write_sparams_csv` / `write_rcs_csv` in `data/loaders/target_loader.py` produce the exact layout.
4. **Iterate** in the dashboard (Optimization tab) or in Python (`CounterfactualOptimizer`):
   plan initial designs → simulate → upload → finish initial design → propose → simulate → upload →
   complete batch, until a stopping rule (objective plateau, hypervolume plateau, budget, uncertainty) fires.
5. **Export**: parameters (JSON/CSV), per-layer watertight STL with a units/materials sidecar,
   candidate tables, plots, metrics, `report.json/.md`, `optimization_report.pdf`, and a
   reproducibility record in `experiments.sqlite`.

## Dashboard tabs

Dashboard · Dataset · Geometry · EM Response · Baseline Analysis · Optimization · Candidate Designs ·
Existing Paper · Model Comparison · Ablation Study · Results · Export · Experiment Log.
Predicted values carry a green PREDICTED badge; evaluated values show their own fidelity badge.

## Layout

```
metasurface_optimizer/
  core/            provenance labels, seeding, logging
  geometry/        parameter schema, geometry templates, validator, STL export/import, image/mask parsing
  electromagnetics/ S-parameters (Touchstone), RCS (mono/bistatic, dBsm), far-field, response features
  physics/         passivity, reciprocity, smoothness, phase-continuity losses; geometry penalties
  data/            dataset with partitions and leakage guards; loaders; public-dataset adapters
  models/          features, target encoding, physics MLP, MLP, RF, XGBoost, GP, CNN, Transformer, ensemble, transfer
  optimization/    objectives, problem, sensitivity, counterfactuals, acquisition, trust region (PGUCO),
                   NSGA-II, DE, PSO, BO, baselines, stopping rules, Pareto/hypervolume
  evaluation/      metrics, multi-seed statistics, robustness, plots, reports, ablation
  ingestion/       PDF extraction, plot digitizer, design reconstruction (evaluation feature only)
  solvers/         solver interface, budgeted evaluator, synthetic oracle, HFSS/CST stubs
  tracking/        experiment record and SQLite repository
  api/             FastAPI backend
ui/                React + Vite + Tailwind + Plotly dashboard
configs/ docs/ examples/ tests/
```

## Status

**Complete and tested** (54 tests): parameter schema and geometry generation; geometry validation;
STL export (per-layer watertight) and STL inspection; S-parameter/RCS I/O; objectives O1–O9 with
three normalizations; physics losses; seven surrogate families plus the heterogeneous ensemble;
local sensitivity (central difference, or a local linear fit after an LHS start); counterfactual
generation; acquisition functions; trust-region active learning (Mode A and Mode B); NSGA-II, DE,
PSO, BO, random and local-search baselines; multi-seed benchmark and ablation runners; reports;
FastAPI backend; the 13-tab dashboard (driven end to end in a browser); PDF value extraction,
plot digitizing and reconstruction confidence.

**Partial**
- Transfer learning uses a synthetic cell prior. The FSSDataBase adapter now follows the real on-disk
  format (checked against a partial local copy: full `DataBase.db`, 614 of 5,607 sample folders; run on
  20 real samples) and
  loads multi-layer masks with complex S11/S21 (TE/TM). A mask-based prior model that consumes it is
  not built yet.
- Image/STL-to-parameter inference handles rectangle-like features only and always asks for verification.
- PDF extraction is regex-based (no OCR); vector-drawn plots are rendered per page for digitizing.
- Ablations cover A2, A3, A5, A7, A9, A10 and A11 (surrogate swaps); the remaining A-items are
  configuration switches but are not in the default ablation run.

**Not done**
- HFSS/CST automation (adapters are stubs; use uploads).
- Any validation against real full-wave or measured data. All reported numbers are on the synthetic oracle.

## Results on the synthetic demo

See `outputs/demo_checkerboard_synthetic/` (benchmark/comparison_table.csv, ablation/ablation_table.csv,
the report PDF). A first benchmark run that preceded a sensitivity bug fix (identical numbers) is kept in
`outputs/superseded_run1_zero_hf_sensitivity/` with an explanation.
