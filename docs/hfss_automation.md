# Two-PC HFSS / PyAEDT high-fidelity simulation automation

**Status (honest):** implemented and tested end to end with the MOCK solver and with a *fake* PyAEDT
(tests/fakes, a closed-form resonator that only checks control flow and file formats). It has **not yet run
against real HFSS**. Real operation is verified only when `tests/hfss` passes on the HFSS PC with a
verified template (section 8).

## 1. Architecture

```
PC-1 (ML controller, no HFSS)                          PC-2 (AEDT 2024 R2 + HFSS + PyAEDT)
 CounterfactualOptimizer                                 hfss_worker/worker.py
   generate_candidates(k) ──► SimulationManager            scan / claim job
                               validate candidate          validate package (hash, signature,
                               HFSSAdapter: job package      trusted runner, template verified)
                               transport ───────────────►  run_hfss.py in AEDT's Python
                                                             open verified template, set variables,
                                                             12 checks, solve, extract
                               collect / import ◄─────────  result package (+ every failed attempt)
   update(results)  ◄───────── parse, validate, feedback
```

* The optimizer never imports PyAEDT. Everything solver-specific is behind `SolverAdapter`
  (`create_job`, `submit`, `get_status`, `collect_result`, `validate_result`);
  `HFSSAdapter` implements it, `CSTAdapter` is a placeholder.
* `hfss_worker/` is standard library only. Copy that one folder to PC-2.
* Code: `metasurface_optimizer/simulation/` (PC-1), `hfss_worker/` (PC-2), `tests/hfss/` (validation suite).

## 2. Job package

`<job_id>/` contains:

| file | content |
|---|---|
| `job.json` | job_id, framework_version, created_at, solver, aedt_version, status, candidate_id, parent_design_id, kind, simulation_type, template id + hash, protocol version, **SHA-256 of every file** |
| `job.sig` | HMAC-SHA256 of job.json (only when a shared key is configured) |
| `parameters.json` | `[{name, variable, value, unit, schema_value, schema_unit}]`, explicit units (`10 mm`) |
| `simulation_config.json` | solver-independent: frequency (start/stop/points), excitation (type, polarizations, θ, φ), outputs, checks |
| `template.json` | the template registration the job must use |
| `run_hfss.py` | the self-contained PyAEDT runner (identical for every job) |
| `geometry/geometry.json` (+ `geometry.stl` if requested) | description of the generated cell |
| `metadata.json` | reproducibility record |
| `prediction.json` | the surrogate's PREDICTED values (never returned as a result) |
| `README.txt` | how to run it |

Parameters are what HFSS uses for optimization. STL is exported for the final design and for
arbitrary geometry; the verified template, not the STL, defines what HFSS simulates.

## 3. Templates

Two kinds:

* **A. `periodic_fss`:** unit cell, master/slave boundaries, Floquet ports, TE/TM modes, scan angle via variables.
* **B. `finite_rcs`:** finite array, plane-wave excitation, radiation/PML boundary, monostatic RCS per polarization/angle.

A registration (UI: High-Fidelity Simulation → Register a template; examples in `configs/hfss/`) gives:
template_id, project path **on PC-2**, design, solution type, setup, sweep, `parameter_mapping`
(schema parameter → AEDT variable + unit), excitation names, angle variables, polarization mode
(`floquet_modes`, `separate_excitations` or `variable`), boundaries, materials, and the extraction
expressions (`S11_TE: "S(FloquetPort1:1,FloquetPort1:1)"`, RCS report category / context / expressions).
Every free schema parameter must be mapped. Unmapped fixed parameters keep the template's own value.

**Verify template** packages a `verify_template` job for the baseline design x0. Run it on PC-2 and import
the result. The template becomes VERIFIED only if the run completed and all 12 checks passed, and only for
that file hash. If the `.aedt` changes, the template is no longer verified and must be checked again.

## 4. What run_hfss.py does (15 steps)

1. Read job.json, parameters.json, simulation_config.json and template.json.
2. Check the template file exists and its SHA-256 matches the registration.
3. Copy it to a scratch folder next to the result (the original is never modified).
4. Import PyAEDT (`ansys.aedt.core`, else `pyaedt`) and start AEDT non-graphically on the copy.
5. Check the AEDT version against the job's.
6. Check the design name and solution type.
7. Check every mapped variable exists, set it with its unit, read it back and compare.
8. Run AEDT design validation (`validate_simple`).
9. Check the registered materials.
10. Check excitations, the requested excitation type, the angle and polarization controls, and the outputs.
11. Check the registered boundaries.
12. Check setup and sweep, apply the frequency sweep, read it back and compare.
13. For each incidence angle (and polarization when it is a variable): set it, solve.
14. Extract S-parameters (re/im/mag/phase), monostatic RCS (m², dBsm) and far-field cuts. Check that values are finite, sit on the requested grid and are passive (|S11|²+|S21|² ≤ 1 + tol).
15. Write the result package and release AEDT, which is always released, even after a failure.

Any failed check stops the job as **FAILED_VALIDATION**, with the reason in `validation.json`.
A solver error gives **FAILED**. Neither produces numbers.

## 5. Version compatibility

All PyAEDT calls go through `hfss_worker/runner/version_compatibility.py` (re-exported as
`metasurface_optimizer/simulation/hfss/version_compatibility.py`). It handles:

* both package names;
* both `Hfss()` keyword generations (`project/design/version/new_desktop` vs
  `projectname/designname/specified_version/new_desktop_session`);
* `SolutionData` accessors (`get_expression_data` vs `data_real/data_imag`);
* variable, excitation, boundary, setup and sweep accessors.

Probed against PyAEDT 0.12.0 (the release line for AEDT 2024 R2). Which path was taken is logged in
`solver_metadata.json → compat_log`, together with the AEDT, PyAEDT, Python and OS versions.

## 6. Result package and job states

`result.json`, `sparameters.csv` (theta_inc, phi_inc, frequency [Hz], S.._re/_im/_mag/_phase_deg),
`rcs.csv` (pol, theta_inc, phi_inc, frequency, rcs_m2, rcs_dbsm[, theta_s, phi_s, ref_m2]),
`farfield.csv`, `solver_metadata.json`, `validation.json` (12 rows), `hfss.log`, `plots/`.

States: PENDING → CLAIMED → RUNNING → SOLVING ⇄ EXTRACTING → COMPLETED, or FAILED / FAILED_VALIDATION /
TIMEOUT / CANCELLED. `job_status.json` keeps the whole history.

* Each attempt writes `result_attempt_<n>/`, and nothing is overwritten.
* FAILED and TIMEOUT are retried up to `max_retries` (default 2). FAILED_VALIDATION is not retried.
* `max_runtime_minutes` kills the runner and AEDT's process tree and records TIMEOUT.
* A failed simulation is never a result: the optimizer logs it in `failed_evaluations` and excludes it.

## 7. Transfer modes

**Manual (no network).**
1. PC-1: High-Fidelity Simulation → Next batch → Download `batch_NNN.zip`.
2. PC-2: unzip, then `python hfss_worker/worker.py --input batch_NNN`. This writes `batch_NNN_results/`.
3. PC-1: Import results (zip of that folder).

On import, the framework checks the schema, file hashes and candidate ids, stores each result under
`candidates/<id>/validated/<job_id>/`, updates the dataset and the surrogate, and runs the next optimizer step.

**Shared folder.** Both PCs see `simulation_queue/{pending,running,completed,failed,archive}`.
* PC-1 copies a job into `.incoming/` and renames it into `pending/`.
* The worker (`--queue <folder>`) claims a job with an atomic rename into `running/`, runs it, and moves it
  to `completed/` or `failed/`. Several workers can share a queue safely.
* Workers write heartbeats to `workers/`. Cancel moves a pending job to `archive/`, or asks a running one to stop (`cancel.request`).

**Network** (REST: POST /jobs, GET /jobs/{id}, cancel, result, health) is designed for but not implemented.

## 8. Validation suite (PC-2)

```
set HFSS_TEMPLATE_JSON=D:\hfss_templates\my_template.json   # registration + baseline_parameters + verification_config
python -m pytest tests/hfss -v                                # test_01_connection ... test_12_cleanup
```

or simply `python hfss_worker/worker.py --verify-template my_template.json`. Both run one real solve
of the baseline and report the 12 rows:

1. connection
2. version
3. open_template
4. variables
5. geometry
6. materials
7. excitations
8. boundaries
9. setup_sweep
10. solve
11. extraction
12. cleanup

Rows 05, 06 and 08 may be SKIP when the feature is unavailable or nothing is registered; every other row must PASS.
The worker refuses candidate jobs for a template until this passed on that PC (`require_validation_suite`).

## 9. Optimizer integration, budget, modes

```python
candidates = optimizer.generate_candidates(k)       # baseline -> initial design -> PGUCO proposals
jobs = manager.create_jobs(candidates)              # pre-HFSS checks: bounds, geometry, manufacturing, material, frequency, config
manager.submit(jobs)
results = manager.collect_results()                 # calls optimizer.update(results) when the batch is complete
```

* **Budget:** Max HFSS sims, Batch size, Max iterations. Each submitted job counts once; worker retries of the same job do not.
  The manager never submits beyond the budget and tracks used/remaining (`simulation_budget_report.json`).
* **Modes:**
  * DRY_RUN writes packages only.
  * MOCK is the deterministic analytic oracle, labeled `MOCK_SOLVER`: a software test, never physics, never mixed with other fidelities.
  * HFSS is the real solver.

## 10. Predicted vs simulated

Each candidate has two folders:
* `predicted/` holds the PREDICTED objectives with σ and the predicted spectra.
* `validated/<job_id>/` holds the simulated result.

Nothing is overwritten. For every simulated candidate that had a prediction:

e(x) = |J_HFSS(x) − J_ML(x)|, per objective, plus:
* relative error;
* z-score;
* composite error;
* |S| (dB) and RCS-reduction RMSE between predicted and simulated spectra;
* peak-frequency error.

These errors are out of sample. The UI and `optimization_history.png` show J_best vs N_HFSS and prediction error vs N_HFSS.

## 11. Baseline reproduction

Before optimizing with a real template, x0 is simulated and compared with the known response (your own
earlier simulation, a measurement, or the paper's data). The comparison reports:
* response RMSE (dB);
* resonance shift;
* peak difference.

Above the thresholds (3 dB, 3 %, 5 dB by default) automatic optimization is blocked. An override needs a
reason, which is recorded. Fidelity labels show whether the check is physical: it is only meaningful when
the reference is measured or reported and the reproduction is HFSS.

## 12. Security

* The worker runs only `run_hfss.py` files byte-identical to its own trusted runner. Job data is JSON, so a job cannot carry code.
* Every file in a package is hashed in `job.json`. With `job_key_file` set, `job.json` must carry a valid HMAC signature.
* Packages may contain only allow-listed files.
* Ids and paths are sanitized (no `..`, absolute paths or drive letters), and zip uploads are extracted safely.
* Variable names must be identifiers, values finite numbers, and units from an allow-list.
* `allowed_template_roots` restricts where template projects may live.
* The future network mode would need authentication.

## 13. Final outputs and known limits

**Export** writes:
* `optimized_parameters.json`, `optimized_geometry.stl`
* `predicted_response.csv`, `hfss_validated_response.csv` (named `MOCK_response_NOT_PHYSICS_*` in MOCK mode)
* `baseline_vs_optimized.png`, `optimization_history.png`, `pareto_front.png`, `sensitivity_analysis.png`
* `simulation_budget_report.json`, `experiment_report.pdf`

Known limits:
* No real-HFSS run yet.
* RCS extraction is monostatic only. Bistatic RCS, which the angular-entropy (diffusion) objective needs, is not extracted from HFSS. The MOCK solver provides it for pipeline tests only.
* A PEC-plate reference for RCS reduction can be computed analytically (4πA²cos²θ/λ²) when registered; it is labeled as a formula, not a simulation.
* Far-field extraction and the RCS report categories depend on your project's setup and are confirmed only by step 11 of the verification.
* CST and the network mode are not implemented.
