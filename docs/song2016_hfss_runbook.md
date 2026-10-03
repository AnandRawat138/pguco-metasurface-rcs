# Song et al. (2016) ring+disc RCS metasurface: optimization and HFSS validation runbook

Paper: Y.-C. Song, J. Ding, C.-J. Guo, Y.-H. Ren, J.-K. Zhang, "Ultra-broadband backscatter radar cross
section reduction based on polarization-insensitive metasurface", IEEE AWPL 15, 329-331, 2016.

## What the paper gives (and what was read off figures)

| Item | Value | Source |
|---|---|---|
| Cell period | 10 mm | text |
| Substrate | FR4, eps_r 4.4, 3 mm, grounded | text (tan d 0.02 assumed, AEDT `FR4_epoxy`) |
| Ring outer / inner radius | R1 = 1.5 ... 4.5 mm, R2 = 0.92 R1 | text |
| Disc radius | R3 = 0.8 R1 | text |
| Array | 300 x 300 mm = 30 x 30 cells, 4 interleaved sub-arrays at 7.2 / 8 / 9 / 10.8 GHz, 4 x 225 random phases | text, Fig. 1 |
| Sub-array placement | 2 x 2 super-cell [[7.2, 8], [9, 10.8]] | Fig. 1 inset |
| Phase vs R1 at the 4 frequencies | `FIG2_*` in `solvers/ring_disc_array.py` | **read by hand from Fig. 2, about +-10 deg** |
| Claim | > 10 dB backscatter reduction vs equal PEC plate, 7-12 GHz | abstract, Fig. 3, Fig. 7 (measured) |

Not given: the random numbers of the published layout (only the rule), the loss tangent, the copper
thickness. The layout seed here is fixed (`layout_seed: 1`), so "the paper's design" means the paper's
design *rule* with one realisation, not the exact fabricated board.

## Stage 1 - optimization on the approximate model (PC-1, done)

`PYTHONPATH=. python examples/05_song2016_rcs_metasurface.py` runs PGUCO (budget 36 evaluations, seeds
10/11/12) on the Fig.-2-calibrated circuit + physical-optics model. Design variables: the four sub-array
design frequencies f1..f4, the phase span, and the R1 range. Objectives (lower is better): worst-frequency
backscatter reduction over 7-12 GHz at normal incidence (weight 1), band-mean reduction (0.5), specular
reduction at 0/15/30 deg (0.5), -10 dB fractional bandwidth (0.5). Results: `outputs/song2016/`.

**These numbers are not HFSS results.** The model has no inter-cell coupling (the paper attributes its
resonance shifts to coupling), no edge diffraction and no surface waves. Its role is to rank candidate
designs for the HFSS budget, not to claim performance.

## Stage 2 - HFSS on PC-2

You need: AEDT 2024 R2 with HFSS, PyAEDT for 2024 R2 (`pip install "pyaedt>=0.9,<0.13"` or
`ansys-aedt-core` 0.12), and the `hfss_worker/` folder plus the `hfss_jobs/song2016/` batches
(both are in `song2016_hfss_handoff.zip`).

### 2.1 Build the three template projects (once)

```
python hfss_worker/template_builders/song2016_unit_cell.py --out D:/hfss_templates/song2016_unit_cell.aedt
python hfss_worker/template_builders/song2016_array.py --layout hfss_jobs/song2016/layouts/baseline_paper.csv --out D:/hfss_templates/song2016_array.aedt
python hfss_worker/template_builders/song2016_array.py --plate --out D:/hfss_templates/song2016_plate.aedt
```

If you use a different folder than `D:/hfss_templates`, regenerate the batches on PC-1 with
`--template-dir <your folder>` (the path is written into every job). The builders print NOTE lines for
anything they could not set or read back; open the project in the AEDT GUI and fix those by hand. They
were written without access to AEDT, so treat them as a first draft and check in the GUI:

* unit cell: Floquet port mode 1 = TE and mode 2 = TM (the registration reads `S(FloquetPort1:1,FloquetPort1:1)` as TE), de-embed = `air_h`;
* array / plate: plane wave `IncPWave_TE` with theta = `180deg-theta_inc`, phi = 0, E along phi (TE);
  infinite sphere `Backscatter` at theta = `theta_inc`, phi = 180 deg; `RCSTotal` available under Far Fields.

### 2.2 Verify each template (12-step suite, one real solve each)

```
python hfss_worker/worker.py --verify-template hfss_jobs/song2016/templates/song2016_unit_cell.json
python hfss_worker/worker.py --verify-template hfss_jobs/song2016/templates/song2016_plate.json
python hfss_worker/worker.py --verify-template hfss_jobs/song2016/templates/song2016_array.json
```

All applicable rows must PASS; the worker refuses the batches of a template that did not pass. The
array verification is a full array solve (hours). Set `max_runtime_minutes: 1440` in
`hfss_worker/config.yaml` before the array and plate runs.

### 2.3 Run the batches

```
python hfss_worker/worker.py --input hfss_jobs/song2016/unit_cell/batch_001     # 13 cells, R1 = 1.5 ... 4.5 mm
python hfss_worker/worker.py --input hfss_jobs/song2016/plate/batch_001         # PEC plate reference
python hfss_worker/worker.py --input hfss_jobs/song2016/array/batch_001         # paper layout, optimized layout, uniform R1 = 4 mm
```

Each writes `<batch>_results/`. Copy those folders back to PC-1.

## Stage 3 - back on PC-1

```
python -m metasurface_optimizer.simulation.song2016 cell-table --results <unit_cell batch_001_results> --out outputs/song2016/hfss_cell_table.csv
python -m metasurface_optimizer.simulation.song2016 array-rcs  --results <folder holding the array and plate results> --out outputs/song2016/hfss_array_rcs.csv
```

1. `cell-table` writes the HFSS R1 x f reflection table and `*.fig2_check.json`: the phase error of the
   HFSS cell against the paper's Fig. 2 (**baseline reproduction**). Mean errors well above ~10 deg mean the
   HFSS model is not the paper's cell (check port mode order, de-embedding, material); do not continue
   until it reproduces.
2. Set `cell_table: outputs/song2016/hfss_cell_table.csv` under `high_fidelity.conditions` in
   `configs/experiments/song2016_rcs.yaml` and re-run the study: the model now uses HFSS cell data instead
   of the Fig.-2 calibration (the array part is still physical optics).
3. `array-rcs` gives the full-wave backscatter reduction vs the plate for each layout
   (`*.summary.json`: worst / mean dB, fraction of 7-12 GHz below -10 dB). This is the validated result
   for the paper's design and for the optimized design.
