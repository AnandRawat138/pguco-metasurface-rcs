"""HFSS job batches and result import for the Song et al. (2016) ring+disc metasurface.

Two templates, built once on the HFSS PC by ``hfss_worker/template_builders/song2016_*.py`` and verified
with ``python hfss_worker/worker.py --verify-template <registration>.json``:

* ``song2016_unit_cell`` (periodic_fss): one 10 mm cell, lattice pairs, one Floquet port (2 modes)
  de-embedded to the substrate top, PEC ground. Variables p, h, R1, k2, k3. Output S11_TE / S11_TM.
  Batch: R1 = 1.5 ... 4.5 mm -> reflection table (``collect_cell_table``) that replaces the
  Fig.-2-calibrated model (``RingDiscCell.from_table``), and a Fig. 2 reproduction check.
* ``song2016_array`` (finite_rcs): 30 x 30 cells, one design variable per cell (r1_RR_CC), plane wave
  (TE: E along y) from theta_inc in the xz plane, radiation boundary, backscatter RCS in an infinite
  sphere that follows theta_inc. ``song2016_plate`` is the equal-size PEC plate reference.
  Batch: the paper's layout, the optimized layout(s), uniform R1 = 4 mm, and the plate.

    python -m metasurface_optimizer.simulation.song2016 make-batches --layouts outputs/song2016/layouts \
        --out hfss_jobs/song2016 --template-dir D:/hfss_templates
    python -m metasurface_optimizer.simulation.song2016 cell-table --results <batch_results> --out cell_table.csv
    python -m metasurface_optimizer.simulation.song2016 array-rcs --results <batch_results> --out array_rcs.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from metasurface_optimizer.geometry.parameter_schema import ParameterSchema
from metasurface_optimizer.simulation.hfss_adapter import HFSSAdapter, make_simulation_config
from metasurface_optimizer.simulation.solver_adapter import SimulationJob
from metasurface_optimizer.simulation.templates import TemplateSpec
from metasurface_optimizer.simulation.transports import ManualBatchTransport

N_CELLS = 30
UNIT_R1_SWEEP = [round(1.5 + 0.25 * i, 2) for i in range(13)]


def cell_var(i: int, j: int) -> str:
    return f"r1_{i:02d}_{j:02d}"


# ------------------------------------------------------------------ schemas
def unit_cell_schema() -> ParameterSchema:
    return ParameterSchema.from_dict({
        "name": "song2016_unit_cell", "length_unit": "mm",
        "parameters": [
            {"name": "r1", "type": "continuous", "min": 1.5, "max": 4.5, "unit": "mm", "default": 3.0, "role": "length"},
            {"name": "k2", "type": "continuous", "min": 0.8, "max": 0.98, "unit": "1", "default": 0.92, "role": "other"},
            {"name": "k3", "type": "continuous", "min": 0.5, "max": 0.9, "unit": "1", "default": 0.8, "role": "other"},
            {"name": "substrate_thickness", "type": "continuous", "min": 3.0, "max": 3.0, "unit": "mm", "default": 3.0, "fixed": True, "role": "thickness"},
            {"name": "period", "type": "continuous", "min": 10.0, "max": 10.0, "unit": "mm", "default": 10.0, "fixed": True, "role": "period"}],
        "metadata": {"description": "Song et al. 2016 unit cell (HFSS Template A)"}})


def array_schema(n: int = N_CELLS) -> ParameterSchema:
    return ParameterSchema.from_dict({
        "name": f"song2016_array_{n}x{n}", "length_unit": "mm",
        "parameters": [{"name": cell_var(i, j), "type": "continuous", "min": 1.5, "max": 4.5, "unit": "mm", "default": 3.0, "role": "length"}
                       for i in range(n) for j in range(n)],
        "metadata": {"description": "per-cell outer ring radius R1 of the 30 x 30 layout (row 0 = +y edge, col 0 = -x edge)"}})


# ------------------------------------------------------------------ templates
def unit_cell_template(template_dir: str) -> TemplateSpec:
    return TemplateSpec.from_dict({
        "template_id": "song2016_unit_cell", "name": "Song 2016 ring+disc unit cell, Floquet port (built by song2016_unit_cell.py)",
        "simulation_type": "periodic_fss", "aedt_project": f"{template_dir.rstrip('/')}/song2016_unit_cell.aedt",
        "design": "UnitCell", "solution_type": "Modal", "setup": "Setup1", "sweep": "Sweep1",
        "parameter_mapping": {"r1": {"variable": "R1", "unit": "mm"}, "k2": {"variable": "k2", "unit": ""}, "k3": {"variable": "k3", "unit": ""}},
        "excitation": {"names": ["FloquetPort1"], "polarization": {"mode": "floquet_modes"}},
        "boundaries": ["LatticePair1", "LatticePair2", "Ground"], "materials": ["FR4_epoxy"],
        "outputs": {"sparameters": {"S11_TE": "S(FloquetPort1:1,FloquetPort1:1)", "S11_TM": "S(FloquetPort1:2,FloquetPort1:2)"}},
        "notes": "Mode 1 = TE, mode 2 = TM is what the builder requests; confirm in the port's mode table (verification step 07)."})


def array_template(template_dir: str, n: int = N_CELLS, plate: bool = False) -> TemplateSpec:
    name = "song2016_plate" if plate else "song2016_array"
    mapping = {"plate_size": {"variable": "plate_L", "unit": "mm"}} if plate else         {cell_var(i, j): {"variable": cell_var(i, j), "unit": "mm"} for i in range(n) for j in range(n)}
    return TemplateSpec.from_dict({
        "template_id": name, "name": ("PEC plate 300 x 300 mm reference" if plate else f"Song 2016 {n}x{n} array") + " (built by song2016_array.py)",
        "simulation_type": "finite_rcs", "aedt_project": f"{template_dir.rstrip('/')}/{name}.aedt",
        "design": "Plate" if plate else "Array", "solution_type": "Modal", "setup": "Setup1", "sweep": "Sweep1",
        "parameter_mapping": mapping,
        "excitation": {"names": ["IncPWave_TE"], "angle_variables": {"theta": "theta_inc"}, "polarization": {"mode": "floquet_modes"}},
        "boundaries": ["Rad1"], "materials": [] if plate else ["FR4_epoxy"],
        "outputs": {"rcs": {"report_category": "Far Fields", "context": "Backscatter", "quantity": "m2", "expressions": {"TE": "RCSTotal"}}},
        "rcs_reference": {"kind": "pec_plate", "area_m2": (n * 0.01) ** 2},
        "notes": "Single TE plane wave (E along y); backscatter = infinite sphere 'Backscatter' at (theta_inc, 180 deg)."})


def _registration(t: TemplateSpec, baseline: List[Dict[str, Any]], sc: Dict[str, Any]) -> Dict[str, Any]:
    return {**t.worker_json(), "name": t.name, "notes": t.notes, "baseline_parameters": baseline, "verification_config": sc}


# ------------------------------------------------------------------ batches
def read_layout(path: Path, n: int = N_CELLS) -> np.ndarray:
    R1 = np.full((n, n), np.nan)
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            R1[int(row["row"]), int(row["col"])] = float(row["r1_mm"])
    if np.isnan(R1).any():
        raise ValueError(f"{path}: layout is not complete {n} x {n}")
    return R1


def make_batches(layout_dir: Path, out: Path, template_dir: str, f_cell=(6e9, 13e9, 71), f_array=(7e9, 12e9, 11),
                 theta_array: Sequence[float] = (0.0,)) -> Dict[str, Any]:
    out = Path(out)
    info: Dict[str, Any] = {"template_dir_on_hfss_pc": template_dir, "registrations": {}, "batches": {}}
    regdir = out / "templates"
    regdir.mkdir(parents=True, exist_ok=True)
    # ---- A: unit cell
    us = unit_cell_schema()
    ut = unit_cell_template(template_dir)
    usc = make_simulation_config(ut, *f_cell, polarizations=("TE", "TM"), theta_deg=(0.0,))
    ua = HFSSAdapter(ut, us, usc, out / "work" / "unit_cell", ManualBatchTransport(out / "unit_cell"), "HFSS", study="s16cell")
    x0 = us.default_vector()
    (regdir / "song2016_unit_cell.json").write_text(json.dumps(_registration(ut, ua.parameters_for(x0), usc), indent=1))
    jobs = []
    for r in UNIT_R1_SWEEP:
        x = x0.copy()
        x[us.index("r1")] = r
        cid = f"r1_{r:.2f}".replace(".", "p")
        d = ua.generate_job_package({"x": x}, "candidate", 0, cid, parent_design_id="paper_cell")
        jobs.append(SimulationJob(d.name, cid, "candidate", 0, str(d), "HFSS"))
    info["batches"]["unit_cell"] = ua.transport.submit(jobs)
    # ---- B: array layouts + plate
    s = array_schema()
    at = array_template(template_dir)
    asc = make_simulation_config(at, *f_array, polarizations=("TE",), theta_deg=theta_array, outputs=["rcs"])
    aa = HFSSAdapter(at, s, asc, out / "work" / "array", ManualBatchTransport(out / "array"), "HFSS", study="s16arr")
    layouts = sorted(Path(layout_dir).glob("*.csv"))
    base = next((p for p in layouts if p.stem == "baseline_paper"), layouts[0])
    (regdir / "song2016_array.json").write_text(json.dumps(_registration(at, aa.parameters_for(read_layout(base).ravel()), asc), indent=1))
    jobs = []
    for p in layouts:
        R1 = read_layout(p)
        d = aa.generate_job_package({"x": R1.ravel()}, "candidate", 0, p.stem, parent_design_id="baseline_paper",
                                    extra_metadata={"layout_file": p.name})
        jobs.append(SimulationJob(d.name, p.stem, "candidate", 0, str(d), "HFSS"))
    info["batches"]["array"] = aa.transport.submit(jobs)
    pt = array_template(template_dir, plate=True)
    psc = make_simulation_config(pt, *f_array, polarizations=("TE",), theta_deg=theta_array, outputs=["rcs"])
    ps = ParameterSchema.from_dict({"name": "song2016_plate", "parameters": [
        {"name": "plate_size", "type": "continuous", "min": 300.0, "max": 300.0, "unit": "mm", "default": 300.0, "fixed": True}]})
    pa = HFSSAdapter(pt, ps, psc, out / "work" / "plate", ManualBatchTransport(out / "plate"), "HFSS", study="s16plate")
    (regdir / "song2016_plate.json").write_text(json.dumps(_registration(pt, pa.parameters_for(ps.default_vector()), psc), indent=1))
    d = pa.generate_job_package({"x": ps.default_vector()}, "candidate", 0, "pec_plate", parent_design_id="reference")
    info["batches"]["plate"] = pa.transport.submit([SimulationJob(d.name, "pec_plate", "candidate", 0, str(d), "HFSS")])
    for k in ("unit_cell", "array", "plate"):
        info["registrations"][k] = str(regdir / f"song2016_{k}.json")
    (out / "batches.json").write_text(json.dumps(info, indent=1))
    return info


# ------------------------------------------------------------------ results
def _result_dirs(results: Path) -> List[Path]:
    return sorted({p.parent for p in Path(results).rglob("result.json")})


def collect_cell_table(results: Path, out_csv: Path) -> Dict[str, Any]:
    """Template-A results -> r1_mm,f_ghz,re,im (TE, normal incidence) + Fig. 2 reproduction errors."""
    from metasurface_optimizer.solvers.ring_disc_array import FIG2_F_GHZ, FIG2_PHASE_DEG, FIG2_R1_MM

    rows, by_r1 = [], {}
    for rd in _result_dirs(results):
        res = json.loads((rd / "result.json").read_text())
        if res.get("status") != "COMPLETED" or res.get("simulation_type") != "HFSS":
            continue
        cid = str(res.get("candidate_id", ""))
        if not cid.startswith("r1_"):
            continue
        r1 = float(cid[3:].replace("p", "."))  # candidate ids are r1_<mm> with '.' -> 'p' (make_batches)
        with (rd / "sparameters.csv").open() as f:
            for r in csv.DictReader(f):
                if abs(float(r["theta_inc"])) > 1e-6:
                    continue
                g = float(r["S11_TE_re"]) + 1j * float(r["S11_TE_im"])
                rows.append((r1, float(r["frequency"]) / 1e9, g.real, g.imag))
                by_r1.setdefault(r1, {})[float(r["frequency"]) / 1e9] = g
    if not rows:
        raise ValueError(f"no completed HFSS unit-cell results under {results}")
    rows.sort()
    with Path(out_csv).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r1_mm", "f_ghz", "re", "im"])
        w.writerows(rows)
    errs = []
    for r1, curve in by_r1.items():
        fs = np.array(sorted(curve))
        g = np.array([curve[v] for v in fs])
        for k, fg in enumerate(FIG2_F_GHZ):
            if fs[0] <= fg <= fs[-1] and FIG2_R1_MM[0] <= r1 <= FIG2_R1_MM[-1]:
                gi = np.interp(fg, fs, g.real) + 1j * np.interp(fg, fs, g.imag)
                ref = np.interp(r1, FIG2_R1_MM, FIG2_PHASE_DEG[k])
                errs.append(abs(np.degrees(np.angle(gi * np.exp(-1j * np.radians(ref))))))
    rep = {"n_r1": len(by_r1), "fig2_phase_error_deg": {"mean": float(np.mean(errs)), "max": float(np.max(errs)), "n": len(errs)} if errs else None,
           "note": "Fig. 2 values were read by hand (+-10 deg); errors well above that mean the HFSS cell does not reproduce the paper's cell."}
    Path(out_csv).with_suffix(".fig2_check.json").write_text(json.dumps(rep, indent=1))
    return rep


def collect_array_rcs(results: Path, out_csv: Path) -> Dict[str, Any]:
    """Template-B results (layouts + PEC plate) -> backscatter RCS and reduction vs the plate per layout."""
    data: Dict[str, Dict[float, Dict[float, float]]] = {}
    for rd in _result_dirs(results):
        res = json.loads((rd / "result.json").read_text())
        if res.get("status") != "COMPLETED" or res.get("simulation_type") != "HFSS" or not (rd / "rcs.csv").is_file():
            continue
        with (rd / "rcs.csv").open() as f:
            for r in csv.DictReader(f):
                data.setdefault(res["candidate_id"], {}).setdefault(float(r["theta_inc"]), {})[float(r["frequency"])] = float(r["rcs_m2"])
    if "pec_plate" not in data:
        raise ValueError("the PEC plate result is needed as the reference")
    plate = data.pop("pec_plate")
    rows, summ = [], {}
    for cid, by_t in sorted(data.items()):
        for th, curve in sorted(by_t.items()):
            red = []
            for fhz, v in sorted(curve.items()):
                ref = plate.get(th, {}).get(fhz)
                if ref:
                    red.append(10 * np.log10(v / ref))
                    rows.append([cid, th, fhz / 1e9, v, ref, red[-1]])
            if red:
                summ[f"{cid}@{th:g}deg"] = {"worst_dB": float(np.max(red)), "mean_dB": float(np.mean(red)), "fraction_below_-10dB": float(np.mean(np.array(red) <= -10))}
    with Path(out_csv).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout", "theta_inc_deg", "f_GHz", "rcs_m2", "plate_rcs_m2", "reduction_dB"])
        w.writerows(rows)
    Path(out_csv).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-batches")
    m.add_argument("--layouts", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--template-dir", default="D:/hfss_templates", help="folder of the .aedt templates ON THE HFSS PC")
    m.add_argument("--theta", type=float, nargs="+", default=[0.0])
    c = sub.add_parser("cell-table")
    c.add_argument("--results", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("array-rcs")
    r.add_argument("--results", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make-batches":
        print(json.dumps(make_batches(Path(a.layouts), Path(a.out), a.template_dir, theta_array=a.theta), indent=1))
    elif a.cmd == "cell-table":
        print(json.dumps(collect_cell_table(Path(a.results), Path(a.out)), indent=1))
    else:
        print(json.dumps(collect_array_rcs(Path(a.results), Path(a.out)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
