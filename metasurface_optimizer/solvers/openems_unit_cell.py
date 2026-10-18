"""Full-wave (openEMS FDTD) reflection of the Song et al. (2016) ring+disc unit cell at normal incidence.

Periodic-cell equivalence: for normal incidence with E along y on a cell that is mirror-symmetric in x
and y, the infinite periodic array is exactly represented by one cell bounded by PMC walls normal to x
and PEC walls normal to y (the TEM "parallel-plate" unit cell). The same symmetry is used once more
inside the cell: only the quarter x, y in [0, P/2] is simulated, with PMC at x = 0 and PEC at y = 0.

Model (mm): PEC ground at z = 0 (boundary), FR4 slab 0 < z < h (eps_r, tan d at f_c through a conductivity),
zero-thickness PEC ring (outer R1, inner k2 R1, drawn as a keyhole polygon) and disc (k3 R1) at z = h,
air above, TEM waveguide port (E = y, H = x, kc = 0) travelling in -z, PML above the port.
Gamma is de-embedded from the port measurement plane to the substrate surface z = h:
Gamma(h) = S11 exp(+j 2 k0 (z_meas - h)). Check: with no metal and h -> 0 this gives -1 (``reference_run``).

Uniform transverse mesh (default 0.05 mm) so that the circular edges are staircased equally at every
angle; the ring width is 0.08 R1 (0.12 mm at R1 = 1.5 mm), i.e. at least 2.4 cells.
"""
from __future__ import annotations

import csv
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

OPENEMS_DIR = os.environ.get("OPENEMS_DIR", r"C:\Users\anand\Downloads\openEMS_x64_v0.37.0-rc1_msvc\openEMS")
C0 = 299_792_458.0
EPS0 = 8.8541878128e-12


def _import():
    if os.name == "nt" and os.path.isdir(OPENEMS_DIR):
        os.add_dll_directory(OPENEMS_DIR)
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    return ContinuousStructure, openEMS


def _circle(r, n):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return r * np.cos(t), r * np.sin(t)


def simulate_cell(r1: float, k2: float = 0.92, k3: float = 0.8, h: float = 3.0, eps_r: float = 4.4, tan_d: float = 0.02,
                  period: float = 10.0, f_start: float = 6e9, f_stop: float = 13e9, n_freq: int = 71, res: float = 0.05,
                  air: float = 14.0, metal: bool = True, substrate: bool = True, workdir: Optional[str] = None, threads: int = 0,
                  end_criteria: float = 1e-5, max_steps: int = 600000, keep: bool = False, calibrate: bool = True) -> Dict[str, np.ndarray]:
    """Return {'f': Hz, 'gamma': complex Gamma at z = h, 'runtime_s', 'cells', 'calibration'}.

    ``calibrate``: divide by the port error measured on the same mesh without substrate and metal, where the exact
    answer is Gamma(h) = -exp(-j 2 k0 h) (PEC ground at distance h below the reference plane). This removes the
    ~4 % magnitude offset of the TEM port's staggered voltage/current integration (phase error < 0.3 deg)."""
    ContinuousStructure, openEMS = _import()
    unit = 1e-3
    a = period / 2.0
    f0, fc = 0.5 * (f_start + f_stop), 0.5 * (f_stop - f_start) + 0.5e9
    FDTD = openEMS(NrTS=int(max_steps), EndCriteria=end_criteria)
    FDTD.SetGaussExcite(f0, fc)
    # [xmin xmax ymin ymax zmin zmax]: PMC at x=0 (symmetry) and x=P/2 (periodic wall), PEC at y=0 and y=P/2, PEC ground, PML top
    FDTD.SetBoundaryCond(["PMC", "PMC", "PEC", "PEC", "PEC", "PML_8"])
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)
    nxy = int(round(a / res))
    mesh.AddLine("x", np.linspace(0, a, nxy + 1))
    mesh.AddLine("y", np.linspace(0, a, nxy + 1))
    lam_min = C0 / f_stop / unit
    z_top = h + air
    zs = list(np.linspace(0, h, max(4, int(np.ceil(h / 0.25))) + 1))
    z_meas, z_exc = h + air - 4.0, h + air - 3.0
    zs += [z_meas, z_exc, z_top]
    mesh.AddLine("z", zs)
    mesh.SmoothMeshLines("z", lam_min / 25, ratio=1.3)
    if h > 0 and substrate:
        sub = CSX.AddMaterial("FR4", epsilon=eps_r, kappa=2 * np.pi * f0 * EPS0 * eps_r * tan_d)
        sub.AddBox([0, 0, 0], [a, a, h], priority=1)
    if metal:
        pec = CSX.AddMetal("metal")
        n = 720
        xo, yo = _circle(r1, n)
        xi, yi = _circle(k2 * r1, n)
        # keyhole polygon: outer circle CCW, cut to the inner circle, inner circle CW, back along the cut
        px = np.concatenate([xo, [xo[0]], xi[::-1], [xi[0]], [xo[0]]])
        py = np.concatenate([yo, [yo[0]], yi[::-1], [yi[0]], [yo[0]]])
        pec.AddPolygon([px, py], "z", h, priority=10)
        xd, yd = _circle(k3 * r1, n)
        pec.AddPolygon([xd, yd], "z", h, priority=10)
    port = FDTD.AddWaveGuidePort(0, [0, 0, z_exc], [a, a, z_meas], "z", E_func=["0", "1", "0"], H_func=["-1", "0", "0"], kc=0, excite=1)
    sim = workdir or tempfile.mkdtemp(prefix="oems_cell_")
    t0 = time.time()
    kw = {"verbose": 0, "cleanup": True}
    if threads:
        kw["numThreads"] = threads
    cwd = os.getcwd()
    try:
        FDTD.Run(sim, **kw)  # changes the working directory to ``sim``
    finally:
        os.chdir(cwd)
    rt = time.time() - t0
    f = np.linspace(f_start, f_stop, n_freq)
    port.CalcPort(sim, f)
    s11 = port.uf_ref / port.uf_inc
    k0 = 2 * np.pi * f / C0
    g = s11 * np.exp(1j * 2 * k0 * (z_meas - h) * unit)
    cells = (nxy + 1) ** 2 * len(mesh.GetLines("z"))
    if not keep and workdir is None:
        shutil.rmtree(sim, ignore_errors=True)
    out = {"f": f, "gamma": g, "s11_port": s11, "runtime_s": rt, "cells": cells, "z_meas_mm": z_meas, "calibration": None}
    if calibrate and (metal or substrate):
        cal = port_calibration(h=h, period=period, f_start=f_start, f_stop=f_stop, n_freq=n_freq, res=res, air=air, threads=threads)
        out["gamma"] = g / cal
        out["calibration"] = cal
    return out


_CAL: Dict[tuple, np.ndarray] = {}


def port_calibration(h, period, f_start, f_stop, n_freq, res, air, threads=0) -> np.ndarray:
    """Measured / exact Gamma of the empty cell (PEC ground at z = 0, reference plane at z = h); cached per setup."""
    key = (h, period, f_start, f_stop, n_freq, res, air)
    if key not in _CAL:
        r = simulate_cell(1.0, h=h, period=period, f_start=f_start, f_stop=f_stop, n_freq=n_freq, res=res, air=air,
                          metal=False, substrate=False, threads=threads, calibrate=False)
        exact = -np.exp(-1j * 2 * (2 * np.pi * r["f"] / C0) * h * 1e-3)
        _CAL[key] = r["gamma"] / exact
    return _CAL[key]


def reference_run(**kw) -> Dict[str, np.ndarray]:
    """No metal, no substrate, h = 0: the PEC ground must give Gamma = -1 after de-embedding (checks port and phase reference)."""
    return simulate_cell(1.0, metal=False, h=0.0, calibrate=False, **kw)


def sweep(r1_values: Sequence[float], out_csv: str | Path, **kw) -> Path:
    """Simulate every R1 and write the r1_mm,f_ghz,re,im table read by ``RingDiscCell.from_table`` (appends; skips done R1)."""
    out = Path(out_csv).resolve()  # FDTD.Run changes the working directory
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {round(float(r["r1_mm"]), 4) for r in csv.DictReader(fh)}
    new = not out.exists()
    for r1 in r1_values:
        if round(r1, 4) in done:
            continue
        res = simulate_cell(r1, **kw)
        with out.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["r1_mm", "f_ghz", "re", "im"])
                new = False
            for fi, g in zip(res["f"], res["gamma"]):
                w.writerow([round(r1, 4), round(fi / 1e9, 6), float(g.real), float(g.imag)])
        print(f"R1 = {r1:.3f} mm: {res['runtime_s']:.0f} s, {res['cells']} cells, |G| {np.abs(res['gamma']).min():.3f}-{np.abs(res['gamma']).max():.3f}", flush=True)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="openEMS unit-cell sweep for the Song et al. (2016) ring+disc cell")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r1", type=float, nargs="+", default=list(np.round(np.arange(1.5, 4.5001, 0.125), 3)))
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    sweep(a.r1, a.out, res=a.res, threads=a.threads)
