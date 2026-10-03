"""Loaders for the user's existing design and its electromagnetic response.

Accepted inputs:

* design parameters: JSON / YAML (``{name: value}``) or CSV (``name,value`` rows);
* S-parameters: Touchstone ``.s1p``/``.s2p`` or CSV (see
  :func:`~metasurface_optimizer.electromagnetics.sparameter.from_columns`);
* RCS: long-format CSV ``frequency,theta_inc,pol,rcs_dbsm[,ref_dbsm]`` (monostatic)
  or ``frequency,theta_inc,pol,theta_s,phi_s,rcs_dbsm`` (bistatic);
* a JSON response previously written by this framework.

The caller states the fidelity (VALIDATED simulation, MEASURED, REPORTED, ...);
nothing loaded here is ever labeled PREDICTED.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from metasurface_optimizer.core.provenance import Fidelity, Provenance
from metasurface_optimizer.electromagnetics.rcs import RCSData, from_dbsm, monostatic_from_table
from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.electromagnetics.sparameter import SParameterSet, from_columns, read_touchstone


def load_parameters(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in (".json",):
        return json.loads(path.read_text())
    if path.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(path.read_text())
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        if "name" in cols and "value" in cols:
            return {r[df.columns[cols.index("name")]]: r[df.columns[cols.index("value")]] for _, r in df.iterrows()}
        return df.iloc[0].to_dict()
    raise ValueError(f"Unsupported parameter file {path}")


def load_sparams(path: str | Path, freq_unit: str = "ghz", pol: str = "TE") -> SParameterSet:
    path = Path(path)
    if path.suffix.lower().startswith(".s") and path.suffix.lower().endswith("p"):
        return read_touchstone(path, pol)
    df = pd.read_csv(path)
    fcol = next(c for c in df.columns if c.lower().startswith("freq"))
    return from_columns(df[fcol].to_numpy(), {c: df[c].to_numpy() for c in df.columns if c != fcol}, freq_unit)


def load_rcs(path: str | Path, freq_unit: str = "ghz") -> RCSData:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "theta_s" not in df.columns:
        return monostatic_from_table(df.to_dict("records"), freq_unit)
    scale = {"hz": 1, "mhz": 1e6, "ghz": 1e9}[freq_unit.lower()]
    freqs = np.sort(df["frequency"].unique()); incs = np.sort(df["theta_inc"].unique())
    pols = sorted(df["pol"].str.upper().unique()); ts = np.sort(df["theta_s"].unique()); ps = np.sort(df["phi_s"].unique())
    arr = np.full((len(pols), len(incs), len(freqs), len(ts), len(ps)), np.nan)
    val = df["rcs_m2"].to_numpy() if "rcs_m2" in df.columns else from_dbsm(df["rcs_dbsm"].to_numpy())
    idx = (np.searchsorted(np.asarray(pols), df["pol"].str.upper().to_numpy()), np.searchsorted(incs, df["theta_inc"]), np.searchsorted(freqs, df["frequency"]),
           np.searchsorted(ts, df["theta_s"]), np.searchsorted(ps, df["phi_s"]))
    arr[idx] = val
    if np.isnan(arr).any():
        raise ValueError("Bistatic RCS table is not a complete grid")
    ref = None
    if "ref_m2" in df.columns or "ref_dbsm" in df.columns:
        # Specular RCS of the same-size PEC reference plate, repeated on every row of a (pol, theta_inc, frequency) block.
        ref = np.full((len(pols), len(incs), len(freqs)), np.nan)
        rv = df["ref_m2"].to_numpy() if "ref_m2" in df.columns else from_dbsm(df["ref_dbsm"].to_numpy())
        ref[idx[:3]] = rv
        if np.isnan(ref).any():
            raise ValueError("Reference-plate column must be filled for every (pol, theta_inc, frequency)")
    return RCSData(freqs * scale, incs, pols, theta_s=ts, phi_s=ps, bistatic=arr, reference=ref)


def load_response(fidelity: Fidelity, sparams: Optional[str | Path] = None, rcs: Optional[str | Path] = None,
                  response_json: Optional[str | Path] = None, origin: str = "", freq_unit: str = "ghz") -> EMResponse:
    if fidelity is Fidelity.PREDICTED:
        raise ValueError("Loaded data cannot be labeled PREDICTED")
    if response_json:
        r = EMResponse.from_dict(json.loads(Path(response_json).read_text()))
        r.provenance = Provenance(fidelity, origin or str(response_json))
        return r
    return EMResponse(Provenance(fidelity, origin or str(sparams or rcs)),
                      load_sparams(sparams, freq_unit) if sparams else None,
                      load_rcs(rcs, freq_unit) if rcs else None)


def write_sparams_csv(sp: SParameterSet, path: str | Path) -> None:
    """Write S-parameters in the CSV layout ``load_sparams`` reads (frequency in Hz; use freq_unit='hz' to load)."""
    pd.DataFrame(sp.to_table()).to_csv(path, index=False)


def write_rcs_csv(rcs: RCSData, path: str | Path) -> None:
    """Write RCS in the long CSV layout ``load_rcs`` reads (frequency in Hz).

    Bistatic data -> columns pol, theta_inc, frequency, theta_s, phi_s, rcs_m2 (+ ref_m2);
    monostatic data -> pol, theta_inc, frequency, rcs_m2 (+ ref_m2)."""
    rows = []
    for p, pol in enumerate(rcs.polarizations):
        for i, inc in enumerate(rcs.theta_inc):
            for k, f in enumerate(rcs.frequency):
                ref = {} if rcs.reference is None else {"ref_m2": float(rcs.reference[p, i, k])}
                if rcs.bistatic is not None:
                    for a, ts in enumerate(rcs.theta_s):
                        for b, ps in enumerate(rcs.phi_s):
                            rows.append({"pol": pol, "theta_inc": float(inc), "frequency": float(f), "theta_s": float(ts), "phi_s": float(ps),
                                         "rcs_m2": float(rcs.bistatic[p, i, k, a, b]), **ref})
                else:
                    rows.append({"pol": pol, "theta_inc": float(inc), "frequency": float(f), "rcs_m2": float(rcs.monostatic[p, i, k]), **ref})
    pd.DataFrame(rows).to_csv(path, index=False)
