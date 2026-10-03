"""Surrogate target vector y = [ J-block | response-block ].

* **J-block** (``m`` entries): the raw objective values J_1..J_m. This block
  drives acquisition and optimisation.
* **Response block**: complex S-parameters as (Re, Im) on a resampled
  frequency grid, plus optional specular RCS-reduction spectra (dB). This
  block is used for (i) physics-informed training losses (passivity,
  reciprocity, smoothness act on it), (ii) predicted-response plots, and
  (iii) multi-task regularisation of the shared network trunk.

Objectives are *not* recomputed from the predicted response block, because
several of them (e.g. angular entropy) need the full bistatic pattern that is
not regressed. The two blocks are predicted jointly by every surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from metasurface_optimizer.core.provenance import Fidelity, Provenance
from metasurface_optimizer.electromagnetics.rcs import to_dbsm
from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.electromagnetics.sparameter import SParameterSet


@dataclass
class TargetEncoder:
    objective_names: List[str]
    frequency: np.ndarray  # response-block grid (Hz)
    sparam_keys: List[Tuple[int, int, str]] = field(default_factory=list)
    rcs_keys: List[Tuple[str, float]] = field(default_factory=list)  # (pol, theta_inc) of specular reduction spectra
    slices: Dict[str, slice] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequency = np.asarray(self.frequency, float)
        k = len(self.objective_names)
        self.slices = {"J": slice(0, k)}
        F = len(self.frequency)
        for key in self.sparam_keys:
            self.slices[f"S{key[0]}{key[1]}_{key[2]}_re"] = slice(k, k + F); k += F
            self.slices[f"S{key[0]}{key[1]}_{key[2]}_im"] = slice(k, k + F); k += F
        for pol, th in self.rcs_keys:
            self.slices[f"RCSred_{pol}_{th:g}"] = slice(k, k + F); k += F
        self.dim = k

    @property
    def m(self) -> int:
        return len(self.objective_names)

    @property
    def response_dim(self) -> int:
        return self.dim - self.m

    def complex_pairs(self) -> List[Tuple[slice, slice, Tuple[int, int, str]]]:
        return [(self.slices[f"S{a}{b}_{p}_re"], self.slices[f"S{a}{b}_{p}_im"], (a, b, p)) for a, b, p in self.sparam_keys]

    def spectral_slices(self) -> List[slice]:
        return [s for name, s in self.slices.items() if name != "J"]

    @classmethod
    def from_response(cls, objective_names: Sequence[str], resp: EMResponse, n_freq: int = 21,
                      band: Optional[Tuple[float, float]] = None, include_rcs: bool = True) -> "TargetEncoder":
        """Infer the response block layout from an example response."""
        f_src = resp.sparams.frequency if resp.sparams is not None else resp.rcs.frequency
        lo, hi = (band if band else (f_src.min(), f_src.max()))
        lo, hi = max(lo, f_src.min()), min(hi, f_src.max())
        f = np.linspace(lo, hi, n_freq)
        skeys = sorted(resp.sparams.data) if resp.sparams is not None else []
        rkeys = []
        if include_rcs and resp.rcs is not None and resp.rcs.reference is not None:
            rkeys = [(p, float(t)) for p in resp.rcs.polarizations for t in resp.rcs.theta_inc]
        return cls(list(objective_names), f, skeys, rkeys)

    def encode(self, resp: Optional[EMResponse], J: np.ndarray) -> np.ndarray:
        y = np.full(self.dim, np.nan)
        y[self.slices["J"]] = J
        if resp is None:
            return y
        if self.sparam_keys and resp.sparams is not None:
            s = resp.sparams.resample(self.frequency)
            for key in self.sparam_keys:
                if key in s.data:
                    y[self.slices[f"S{key[0]}{key[1]}_{key[2]}_re"]] = s.data[key].real
                    y[self.slices[f"S{key[0]}{key[1]}_{key[2]}_im"]] = s.data[key].imag
        if self.rcs_keys and resp.rcs is not None and resp.rcs.reference is not None:
            red = to_dbsm(resp.rcs.specular()) - to_dbsm(resp.rcs.reference)
            for pol, th in self.rcs_keys:
                pi, ii = resp.rcs.pol_index(pol), resp.rcs.inc_index(th)
                y[self.slices[f"RCSred_{pol}_{th:g}"]] = np.interp(self.frequency, resp.rcs.frequency, red[pi, ii])
        return y

    def decode(self, y: np.ndarray, std: Optional[np.ndarray] = None, origin: str = "surrogate") -> Dict[str, object]:
        """Decode one predicted vector into a PREDICTED response + spectra dictionary."""
        data = {}
        for sre, sim, key in self.complex_pairs():
            data[key] = y[sre] + 1j * y[sim]
        sp = SParameterSet(self.frequency, data) if data else None
        rcs_spec = {name: y[s] for name, s in self.slices.items() if name.startswith("RCSred_")}
        rcs_std = {name: std[s] for name, s in self.slices.items() if name.startswith("RCSred_")} if std is not None else {}
        return {
            "response": EMResponse(Provenance(Fidelity.PREDICTED, origin, "surrogate prediction; not a simulation"), sp),
            "J": y[self.slices["J"]],
            "J_std": None if std is None else std[self.slices["J"]],
            "rcs_reduction_db": rcs_spec,
            "rcs_reduction_db_std": rcs_std,
            "frequency": self.frequency,
        }

    def to_dict(self) -> dict:
        return {"objective_names": self.objective_names, "frequency": self.frequency.tolist(),
                "sparam_keys": [list(k) for k in self.sparam_keys], "rcs_keys": [list(k) for k in self.rcs_keys]}
