"""Complex S-parameter container, conversions and Touchstone/CSV I/O.

S-parameters are stored as complex arrays ``S_ij(f) = S^R_ij(f) + j S^I_ij(f)``
keyed by ``(i, j, polarization)``. Magnitude/phase/dB views are derived on
demand; models are trained on real and imaginary parts whenever complex data
is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Key = Tuple[int, int, str]  # (i, j, polarization)

_FREQ_SCALE = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}


@dataclass
class SParameterSet:
    """Complex S-parameters on a common frequency grid.

    Attributes:
        frequency: frequencies in Hz, shape (F,).
        data: mapping (i, j, pol) -> complex array of shape (F,).
        z0: reference impedance (ohm).
        port_definition: free-text description (e.g. 'Floquet port, normal incidence').
    """

    frequency: np.ndarray
    data: Dict[Key, np.ndarray] = field(default_factory=dict)
    z0: float = 50.0
    port_definition: str = "Floquet"
    magnitude_only_db: Dict[Key, np.ndarray] = field(default_factory=dict)
    """|S_ij| in dB where no phase is available (e.g. plot-digitised curves)."""

    def __post_init__(self) -> None:
        self.frequency = np.asarray(self.frequency, dtype=float)
        for k, v in list(self.data.items()):
            v = np.asarray(v, dtype=complex)
            if v.shape != self.frequency.shape:
                raise ValueError(f"S{k[0]}{k[1]} ({k[2]}) has shape {v.shape}, expected {self.frequency.shape}")
            self.data[k] = v
        for k, v in list(self.magnitude_only_db.items()):
            self.magnitude_only_db[k] = np.asarray(v, dtype=float)

    # ---------------------------------------------------------- access
    def get(self, i: int, j: int, pol: str = "TE") -> np.ndarray:
        return self.data[(i, j, pol)]

    def has(self, i: int, j: int, pol: str = "TE") -> bool:
        return (i, j, pol) in self.data

    @property
    def polarizations(self) -> List[str]:
        return sorted({k[2] for k in self.data})

    @property
    def n_ports(self) -> int:
        return max((max(k[0], k[1]) for k in self.data), default=0)

    def magnitude(self, i: int, j: int, pol: str = "TE") -> np.ndarray:
        return np.abs(self.get(i, j, pol))

    def db(self, i: int, j: int, pol: str = "TE") -> np.ndarray:
        if (i, j, pol) not in self.data and (i, j, pol) in self.magnitude_only_db:
            return self.magnitude_only_db[(i, j, pol)]
        return 20.0 * np.log10(np.maximum(np.abs(self.get(i, j, pol)), 1e-15))

    def phase(self, i: int, j: int, pol: str = "TE", unwrap: bool = False, degrees: bool = True) -> np.ndarray:
        ph = np.angle(self.get(i, j, pol))
        if unwrap:
            ph = np.unwrap(ph)
        return np.degrees(ph) if degrees else ph

    def real(self, i: int, j: int, pol: str = "TE") -> np.ndarray:
        return self.get(i, j, pol).real

    def imag(self, i: int, j: int, pol: str = "TE") -> np.ndarray:
        return self.get(i, j, pol).imag

    def resample(self, frequency: np.ndarray) -> "SParameterSet":
        """Linear interpolation of real and imaginary parts onto a new grid (no extrapolation)."""
        f = np.asarray(frequency, float)
        if f.min() < self.frequency.min() - 1e-6 * f.max() or f.max() > self.frequency.max() + 1e-6 * f.max():
            raise ValueError("Requested frequency grid lies outside the available data; refusing to extrapolate")
        new = {k: np.interp(f, self.frequency, v.real) + 1j * np.interp(f, self.frequency, v.imag) for k, v in self.data.items()}
        return SParameterSet(f, new, self.z0, self.port_definition)

    def to_table(self) -> Dict[str, np.ndarray]:
        cols: Dict[str, np.ndarray] = {"frequency_hz": self.frequency}
        for (i, j, p), v in sorted(self.data.items()):
            cols[f"S{i}{j}_{p}_re"] = v.real
            cols[f"S{i}{j}_{p}_im"] = v.imag
            cols[f"S{i}{j}_{p}_db"] = 20 * np.log10(np.maximum(np.abs(v), 1e-15))
            cols[f"S{i}{j}_{p}_phase_deg"] = np.degrees(np.angle(v))
        return cols

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "frequency": self.frequency.tolist(),
            "z0": self.z0,
            "port_definition": self.port_definition,
            "data": {f"{i},{j},{p}": {"re": v.real.tolist(), "im": v.imag.tolist()} for (i, j, p), v in self.data.items()},
            "magnitude_only_db": {f"{i},{j},{p}": v.tolist() for (i, j, p), v in self.magnitude_only_db.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SParameterSet":
        data = {}
        for k, v in d["data"].items():
            i, j, p = k.split(",")
            data[(int(i), int(j), p)] = np.asarray(v["re"]) + 1j * np.asarray(v["im"])
        mo = {}
        for k, v in d.get("magnitude_only_db", {}).items():
            i, j, p = k.split(",")
            mo[(int(i), int(j), p)] = np.asarray(v)
        return cls(np.asarray(d["frequency"]), data, d.get("z0", 50.0), d.get("port_definition", "Floquet"), mo)


# ------------------------------------------------------------------ I/O
def read_touchstone(path: str | Path, pol: str = "TE") -> SParameterSet:
    """Read a Touchstone v1 file (.s1p / .s2p) in RI, MA or DB format."""
    path = Path(path)
    m = re.search(r"\.s(\d)p$", path.name.lower())
    if not m:
        raise ValueError("Touchstone file must end in .s<N>p")
    n = int(m.group(1))
    if n > 2:
        raise NotImplementedError("Only 1- and 2-port Touchstone files are supported")
    unit, fmt, z0 = "ghz", "ma", 50.0
    numbers: List[float] = []
    for line in path.read_text().splitlines():
        line = line.split("!")[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            tok = line[1:].lower().split()
            for t_i, t in enumerate(tok):
                if t in _FREQ_SCALE:
                    unit = t
                elif t in ("ri", "ma", "db"):
                    fmt = t
                elif t == "r" and t_i + 1 < len(tok):
                    z0 = float(tok[t_i + 1])
            continue
        numbers.extend(float(t) for t in line.split())
    ncol = 1 + 2 * n * n
    arr = np.asarray(numbers).reshape(-1, ncol)
    f = arr[:, 0] * _FREQ_SCALE[unit]

    def cplx(a, b):
        if fmt == "ri":
            return a + 1j * b
        mag = a if fmt == "ma" else 10 ** (a / 20.0)
        return mag * np.exp(1j * np.radians(b))

    data: Dict[Key, np.ndarray] = {}
    # Touchstone 2-port order: S11 S21 S12 S22
    order = [(1, 1)] if n == 1 else [(1, 1), (2, 1), (1, 2), (2, 2)]
    for k, (i, j) in enumerate(order):
        data[(i, j, pol)] = cplx(arr[:, 1 + 2 * k], arr[:, 2 + 2 * k])
    return SParameterSet(f, data, z0)


def write_touchstone(s: SParameterSet, path: str | Path, pol: str = "TE") -> None:
    """Write a 1- or 2-port Touchstone file in RI format (GHz)."""
    path = Path(path)
    n = 2 if s.has(2, 1, pol) else 1
    order = [(1, 1)] if n == 1 else [(1, 1), (2, 1), (1, 2), (2, 2)]
    lines = ["! written by metasurface_optimizer", f"# GHZ S RI R {s.z0:g}"]
    for fi, f in enumerate(s.frequency):
        row = [f"{f / 1e9:.9g}"]
        for i, j in order:
            v = s.data.get((i, j, pol), np.zeros_like(s.frequency, dtype=complex))[fi]
            row += [f"{v.real:.9g}", f"{v.imag:.9g}"]
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n")


def from_columns(
    frequency: Iterable[float],
    columns: Dict[str, Iterable[float]],
    freq_unit: str = "ghz",
) -> SParameterSet:
    """Build an SParameterSet from table columns.

    Recognised column patterns (case-insensitive), with optional polarization
    suffix ``_TE``/``_TM`` (default TE):
    ``S11_re``/``S11_im``; ``S11_mag``/``S11_phase_deg``; ``S11_db``/``S11_phase_deg``.
    A magnitude/dB column without phase is stored in ``magnitude_only_db``;
    complex values are never invented from magnitude alone.
    """
    f = np.asarray(list(frequency), float) * _FREQ_SCALE[freq_unit.lower()]
    cols = {k.lower(): np.asarray(list(v), float) for k, v in columns.items()}
    pat = re.compile(r"^s(\d)(\d)(?:_((?:te|tm|x|y|rhcp|lhcp)(?:_[a-z0-9]+)?))?_(re|im|mag|db|phase_deg|phase)$")
    groups: Dict[Key, Dict[str, np.ndarray]] = {}
    for name, v in cols.items():
        m = pat.match(name)
        if m:
            key = (int(m.group(1)), int(m.group(2)), (m.group(3) or "te").upper())
            groups.setdefault(key, {})[m.group(4)] = v
    data: Dict[Key, np.ndarray] = {}
    mag_only: Dict[Key, np.ndarray] = {}
    for key, g in groups.items():
        if "re" in g and "im" in g:
            data[key] = g["re"] + 1j * g["im"]
        elif ("mag" in g or "db" in g) and ("phase_deg" in g or "phase" in g):
            mag = g["mag"] if "mag" in g else 10 ** (g["db"] / 20)
            ph = g.get("phase_deg", g.get("phase"))
            data[key] = mag * np.exp(1j * np.radians(ph))
        elif "db" in g:
            mag_only[key] = g["db"]
        elif "mag" in g:
            mag_only[key] = 20 * np.log10(np.maximum(g["mag"], 1e-15))
    return SParameterSet(f, data, magnitude_only_db=mag_only)
