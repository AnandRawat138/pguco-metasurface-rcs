"""User-editable parameter schema for metasurface designs.

A design is the vector ``x = [x_1, ..., x_d]`` whose entries are described by
:class:`ParameterSpec`. The schema is loaded from YAML/JSON so that any unit
cell family (patch, slot, loop, cross, multilayer, ...) can be described
without code changes. Optimisers always operate in the normalised unit cube
``u = (x - x_min) / (x_max - x_min)`` and map back through :meth:`ParameterSchema.denormalize`.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import yaml

_UNIT_TO_MM = {"mm": 1.0, "um": 1e-3, "µm": 1e-3, "cm": 10.0, "m": 1000.0, "mil": 0.0254, "in": 25.4}


@dataclass
class ParameterSpec:
    """Specification of one design variable.

    Attributes:
        name: unique identifier.
        type: ``continuous`` | ``integer`` | ``categorical``.
        min, max: bounds (continuous / integer).
        unit: physical unit string (``mm``, ``degree``, ``1`` ...).
        default: nominal value.
        choices: allowed values for categorical parameters.
        step: default counterfactual perturbation ``delta_i``; if ``None``
            it is 5 % of the range.
        resolution: manufacturing resolution; optimised values are snapped
            to multiples of it (``None`` = no snapping).
        fixed: excluded from optimisation (e.g. a substrate the user cannot change).
        role: semantic tag (``period``, ``length``, ``width``, ``gap``,
            ``rotation``, ``thickness``, ``permittivity``, ``loss_tangent``,
            ``metal_thickness``, ``offset``, ``other``). Used by the
            physics-aware feature extractor and geometry templates.
        description: free text.
    """

    name: str
    type: str = "continuous"
    min: Optional[float] = None
    max: Optional[float] = None
    unit: str = "1"
    default: Optional[Any] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    resolution: Optional[float] = None
    fixed: bool = False
    role: str = "other"
    description: str = ""

    def __post_init__(self) -> None:
        if self.type not in ("continuous", "integer", "categorical"):
            raise ValueError(f"Parameter {self.name}: unknown type {self.type!r}")
        if self.type == "categorical":
            if not self.choices:
                raise ValueError(f"Categorical parameter {self.name} needs 'choices'")
            self.min, self.max = 0.0, float(len(self.choices) - 1)
            if self.default is None:
                self.default = self.choices[0]
        else:
            if self.min is None or self.max is None:
                raise ValueError(f"Parameter {self.name} needs min and max")
            self.min, self.max = float(self.min), float(self.max)
            if not self.max > self.min and not self.fixed:
                raise ValueError(f"Parameter {self.name}: max must exceed min")
            if self.default is None:
                self.default = 0.5 * (self.min + self.max)
        if self.step is None:
            self.step = 0.05 * (self.max - self.min) if self.max > self.min else 0.0

    @property
    def span(self) -> float:
        return float(self.max - self.min)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LinearConstraint:
    """Generic linear inequality ``sum_i a_i x_i <= b`` declared in the schema."""

    name: str
    coeffs: Dict[str, float]
    rhs: float
    description: str = ""

    def value(self, x: Mapping[str, float]) -> float:
        """Return g(x) = a.x - b (feasible iff <= 0)."""
        return float(sum(c * float(x[k]) for k, c in self.coeffs.items()) - self.rhs)


@dataclass
class CorrelatedDirection:
    """A physically meaningful joint perturbation direction (e.g. scale patch and period together)."""

    name: str
    direction: Dict[str, float]
    description: str = ""


@dataclass
class ParameterSchema:
    """Ordered collection of :class:`ParameterSpec` plus schema-level constraints."""

    parameters: List[ParameterSpec]
    name: str = "unnamed"
    version: str = "1.0"
    template: Optional[str] = None
    length_unit: str = "mm"
    linear_constraints: List[LinearConstraint] = field(default_factory=list)
    correlated_directions: List[CorrelatedDirection] = field(default_factory=list)
    manufacturing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate parameter names in schema")
        self._index = {n: i for i, n in enumerate(names)}

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ParameterSchema":
        params = [ParameterSpec(**p) for p in d["parameters"]]
        lc = [LinearConstraint(**c) for c in d.get("linear_constraints", [])]
        cd = [CorrelatedDirection(**c) for c in d.get("correlated_directions", [])]
        return cls(
            parameters=params,
            name=d.get("name", "unnamed"),
            version=str(d.get("version", "1.0")),
            template=d.get("template"),
            length_unit=d.get("length_unit", "mm"),
            linear_constraints=lc,
            correlated_directions=cd,
            manufacturing=dict(d.get("manufacturing", {})),
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ParameterSchema":
        path = Path(path)
        text = path.read_text()
        d = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "length_unit": self.length_unit,
            "parameters": [p.to_dict() for p in self.parameters],
            "linear_constraints": [asdict(c) for c in self.linear_constraints],
            "correlated_directions": [asdict(c) for c in self.correlated_directions],
            "manufacturing": self.manufacturing,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        d = self.to_dict()
        path.write_text(json.dumps(d, indent=2) if path.suffix == ".json" else yaml.safe_dump(d, sort_keys=False))

    def fingerprint(self) -> str:
        """Stable hash of the schema used for experiment versioning."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    # ------------------------------------------------------------- access
    @property
    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    @property
    def dim(self) -> int:
        return len(self.parameters)

    @property
    def free_mask(self) -> np.ndarray:
        return np.array([not p.fixed for p in self.parameters])

    @property
    def free_names(self) -> List[str]:
        return [p.name for p in self.parameters if not p.fixed]

    def __getitem__(self, name: str) -> ParameterSpec:
        return self.parameters[self._index[name]]

    def index(self, name: str) -> int:
        return self._index[name]

    def by_role(self, role: str) -> List[ParameterSpec]:
        return [p for p in self.parameters if p.role == role]

    @property
    def lower(self) -> np.ndarray:
        return np.array([p.min for p in self.parameters], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.array([p.max for p in self.parameters], dtype=float)

    @property
    def steps(self) -> np.ndarray:
        return np.array([p.step for p in self.parameters], dtype=float)

    def default_vector(self) -> np.ndarray:
        return self.to_vector({p.name: p.default for p in self.parameters})

    def with_overrides(self, **bounds: Dict[str, Any]) -> "ParameterSchema":
        """Return a copy with per-parameter attribute overrides, e.g. ``slot_length={'max': 7}``."""
        new = copy.deepcopy(self)
        for name, attrs in bounds.items():
            spec = new[name]
            for k, v in attrs.items():
                setattr(spec, k, v)
            spec.__post_init__()
        return new

    # -------------------------------------------------------- conversions
    def to_vector(self, values: Mapping[str, Any]) -> np.ndarray:
        """Dict of physical values -> numeric vector (categoricals become indices)."""
        out = np.empty(self.dim)
        for i, p in enumerate(self.parameters):
            if p.name not in values:
                raise KeyError(f"Missing value for parameter {p.name!r}")
            v = values[p.name]
            out[i] = p.choices.index(v) if p.type == "categorical" else float(v)
        return out

    def to_dict_values(self, x: Sequence[float]) -> Dict[str, Any]:
        """Numeric vector -> dict of physical values."""
        x = np.asarray(x, dtype=float)
        d: Dict[str, Any] = {}
        for i, p in enumerate(self.parameters):
            if p.type == "categorical":
                d[p.name] = p.choices[int(round(np.clip(x[i], 0, len(p.choices) - 1)))]
            elif p.type == "integer":
                d[p.name] = int(round(x[i]))
            else:
                d[p.name] = float(x[i])
        return d

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        span = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)
        return (x - self.lower) / span

    def denormalize(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return self.lower + u * (self.upper - self.lower)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def snap(self, x: np.ndarray) -> np.ndarray:
        """Round to manufacturing resolution / integer / categorical grid, then clip."""
        x = np.array(x, dtype=float, copy=True)
        for i, p in enumerate(self.parameters):
            xi = x[..., i]
            if p.type in ("integer", "categorical"):
                xi = np.round(xi)
            elif p.resolution:
                xi = p.min + np.round((xi - p.min) / p.resolution) * p.resolution
            x[..., i] = xi
        return self.clip(x)

    def in_bounds(self, x: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.all((x >= self.lower - tol) & (x <= self.upper + tol), axis=-1)

    def linear_constraint_values(self, x: Sequence[float]) -> Dict[str, float]:
        vals = self.to_dict_values(x)
        return {c.name: c.value(vals) for c in self.linear_constraints}

    def length_scale_to_mm(self) -> float:
        return _UNIT_TO_MM.get(self.length_unit, 1.0)

    def sample_uniform(self, n: int, rng: np.random.Generator, fix_to: Optional[np.ndarray] = None) -> np.ndarray:
        """Uniform samples in bounds; fixed parameters taken from ``fix_to`` (or default)."""
        u = rng.random((n, self.dim))
        x = self.denormalize(u)
        ref = self.default_vector() if fix_to is None else np.asarray(fix_to, float)
        x[:, ~self.free_mask] = ref[~self.free_mask]
        return self.snap(x)

    def describe(self, x: Sequence[float]) -> List[Dict[str, Any]]:
        vals = self.to_dict_values(x)
        return [{"name": p.name, "value": vals[p.name], "unit": p.unit, "min": p.min, "max": p.max, "fixed": p.fixed} for p in self.parameters]


def iter_free_indices(schema: ParameterSchema) -> Iterable[int]:
    return (i for i, p in enumerate(schema.parameters) if not p.fixed)
