"""Provenance labels for every number the framework handles.

The framework must never blur the line between an ML prediction, a
high-fidelity electromagnetic simulation, an experimental measurement and a
value read off a published figure. Every response object and every exported
table therefore carries a :class:`Fidelity` label and, where relevant, a
:class:`ValueSource` + confidence for individual extracted values.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Fidelity(str, Enum):
    """Fidelity / origin of an electromagnetic response."""

    PREDICTED = "PREDICTED"
    """Output of a surrogate model. Never equivalent to a simulation."""

    SIMULATED_HF = "VALIDATED_SIMULATION"
    """Full-wave result supplied by the user (HFSS, CST, ...)."""

    MEASURED = "MEASURED"
    """Experimental measurement supplied by the user."""

    REPORTED = "REPORTED"
    """Value reported in a publication (table or text)."""

    PLOT_DIGITIZED = "REPORTED_PLOT_DIGITIZED"
    """Value digitised from a published figure; approximate."""

    SYNTHETIC_ORACLE = "SYNTHETIC_ANALYTIC_ORACLE"
    """Closed-form analytic benchmark model used for demos/benchmarks.

    It plays the role of the expensive oracle in synthetic experiments but is
    NOT a full-wave solver and must never be reported as HFSS/CST output.
    """

    @property
    def is_prediction(self) -> bool:
        return self is Fidelity.PREDICTED

    @property
    def counts_as_high_fidelity(self) -> bool:
        """Whether an evaluation consumes the high-fidelity budget N_HF."""
        return self in (Fidelity.SIMULATED_HF, Fidelity.MEASURED, Fidelity.SYNTHETIC_ORACLE)

    @property
    def display(self) -> str:
        return {
            Fidelity.PREDICTED: "PREDICTED",
            Fidelity.SIMULATED_HF: "VALIDATED (full-wave simulation)",
            Fidelity.MEASURED: "MEASURED",
            Fidelity.REPORTED: "REPORTED",
            Fidelity.PLOT_DIGITIZED: "REPORTED (plot-digitized, approximate)",
            Fidelity.SYNTHETIC_ORACLE: "SYNTHETIC ORACLE (analytic model, not full-wave)",
        }[self]


class ValueSource(str, Enum):
    """Source of an individually extracted scalar value (e.g. from a paper)."""

    TABLE = "table"
    TEXT = "text"
    PLOT_DIGITIZED = "plot_digitized"
    IMAGE_ESTIMATE = "image_estimate"
    STL_INFERRED = "stl_inferred"
    USER = "user"
    DEFAULT = "default"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


_DEFAULT_CONFIDENCE = {
    ValueSource.TABLE: Confidence.HIGH,
    ValueSource.USER: Confidence.HIGH,
    ValueSource.TEXT: Confidence.MEDIUM,
    ValueSource.STL_INFERRED: Confidence.MEDIUM,
    ValueSource.PLOT_DIGITIZED: Confidence.LOW,
    ValueSource.IMAGE_ESTIMATE: Confidence.LOW,
    ValueSource.DEFAULT: Confidence.LOW,
}


@dataclass
class ExtractedValue:
    """A scalar value with its unit, source, confidence and uncertainty."""

    name: str
    value: Any
    unit: Optional[str] = None
    source: ValueSource = ValueSource.USER
    confidence: Optional[Confidence] = None
    uncertainty: Optional[float] = None
    note: str = ""
    verified_by_user: bool = False

    def __post_init__(self) -> None:
        self.source = ValueSource(self.source)
        if self.confidence is None:
            self.confidence = _DEFAULT_CONFIDENCE[self.source]
        else:
            self.confidence = Confidence(self.confidence)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        d["confidence"] = self.confidence.value if self.confidence else None
        return d


@dataclass
class Provenance:
    """Provenance attached to a response or design record."""

    fidelity: Fidelity
    origin: str = ""  # e.g. file name, solver adapter name, model id
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"fidelity": self.fidelity.value, "origin": self.origin, "notes": self.notes, "extra": self.extra}

    @classmethod
    def from_dict(cls, d: dict) -> "Provenance":
        return cls(Fidelity(d["fidelity"]), d.get("origin", ""), d.get("notes", ""), d.get("extra", {}))
