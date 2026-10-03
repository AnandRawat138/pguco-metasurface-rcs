"""Unified electromagnetic response record with provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from metasurface_optimizer.core.provenance import Fidelity, Provenance
from metasurface_optimizer.electromagnetics.farfield import FarField
from metasurface_optimizer.electromagnetics.rcs import RCSData
from metasurface_optimizer.electromagnetics.sparameter import SParameterSet


@dataclass
class EMResponse:
    """Electromagnetic response of one design under stated conditions.

    Any subset of S-parameters, RCS and far-field may be present. ``provenance``
    states whether the numbers are PREDICTED, VALIDATED (full-wave),
    MEASURED, REPORTED, ... — downstream code never drops this label.
    """

    provenance: Provenance
    sparams: Optional[SParameterSet] = None
    rcs: Optional[RCSData] = None
    farfield: Optional[FarField] = None
    conditions: Dict[str, Any] = field(default_factory=dict)

    @property
    def fidelity(self) -> Fidelity:
        return self.provenance.fidelity

    @property
    def label(self) -> str:
        return self.provenance.fidelity.display

    def to_dict(self) -> dict:
        return {
            "provenance": self.provenance.to_dict(),
            "sparams": self.sparams.to_dict() if self.sparams else None,
            "rcs": self.rcs.to_dict() if self.rcs else None,
            "conditions": self.conditions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EMResponse":
        return cls(
            Provenance.from_dict(d["provenance"]),
            SParameterSet.from_dict(d["sparams"]) if d.get("sparams") else None,
            RCSData.from_dict(d["rcs"]) if d.get("rcs") else None,
            None,
            d.get("conditions", {}),
        )
