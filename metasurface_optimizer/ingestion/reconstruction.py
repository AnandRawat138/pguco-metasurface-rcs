"""Assemble a baseline design from extracted / user-entered values with explicit confidence.

The user maps extracted symbols (e.g. ``dim:P``) to schema parameters. Every
parameter in the result states its source and confidence; parameters with no
mapped value fall back to the schema default and are flagged, never invented.
Overall confidence = the lowest confidence among geometric parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np

from metasurface_optimizer.core.provenance import Confidence, ExtractedValue, ValueSource
from metasurface_optimizer.geometry.geometry_validator import GeometryValidator
from metasurface_optimizer.geometry.parameter_schema import ParameterSchema

_ORDER = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}


@dataclass
class ReconstructedDesign:
    values: Dict[str, float]
    provenance: Dict[str, Dict]
    confidence: Confidence
    validation: Dict
    unresolved: List[str] = field(default_factory=list)
    requires_user_verification: bool = True

    def to_dict(self) -> Dict:
        return {"values": self.values, "provenance": self.provenance, "reconstruction_confidence": self.confidence.value,
                "validation": self.validation, "unresolved_parameters": self.unresolved,
                "requires_user_verification": self.requires_user_verification}


def reconstruct(schema: ParameterSchema, extracted: List[ExtractedValue], mapping: Mapping[str, str],
                user_values: Optional[Mapping[str, float]] = None) -> ReconstructedDesign:
    """mapping: schema parameter -> extracted value name (e.g. {'period': 'dim:P'})."""
    by_name: Dict[str, List[ExtractedValue]] = {}
    for v in extracted:
        by_name.setdefault(v.name, []).append(v)
    values, prov, unresolved = {}, {}, []
    for p in schema.parameters:
        if user_values and p.name in user_values:
            values[p.name] = float(user_values[p.name])
            prov[p.name] = {"source": ValueSource.USER.value, "confidence": Confidence.HIGH.value}
            continue
        key = mapping.get(p.name)
        cands = by_name.get(key, []) if key else []
        if cands:
            best = max(cands, key=lambda c: _ORDER[c.confidence])
            distinct = {round(float(c.value), 6) for c in cands if isinstance(c.value, (int, float))}
            values[p.name] = float(best.value)
            conf = best.confidence if len(distinct) <= 1 else Confidence.LOW
            prov[p.name] = {"source": best.source.value, "confidence": conf.value, "note": best.note,
                            "conflicting_values": sorted(distinct) if len(distinct) > 1 else None}
        else:
            values[p.name] = p.default
            prov[p.name] = {"source": ValueSource.DEFAULT.value, "confidence": Confidence.LOW.value, "note": "schema default; not found in the source"}
            unresolved.append(p.name)
    geo_roles = {"period", "length", "width", "gap", "thickness", "rotation"}
    confs = [Confidence(prov[p.name]["confidence"]) for p in schema.parameters if p.role in geo_roles]
    overall = min(confs, key=lambda c: _ORDER[c]) if confs else Confidence.LOW
    x = schema.to_vector(values)
    val = GeometryValidator(schema).validate(x).to_dict() if schema.template else {}
    return ReconstructedDesign(values, prov, overall, val, unresolved)
