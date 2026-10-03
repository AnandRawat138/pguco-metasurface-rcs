"""Adapter interface for public / prior datasets.

An adapter converts a dataset's native layout into one of two prior formats:

* :class:`~metasurface_optimizer.models.transfer.CellSample` for
  parameterised cells (used by the cell prior model), or
* :class:`MaskSample` for pixel/mask-described cells (used by mask-based
  representation learning).

Adapters must state which fields are compatible with the target problem;
incompatible fields are dropped, never guessed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MaskSample:
    masks: np.ndarray  # (n, H, W) or (n, layers, H, W) binary
    frequency: np.ndarray  # (F,) Hz
    responses: np.ndarray  # (n, F) or (n, F, C)
    response_kind: str  # e.g. "S21_mag_db", "S11_complex"
    meta: List[Dict] = field(default_factory=list)


class PublicDatasetAdapter(ABC):
    name: str = "abstract"
    version: str = "unknown"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def load(self, **kw): ...

    def compatibility_report(self) -> Dict[str, object]:
        return {"name": self.name, "version": self.version}
