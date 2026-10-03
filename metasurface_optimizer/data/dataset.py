"""Design/response records, data partitions and leakage protection.

Partitions:

* ``TRAINING``             - may be used to fit surrogates.
* ``VALIDATION``           - held out for model selection / prediction metrics.
* ``EXTERNAL_EVALUATION``  - an external case (e.g. a design recovered from a
  paper). Only its *baseline* design and baseline response may be shown to the
  optimiser; its reported optimised design is stored as ``REFERENCE_ONLY`` and
  can never enter training.
* ``PRIOR``                - public/prior data used only for pre-training.

:meth:`DesignDataset.training_arrays` refuses to return records from the
evaluation partitions, and :func:`assert_no_leakage` checks that no design in
a training set coincides with a held-out design.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.electromagnetics.response import EMResponse


class Partition(str, Enum):
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    EXTERNAL_EVALUATION = "EXTERNAL_EVALUATION"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    PRIOR = "PRIOR"


class DataLeakageError(RuntimeError):
    pass


def design_hash(x: np.ndarray, decimals: int = 6) -> str:
    return hashlib.sha1(np.round(np.asarray(x, float), decimals).tobytes()).hexdigest()[:16]


@dataclass
class DesignRecord:
    x: np.ndarray
    response: Optional[EMResponse]
    J: Optional[np.ndarray] = None
    partition: Partition = Partition.TRAINING
    source: str = ""
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    iteration: int = 0
    tag: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def fidelity(self) -> Optional[Fidelity]:
        return self.response.fidelity if self.response else None

    @property
    def hash(self) -> str:
        return design_hash(self.x)

    def to_dict(self, include_response: bool = False) -> dict:
        d = {
            "record_id": self.record_id, "x": np.asarray(self.x).tolist(),
            "J": None if self.J is None else np.asarray(self.J).tolist(), "partition": self.partition.value,
            "source": self.source, "iteration": self.iteration, "tag": self.tag,
            "fidelity": self.fidelity.value if self.fidelity else None, "meta": self.meta,
        }
        if include_response and self.response is not None:
            d["response"] = self.response.to_dict()
        return d


class DesignDataset:
    """In-memory collection of :class:`DesignRecord` with partition guards."""

    TRAINABLE = (Partition.TRAINING,)

    def __init__(self, records: Optional[Iterable[DesignRecord]] = None):
        self.records: List[DesignRecord] = list(records or [])

    def add(self, rec: DesignRecord) -> DesignRecord:
        if rec.response is not None and rec.response.fidelity.is_prediction and rec.partition == Partition.TRAINING:
            raise DataLeakageError("PREDICTED responses can never be used as training data")
        if rec.partition == Partition.TRAINING:
            held = {r.hash for r in self.records if r.partition in (Partition.EXTERNAL_EVALUATION, Partition.REFERENCE_ONLY, Partition.VALIDATION)}
            if rec.hash in held and not rec.meta.get("is_external_baseline"):
                raise DataLeakageError(f"Design {rec.hash} is held out and cannot be added to TRAINING")
        self.records.append(rec)
        return rec

    def __len__(self) -> int:
        return len(self.records)

    def by_partition(self, *parts: Partition) -> List[DesignRecord]:
        return [r for r in self.records if r.partition in parts]

    def training_arrays(self, require_J: bool = True):
        """X, J for trainable records only."""
        recs = [r for r in self.records if r.partition in self.TRAINABLE and (r.J is not None or not require_J)]
        if not recs:
            return np.empty((0, 0)), np.empty((0, 0)), []
        X = np.vstack([r.x for r in recs])
        J = np.vstack([r.J for r in recs]) if require_J else None
        return X, J, recs

    def high_fidelity_count(self) -> int:
        return sum(1 for r in self.records if r.fidelity is not None and r.fidelity.counts_as_high_fidelity and r.meta.get("counted", True))

    def save(self, path: str | Path, include_responses: bool = True) -> None:
        Path(path).write_text(json.dumps([r.to_dict(include_responses) for r in self.records], default=float))

    @classmethod
    def load(cls, path: str | Path) -> "DesignDataset":
        out = cls()
        for d in json.loads(Path(path).read_text()):
            resp = EMResponse.from_dict(d["response"]) if d.get("response") else None
            out.records.append(DesignRecord(np.asarray(d["x"]), resp, None if d["J"] is None else np.asarray(d["J"]),
                                            Partition(d["partition"]), d["source"], d["record_id"], d["iteration"], d["tag"], d.get("meta", {})))
        return out


def assert_no_leakage(train: Iterable[np.ndarray], held_out: Iterable[np.ndarray]) -> None:
    th = {design_hash(x) for x in train}
    clash = [design_hash(x) for x in held_out if design_hash(x) in th]
    if clash:
        raise DataLeakageError(f"{len(clash)} held-out design(s) found in the training set")
