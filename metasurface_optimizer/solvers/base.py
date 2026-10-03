"""Solver-adapter interface separating ML predictions from high-fidelity evaluations.

Every high-fidelity source (full-wave solver automation, measurement upload,
synthetic benchmark oracle) implements :class:`HighFidelitySolver`. The
framework only ever obtains non-PREDICTED responses through this interface
and counts each call against the simulation budget ``N_HF``.

No class in this package imitates HFSS or CST. HFSS/CST automation is left
as explicit adapters that raise until they are implemented against a real
installation.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.electromagnetics.response import EMResponse


class BudgetExceeded(RuntimeError):
    pass


class PendingExternalEvaluation(RuntimeError):
    """Raised in human-in-the-loop mode: results must be supplied by the user."""

    def __init__(self, request_file: Path, n: int):
        super().__init__(f"{n} design(s) written to {request_file}; upload their full-wave results to continue")
        self.request_file = request_file
        self.n = n


class HighFidelitySolver(ABC):
    """Abstract high-fidelity evaluator."""

    name: str = "abstract"
    fidelity: Fidelity = Fidelity.SIMULATED_HF

    @abstractmethod
    def evaluate(self, values: Mapping[str, Any], conditions: Optional[Mapping[str, Any]] = None) -> EMResponse:
        """Return the response of the design described by ``values``."""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "fidelity": self.fidelity.value}


class ManualUploadSolver(HighFidelitySolver):
    """Mode B (human-in-the-loop): writes evaluation requests; the user runs HFSS/CST and uploads results."""

    name = "manual_upload"

    def __init__(self, request_dir: str | Path, fidelity: Fidelity = Fidelity.SIMULATED_HF):
        self.request_dir = Path(request_dir)
        self.fidelity = fidelity

    def request(self, designs: List[Mapping[str, Any]], tag: str) -> Path:
        self.request_dir.mkdir(parents=True, exist_ok=True)
        path = self.request_dir / f"hf_request_{tag}.json"
        path.write_text(json.dumps({"status": "PENDING_HIGH_FIDELITY_EVALUATION", "designs": list(designs)}, indent=2, default=float))
        return path

    def evaluate(self, values, conditions=None) -> EMResponse:
        path = self.request([dict(values)], "single")
        raise PendingExternalEvaluation(path, 1)


class HFSSAdapter(HighFidelitySolver):  # pragma: no cover - requires a licensed installation
    """Placeholder for Ansys HFSS automation (e.g. via PyAEDT). Not implemented."""

    name = "hfss"

    def evaluate(self, values, conditions=None) -> EMResponse:
        raise NotImplementedError(
            "HFSS automation is not implemented. Use ManualUploadSolver (export candidates, simulate in HFSS, "
            "upload Touchstone/CSV results) or implement this adapter against your PyAEDT installation."
        )


class CSTAdapter(HighFidelitySolver):  # pragma: no cover
    """Placeholder for CST Studio Suite automation. Not implemented."""

    name = "cst"

    def evaluate(self, values, conditions=None) -> EMResponse:
        raise NotImplementedError("CST automation is not implemented; use ManualUploadSolver.")


class BudgetedEvaluator:
    """Wraps a solver, enforces N_HF <= N_max and logs every high-fidelity call."""

    def __init__(self, solver: HighFidelitySolver, max_evaluations: Optional[int] = None):
        self.solver = solver
        self.max_evaluations = max_evaluations
        self.n_evaluations = 0
        self.log: List[Dict[str, Any]] = []

    @property
    def remaining(self) -> Optional[int]:
        return None if self.max_evaluations is None else self.max_evaluations - self.n_evaluations

    def __call__(self, values: Mapping[str, Any], conditions=None, tag: str = "") -> EMResponse:
        if self.max_evaluations is not None and self.n_evaluations >= self.max_evaluations:
            raise BudgetExceeded(f"High-fidelity budget of {self.max_evaluations} evaluations exhausted")
        resp = self.solver.evaluate(values, conditions)
        if resp.fidelity.counts_as_high_fidelity:
            self.n_evaluations += 1
        self.log.append({"index": self.n_evaluations, "tag": tag, "values": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in values.items()}})
        return resp
