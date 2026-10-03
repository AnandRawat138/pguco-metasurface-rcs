"""Solver-independent interface used by the optimizer side.

The optimizer and the simulation manager only talk to :class:`SolverAdapter`;
nothing on PC-1 imports PyAEDT. HFSS (``hfss_adapter``), a future CST adapter
(``cst_adapter``) and the MOCK pipeline test all implement this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hfss_worker.protocol import JobState
from metasurface_optimizer.simulation.result_parser import SimulationResult


@dataclass
class SimulationJob:
    job_id: str
    candidate_id: str
    kind: str                      # baseline | initial | candidate | verify_template
    iteration: int
    job_dir: str                   # canonical package on PC-1 (never edited after creation)
    simulation_type: str           # HFSS | MOCK | DRY_RUN
    x: List[float] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)
    tag: str = ""
    prediction: Optional[Dict[str, Any]] = None
    state: str = JobState.PENDING.value
    submitted: bool = False
    transport: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    result_dir: Optional[str] = None
    error: str = ""
    imported: bool = False
    created: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SolverAdapter(ABC):
    """create_job -> submit -> get_status -> collect_result -> validate_result."""

    solver_name = "abstract"

    def __init__(self, transport=None, simulation_type: str = "HFSS"):
        self.transport = transport
        self.simulation_type = simulation_type

    @abstractmethod
    def create_job(self, candidate: Dict[str, Any], kind: str, iteration: int, candidate_id: str) -> SimulationJob:
        """Write a job package for one candidate (validated beforehand) and return its record."""

    def submit(self, jobs: List[SimulationJob]) -> Dict[str, Any]:
        if self.simulation_type == "DRY_RUN":
            raise RuntimeError("DRY_RUN mode only generates job packages; nothing is submitted")
        if self.transport is None:
            raise RuntimeError("no transport configured (manual batch, shared folder or mock)")
        info = self.transport.submit(jobs)
        for j in jobs:
            j.submitted = True
        return info

    def get_status(self, job: SimulationJob) -> str:
        if self.transport is None or not job.submitted:
            return job.state
        return self.transport.status(job)

    def collect_result(self, job: SimulationJob) -> Optional[SimulationResult]:
        """The parsed result once the job is final on the worker side, else None."""
        if self.transport is None:
            return None
        rd = self.transport.result_dir(job)
        if rd is None:
            return None
        return self.parse_results(Path(rd), job)

    @abstractmethod
    def parse_results(self, result_dir: Path, job: SimulationJob) -> SimulationResult:
        ...

    @abstractmethod
    def validate_result(self, result_dir: Path, job: SimulationJob) -> List[str]:
        ...
