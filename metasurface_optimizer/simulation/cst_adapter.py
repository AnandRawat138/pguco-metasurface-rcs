"""CST Studio Suite adapter: interface placeholder.

Job packages are solver-independent (parameters with units, frequency,
excitation, outputs), so a CST worker could consume the same packages, but the
CST-side runner (CST Python API / VBA) is NOT implemented. This class exists
so the optimizer can be pointed at CST later without changes elsewhere.
"""
from __future__ import annotations

from pathlib import Path

from metasurface_optimizer.simulation.solver_adapter import SolverAdapter


class CSTAdapter(SolverAdapter):
    solver_name = "CST"

    def create_job(self, candidate, kind, iteration, candidate_id):
        raise NotImplementedError("CST automation is not implemented: use HFSS, the manual upload workflow, or contribute a CST runner")

    def parse_results(self, result_dir: Path, job):
        raise NotImplementedError("CST automation is not implemented")

    def validate_result(self, result_dir: Path, job):
        raise NotImplementedError("CST automation is not implemented")
