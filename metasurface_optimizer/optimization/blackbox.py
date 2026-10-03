"""High-fidelity black-box objective used by the baseline optimisers.

Wraps the budgeted solver so that every baseline is charged exactly one unit
of N_HF per evaluated *valid* design. Invalid geometries are rejected before
simulation (as a real workflow would) and receive the constraint penalty
without consuming budget. The full evaluation history is recorded for the
convergence curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from metasurface_optimizer.optimization.problem import OptimizationProblem
from metasurface_optimizer.solvers.base import BudgetedEvaluator, BudgetExceeded


@dataclass
class EvalRecord:
    x: np.ndarray
    J: Optional[np.ndarray]
    composite: float
    feasible: bool
    n_hf: int
    tag: str = ""


class SpaceMap:
    """Normalised free coordinates z in [0,1]^d_free <-> physical design vectors."""

    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self.schema = problem.schema
        self.free = np.where(self.schema.free_mask)[0]
        self.dim = len(self.free)

    def to_x(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(z)
        X = np.repeat(self.problem.x0[None], len(z), 0)
        X[:, self.free] = self.schema.lower[self.free] + np.clip(z, 0, 1) * (self.schema.upper[self.free] - self.schema.lower[self.free])
        return self.schema.snap(X)

    def to_z(self, x: np.ndarray) -> np.ndarray:
        return self.schema.normalize(np.atleast_2d(x))[:, self.free]


class HFObjective:
    def __init__(self, problem: OptimizationProblem, evaluator: BudgetedEvaluator, conditions: Optional[dict] = None):
        self.problem, self.evaluator, self.conditions = problem, evaluator, conditions
        self.space = SpaceMap(problem)
        self.history: List[EvalRecord] = []
        self.responses: List = []

    @property
    def exhausted(self) -> bool:
        r = self.evaluator.remaining
        return r is not None and r <= 0

    def evaluate_x(self, x: np.ndarray, tag: str = "") -> EvalRecord:
        x = np.asarray(x, float)
        pen = self.problem.validator.penalty(x)
        if not self.problem.validator.is_valid(x, generic=True):
            rec = EvalRecord(x, None, float(1.0 + self.problem.lambda_c * max(pen, 1e-3)), False, self.evaluator.n_evaluations, tag)
            self.history.append(rec)
            return rec
        if self.exhausted:
            raise BudgetExceeded("budget exhausted")
        resp = self.evaluator(self.problem.schema.to_dict_values(x), self.conditions, tag)
        J = self.problem.evaluate_response(resp, x)
        c = float(self.problem.scalarize(J[None], x[None])[0])
        rec = EvalRecord(x, J, c, True, self.evaluator.n_evaluations, tag)
        self.history.append(rec)
        self.responses.append(resp)
        return rec

    def scalar_z(self, Z: np.ndarray, tag: str = "") -> np.ndarray:
        return np.array([self.evaluate_x(x, tag).composite for x in self.space.to_x(Z)])

    def vector_z(self, Z: np.ndarray, tag: str = "") -> tuple:
        F, CV = [], []
        for x in self.space.to_x(Z):
            r = self.evaluate_x(x, tag)
            if r.J is None:
                F.append(np.full(self.problem.objectives.m, 1e6)); CV.append(max(1e-3, self.problem.validator.penalty(x)))
            else:
                F.append(self.problem.normalized(r.J)[0]); CV.append(0.0)
        return np.array(F), np.array(CV)

    def best(self) -> Optional[EvalRecord]:
        feas = [r for r in self.history if r.feasible]
        return min(feas, key=lambda r: r.composite) if feas else None

    def convergence(self) -> List[Dict[str, float]]:
        """Best-so-far composite versus N_HF."""
        out, best = [], np.inf
        for r in self.history:
            if r.feasible:
                best = min(best, r.composite)
                out.append({"n_hf": r.n_hf, "best": best})
        return out
