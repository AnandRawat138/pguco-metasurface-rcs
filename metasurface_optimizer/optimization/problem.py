"""The optimisation problem: schema + objectives + normalisation + constraints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from metasurface_optimizer.electromagnetics.response import EMResponse
from metasurface_optimizer.geometry.geometry_validator import GeometryValidator
from metasurface_optimizer.geometry.parameter_schema import ParameterSchema
from metasurface_optimizer.optimization.objectives import ObjectiveNormalizer, ObjectiveSet, composite
from metasurface_optimizer.physics.config import PhysicsConfig


@dataclass
class OptimizationProblem:
    """min_x  J(x) = sum_k lambda_k J~_k(x) + lambda_c J_constraint(x)   s.t.  g_i(x) <= 0, x in bounds.

    ``x0``/``J0`` are the existing design and its baseline objective vector.
    In multi-objective mode the vector J~(x) is optimised instead (Pareto).
    """

    schema: ParameterSchema
    objectives: ObjectiveSet
    validator: GeometryValidator
    normalizer: ObjectiveNormalizer
    x0: np.ndarray
    J0: Optional[np.ndarray] = None
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    lambda_c: float = 10.0

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, float)
        self.objectives.context.setdefault("validator", self.validator)
        if self.normalizer.floors is None:
            self.normalizer.floors = self.objectives.norm_floors.tolist()

    @property
    def weights(self) -> np.ndarray:
        return self.objectives.weights

    def set_baseline(self, J0: np.ndarray) -> None:
        """Record J0; baseline-relative normalisation is fitted here, min-max / z-score
        are fitted later on the initial high-fidelity data (see fit_normalizer)."""
        self.J0 = np.asarray(J0, float)
        if self.normalizer.method == "baseline_relative":
            self.normalizer.fit(self.J0[None], self.J0)

    def fit_normalizer(self, J: np.ndarray) -> None:
        """Fit (once) min-max / z-score statistics on initial data, then freeze them."""
        if self.normalizer.method != "baseline_relative" and not self.normalizer.stats:
            self.normalizer.fit(np.atleast_2d(J), self.J0)

    def evaluate_response(self, resp: EMResponse, x: np.ndarray) -> np.ndarray:
        return self.objectives.evaluate(resp, x)

    def penalty(self, X: np.ndarray) -> np.ndarray:
        if not self.physics.use_geometry:
            return np.zeros(len(np.atleast_2d(X)))
        return np.array([self.validator.penalty(x) for x in np.atleast_2d(X)])

    def normalized(self, J: np.ndarray) -> np.ndarray:
        return self.normalizer.transform(np.atleast_2d(J))

    def scalarize(self, J: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Composite J(x) (Eq. C1) for raw objective vectors J (n x m)."""
        pen = 0.0 if X is None else self.penalty(X)
        return composite(self.normalized(J), self.weights, pen, self.lambda_c * (self.physics.lambda_g / 10.0 if self.physics.use_geometry else 0.0))

    def scalarize_with_std(self, mu: np.ndarray, sd: np.ndarray, X: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Composite mean and std assuming independent objective errors:
        sigma_J = sqrt( sum_k (lambda_k s_k sigma_k)^2 ), s_k = dJ~_k/dJ_k."""
        mean = self.scalarize(mu, X)
        s = self.normalizer.scale() * self.weights
        std = np.sqrt(((np.atleast_2d(sd) * s) ** 2).sum(1))
        return mean, std

    def is_feasible(self, x: np.ndarray, generic: bool = True) -> bool:
        return self.validator.is_valid(x, generic)

    def describe(self) -> Dict[str, Any]:
        return {
            "objectives": self.objectives.to_config(), "normalization": self.normalizer.to_dict(),
            "lambda_c": self.lambda_c, "physics": self.physics.to_dict(),
            "x0": self.schema.to_dict_values(self.x0), "J0": None if self.J0 is None else self.J0.tolist(),
        }
