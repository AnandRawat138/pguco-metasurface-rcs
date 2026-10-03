"""Physics-aware design representation F_phys(x) (Eq. F1).

The raw design vector mixes millimetres, degrees and dimensionless material
constants. Surrogates instead receive

    phi(x) = [ u(x) ,  r(x) ,  e(x) ,  a(x) ,  p(x) ]

* u(x): free parameters mapped to [0, 1] by their schema bounds;
* r(x): geometric ratios to the period (L/P, W/P, gap/P = 1 - L/P), by role;
* e(x): electrical sizes at the reference frequency f_ref (taken from the
  objective band centre, never hard-coded): P/lambda, h sqrt(eps_r)/lambda,
  L sqrt(eps_eff)/lambda;
* a(x): rotation encoded as (cos 2psi, sin 2psi) (pi-periodic symmetry of a
  rectangular patch);
* p(x): optional prior-informed features from a pre-trained cell model
  (transfer learning, see :mod:`metasurface_optimizer.models.transfer`).

These features are generic dimensionless quantities. They deliberately do
NOT include the closed-form resonance of any specific circuit model, so that
results on the synthetic analytic oracle are not inflated by feeding the
oracle's own formula to the surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from metasurface_optimizer.geometry.parameter_schema import ParameterSchema

C0 = 299_792_458.0


@dataclass
class FeatureExtractor:
    schema: ParameterSchema
    reference_frequency: Optional[float] = None  # Hz
    mode: str = "physics"  # "raw" | "physics"
    prior_feature_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._free = np.where(self.schema.free_mask)[0]
        self.names = self._build_names()

    def _roles(self, role: str) -> List[int]:
        return [self.schema.index(p.name) for p in self.schema.by_role(role)]

    def _build_names(self) -> List[str]:
        n = [f"u:{self.schema.names[i]}" for i in self._free]
        if self.mode == "physics":
            n += self._physics(np.atleast_2d(self.schema.default_vector()), names_only=True)
        return n

    def _physics(self, X: np.ndarray, names_only: bool = False):
        cols, names = [], []
        L = self.schema.length_scale_to_mm() * 1e-3  # schema length unit -> metres
        per = self._roles("period")
        P = X[:, per[0]] if per else None
        lam = C0 / self.reference_frequency if self.reference_frequency else None
        eps_i = self._roles("permittivity")
        eps = X[:, eps_i[0]] if eps_i else np.ones(len(X))
        for role in ("length", "width"):
            for i in self._roles(role):
                if P is not None:
                    cols.append(X[:, i] / P); names.append(f"r:{self.schema.names[i]}/P")
                if lam is not None:
                    cols.append(X[:, i] * L * np.sqrt((eps + 1) / 2) / lam); names.append(f"e:{self.schema.names[i]}_elec")
        if P is not None and lam is not None:
            cols.append(P * L / lam); names.append("e:P/lambda")
        for i in self._roles("thickness"):
            if lam is not None:
                cols.append(X[:, i] * L * np.sqrt(eps) / lam); names.append(f"e:{self.schema.names[i]}_elec")
        for i in self._roles("rotation"):
            ang = np.radians(X[:, i]) if self.schema.parameters[i].unit.startswith("deg") else X[:, i]
            cols.append(np.cos(2 * ang)); names.append(f"a:cos2_{self.schema.names[i]}")
            cols.append(np.sin(2 * ang)); names.append(f"a:sin2_{self.schema.names[i]}")
        if names_only:
            return names
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        parts = [self.schema.normalize(X)[:, self._free]]
        if self.mode == "physics":
            parts.append(self._physics(X))
        if self.prior_feature_fn is not None:
            parts.append(self.prior_feature_fn(X))
        return np.hstack(parts)

    @property
    def dim(self) -> int:
        return self.transform(self.schema.default_vector()).shape[1]

    def describe(self) -> dict:
        return {"mode": self.mode, "reference_frequency": self.reference_frequency, "names": self.names,
                "prior_features": self.prior_feature_fn is not None}
