"""Random-forest surrogate; uncertainty = spread of per-tree predictions."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from metasurface_optimizer.models.base import Surrogate


class RandomForestSurrogate(Surrogate):
    name = "random_forest"
    supports_std = True

    def _fit(self, Phi, Z, mask, Phi_colloc) -> None:
        self.model = RandomForestRegressor(
            n_estimators=self.params.get("n_estimators", 300), min_samples_leaf=self.params.get("min_samples_leaf", 1),
            max_features=self.params.get("max_features", 0.6), random_state=self.seed, n_jobs=1,
        ).fit(Phi, Z)

    def _predict(self, Phi):
        P = np.stack([t.predict(Phi) for t in self.model.estimators_])
        if P.ndim == 2:
            P = P[..., None]
        return P.mean(0), P.std(0)
