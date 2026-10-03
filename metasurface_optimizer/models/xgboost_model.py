"""Gradient-boosted trees (XGBoost); uncertainty from a small bootstrap ensemble."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from metasurface_optimizer.models.base import Surrogate


class XGBoostSurrogate(Surrogate):
    name = "xgboost"
    supports_std = True

    def _fit(self, Phi, Z, mask, Phi_colloc) -> None:
        nb = int(self.params.get("n_bootstrap", 5))
        rng = np.random.default_rng(self.seed)
        self.models = []
        for b in range(nb):
            idx = rng.integers(0, len(Phi), len(Phi)) if nb > 1 else np.arange(len(Phi))
            m = xgb.XGBRegressor(
                n_estimators=self.params.get("n_estimators", 200), max_depth=self.params.get("max_depth", 3),
                learning_rate=self.params.get("learning_rate", 0.08), subsample=0.9, colsample_bytree=0.8,
                reg_lambda=1.0, tree_method="hist", random_state=self.seed * 100 + b, n_jobs=1, verbosity=0,
            )
            m.fit(Phi[idx], Z[idx])
            self.models.append(m)

    def _predict(self, Phi):
        P = np.stack([np.atleast_2d(m.predict(Phi)).reshape(len(Phi), -1) for m in self.models])
        return P.mean(0), (P.std(0) if len(self.models) > 1 else None)
