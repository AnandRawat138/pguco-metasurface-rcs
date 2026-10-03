"""Heterogeneous ensemble with epistemic uncertainty (Eq. E1-E2).

    mu(x)      = (1/M) sum_m f_m(x)
    sigma^2(x) = (1/M) sum_m (f_m(x) - mu(x))^2

where M counts *every* predictor: a deep-ensemble member contributes each of
its networks as a separate predictor, a GP / forest contributes its mean.
Optionally (``include_member_variance``) the average of the members' own
predictive variances is added (total = epistemic disagreement + mean
member variance); the default follows Eq. E2 exactly.

A post-hoc passivity projection is applied to the decoded complex response
block when the physics configuration declares the structure passive, so that
non-neural members (GP, trees) cannot output |S| > 1.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from metasurface_optimizer.models.base import Surrogate
from metasurface_optimizer.physics.config import PhysicsConfig
from metasurface_optimizer.physics.passivity import project_passive


class EnsembleSurrogate:
    name = "ensemble"
    supports_std = True

    def __init__(self, members: List[Surrogate], include_member_variance: bool = False, physics: Optional[PhysicsConfig] = None,
                 name: str = "ensemble"):
        if not members:
            raise ValueError("Ensemble needs at least one member")
        self.members = members
        self.include_member_variance = include_member_variance
        self.physics = physics
        self.encoder = members[0].encoder
        self.name = name

    def fit(self, X, Y, X_colloc=None) -> "EnsembleSurrogate":
        for m in self.members:
            m.fit(X, Y, X_colloc)
        return self

    def member_predictions(self, X) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
        preds, stds = [], []
        for m in self.members:
            if hasattr(m, "nets"):  # deep ensemble: each network is a predictor
                Phi = m._inputs(X) if hasattr(m, "_inputs") else m.features.transform(X)
                import torch

                with torch.no_grad():
                    for net in m.nets:
                        z = net(torch.tensor(Phi, dtype=torch.float32)).numpy()
                        y, _ = m.out.inverse(z, None)
                        preds.append(y); stds.append(None)
            else:
                mu, sd = m.predict(X)
                preds.append(mu); stds.append(sd)
        return np.stack(preds), stds

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        P, stds = self.member_predictions(np.atleast_2d(X))
        mu = P.mean(0)
        var = P.var(0)
        if self.include_member_variance:
            mv = [s**2 for s in stds if s is not None]
            if mv:
                var = var + np.mean(mv, axis=0)
        mu = self._project(mu)
        return mu, np.sqrt(var)

    def _project(self, Y: np.ndarray) -> np.ndarray:
        if self.physics is None or not self.physics.passivity_active:
            return Y
        Y = Y.copy()
        for sre, sim, (i, j, p) in self.encoder.complex_pairs():
            if i == j:
                s = Y[:, sre] + 1j * Y[:, sim]
                s, _ = project_passive(s)
                Y[:, sre], Y[:, sim] = s.real, s.imag
        return Y

    def describe(self) -> Dict:
        return {"name": self.name, "members": [m.describe() for m in self.members], "include_member_variance": self.include_member_variance,
                "uncertainty": "sigma^2 = (1/M) sum_m (f_m - mu)^2" + (" + mean member variance" if self.include_member_variance else "")}
