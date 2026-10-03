"""Physics-informed deep-ensemble surrogate (the neural member of the proposed ensemble).

A multi-task MLP predicts the full target vector (objective block + complex
response block) and is trained with the physics-aware loss of
:mod:`metasurface_optimizer.models.torch_common` (Eq. L1). Uncertainty comes
from a deep ensemble of independently initialised networks.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from metasurface_optimizer.models.base import Surrogate
from metasurface_optimizer.models.torch_common import DeepEnsembleMixin, PhysicsLoss, TrainConfig, make_mlp, train_network
from metasurface_optimizer.physics.config import PhysicsConfig


class PhysicsInformedMLP(DeepEnsembleMixin, Surrogate):
    name = "physics_mlp"
    supports_std = True
    use_pca = False

    def __init__(self, features, encoder, seed: int = 0, physics: Optional[PhysicsConfig] = None,
                 train: Optional[TrainConfig] = None, **kw):
        super().__init__(features, encoder, seed, **kw)
        self.physics_cfg = physics if physics is not None else PhysicsConfig()
        self.tc = train or TrainConfig()
        self.train_log = []

    def _physics_loss(self, Y_raw: Optional[np.ndarray]) -> Optional[PhysicsLoss]:
        cfg = self.physics_cfg
        if not (cfg.passivity_active or cfg.reciprocity_active or cfg.use_smoothness or cfg.use_phase_continuity):
            return None
        return PhysicsLoss(self.encoder, cfg, self.out.col_mean, self.out.col_std, Y_raw)

    def fit(self, X, Y, X_colloc=None):
        self._Y_raw = np.asarray(Y, float)
        return super().fit(X, Y, X_colloc)

    def _fit(self, Phi, Z, mask, Phi_colloc) -> None:
        phys = self._physics_loss(self._Y_raw)
        self.nets = []
        for k in range(self.tc.n_members):
            net = make_mlp(Phi.shape[1], Z.shape[1], self.tc.hidden, self.tc.depth, self.tc.dropout)
            log = train_network(net, Phi, Z, mask, self.encoder.m, self.tc, phys, Phi_colloc, seed=self.seed * 1000 + k)
            self.nets.append(net)
            self.train_log.append(log)

    def _predict(self, Phi) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._members_predict(Phi)

    def describe(self):
        d = super().describe()
        d["physics"] = self.physics_cfg.to_dict()
        d["train"] = self.tc.__dict__
        d["final_losses"] = self.train_log[-self.tc.n_members:] if self.train_log else []
        return d
