"""Transformer surrogate treating each design feature as a token.

Token_i = E_id[i] + w_i * value_i (learned feature-identity embedding plus a
value projection). A small TransformerEncoder mixes tokens; mean-pooling and
a linear head produce the target vector. Being permutation-aware through the
identity embedding, it accepts any parameter schema. Deep ensemble for
uncertainty.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from metasurface_optimizer.models.base import Surrogate
from metasurface_optimizer.models.torch_common import DeepEnsembleMixin, PhysicsLoss, TrainConfig, train_network
from metasurface_optimizer.physics.config import PhysicsConfig


class _TokenTransformer(nn.Module):
    def __init__(self, n_tokens: int, n_out: int, d: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.ident = nn.Parameter(torch.randn(n_tokens, d) * 0.1)
        self.val = nn.Parameter(torch.randn(n_tokens, d) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_tokens, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.ident[None] + x[..., None] * self.val[None] + self.bias[None]
        return self.head(self.encoder(tok).mean(1))


class TransformerSurrogate(DeepEnsembleMixin, Surrogate):
    name = "transformer"
    supports_std = True
    use_pca = False

    def __init__(self, features, encoder, seed: int = 0, physics: Optional[PhysicsConfig] = None, train: Optional[TrainConfig] = None, **kw):
        super().__init__(features, encoder, seed, **kw)
        self.physics_cfg = physics if physics is not None else PhysicsConfig.none()
        self.tc = train or TrainConfig(epochs=400, lr=2e-3, n_members=3)

    def _fit(self, Phi, Z, mask, Phi_colloc) -> None:
        phys = None
        if self.physics_cfg.passivity_active or self.physics_cfg.use_smoothness:
            phys = PhysicsLoss(self.encoder, self.physics_cfg, self.out.col_mean, self.out.col_std, None)
        self.nets = []
        for k in range(self.tc.n_members):
            torch.manual_seed(self.seed * 1000 + k)
            net = _TokenTransformer(Phi.shape[1], Z.shape[1])
            train_network(net, Phi, Z, mask, self.encoder.m, self.tc, phys, Phi_colloc, seed=self.seed * 1000 + k)
            self.nets.append(net)

    def _predict(self, Phi):
        return self._members_predict(Phi)
