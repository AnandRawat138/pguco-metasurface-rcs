"""CNN surrogate on the rasterised unit-cell geometry.

Input: binary top-metal mask of the (super-)cell from the geometry generator
(image representation of the design) concatenated, after the convolutional
encoder, with the physics-aware scalar features (thickness, permittivity,
electrical sizes, ...). Deep ensemble for uncertainty; optional physics loss.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from metasurface_optimizer.geometry.geometry_generator import GeometryGenerator
from metasurface_optimizer.models.base import Surrogate
from metasurface_optimizer.models.torch_common import DeepEnsembleMixin, PhysicsLoss, TrainConfig, train_network
from metasurface_optimizer.physics.config import PhysicsConfig


class _CNN(nn.Module):
    def __init__(self, res: int, n_scalar: int, n_out: int, hidden: int):
        super().__init__()
        self.res = res
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, padding_mode="circular"), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1, padding_mode="circular"), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1, padding_mode="circular"), nn.SiLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(nn.Linear(16 * 16 + n_scalar, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_out))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        r2 = self.res * self.res
        img = t[:, :r2].reshape(-1, 1, self.res, self.res)
        return self.head(torch.cat([self.conv(img).flatten(1), t[:, r2:]], 1))


class CNNSurrogate(DeepEnsembleMixin, Surrogate):
    name = "cnn"
    supports_std = True
    use_pca = False

    def __init__(self, features, encoder, seed: int = 0, generator: Optional[GeometryGenerator] = None,
                 resolution: int = 32, physics: Optional[PhysicsConfig] = None, train: Optional[TrainConfig] = None, **kw):
        super().__init__(features, encoder, seed, **kw)
        if generator is None:
            generator = GeometryGenerator(features.schema)
        self.generator, self.res = generator, resolution
        self.physics_cfg = physics if physics is not None else PhysicsConfig.none()
        self.tc = train or TrainConfig(epochs=400, hidden=64, n_members=3)

    def _inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        masks = np.stack([self.generator.rasterize(x, self.res).astype(np.float32).ravel() for x in X])
        return np.hstack([masks, self.features.transform(X)])

    def fit(self, X, Y, X_colloc=None):
        from metasurface_optimizer.models.base import OutputTransform

        Phi = self._inputs(X)
        self.out = OutputTransform(self.encoder.m, None).fit(Y)
        Z = self.out.forward(Y)
        mask = ~np.isnan(np.asarray(Y, float))
        phys = None
        if self.physics_cfg.passivity_active or self.physics_cfg.use_smoothness:
            phys = PhysicsLoss(self.encoder, self.physics_cfg, self.out.col_mean, self.out.col_std, np.asarray(Y, float))
        Pc = None if X_colloc is None or phys is None else self._inputs(X_colloc)
        self.nets = []
        for k in range(self.tc.n_members):
            net = _CNN(self.res, Phi.shape[1] - self.res**2, Z.shape[1], self.tc.hidden)
            train_network(net, Phi, Z, mask, self.encoder.m, self.tc, phys, Pc, seed=self.seed * 1000 + k)
            self.nets.append(net)
        self.fitted, self.n_train = True, len(X)
        return self

    def predict(self, X):
        Zm, Zs = self._members_predict(self._inputs(X))
        return self.out.inverse(Zm, Zs)

    def _fit(self, *a):  # pragma: no cover - fit() is overridden
        raise NotImplementedError

    def _predict(self, Phi):  # pragma: no cover
        raise NotImplementedError
