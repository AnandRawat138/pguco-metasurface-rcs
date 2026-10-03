"""Gaussian-process surrogate (Matern-5/2 ARD + white noise).

The J-block outputs get one GP each (own hyper-parameters); the PCA-compressed
response block shares one multi-output GP. Predictive std is the GP posterior std.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from metasurface_optimizer.models.base import Surrogate


def _kernel(d: int):
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e0))


class GPSurrogate(Surrogate):
    name = "gp"
    supports_std = True

    def _fit(self, Phi, Z, mask, Phi_colloc) -> None:
        m = self.encoder.m
        d = Phi.shape[1]
        restarts = int(self.params.get("n_restarts", 2))
        self.gps = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for k in range(m):
                self.gps.append(GaussianProcessRegressor(_kernel(d), normalize_y=False, n_restarts_optimizer=restarts, random_state=self.seed).fit(Phi, Z[:, k]))
            self.gp_resp = None
            if Z.shape[1] > m:
                self.gp_resp = GaussianProcessRegressor(_kernel(d), normalize_y=False, n_restarts_optimizer=0, random_state=self.seed).fit(Phi, Z[:, m:])

    def _predict(self, Phi):
        mus, sds = [], []
        for gp in self.gps:
            mu, sd = gp.predict(Phi, return_std=True)
            mus.append(mu); sds.append(sd)
        mean, std = np.column_stack(mus), np.column_stack(sds)
        if self.gp_resp is not None:
            mu, sd = self.gp_resp.predict(Phi, return_std=True)
            mu = mu.reshape(len(Phi), -1)
            sd = np.broadcast_to(np.asarray(sd).reshape(len(Phi), -1), mu.shape)
            mean, std = np.hstack([mean, mu]), np.hstack([std, sd])
        return mean, std
