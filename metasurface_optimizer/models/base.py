"""Common surrogate interface and output transforms.

Every surrogate maps physical design vectors X (n x d) to the target vector
Y (n x k) defined by :class:`~metasurface_optimizer.models.target_encoding.TargetEncoder`
and returns a predictive mean and, where the model supports it, a standard
deviation. Inputs pass through the physics-aware feature map phi(x); outputs
are standardised (and, for non-neural models, the response block is
compressed with PCA to keep small-data regression well posed).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from metasurface_optimizer.models.features import FeatureExtractor
from metasurface_optimizer.models.target_encoding import TargetEncoder


class OutputTransform:
    """Standardise J-block and response block; optional PCA of the response block."""

    def __init__(self, m: int, n_pca: Optional[int] = None):
        self.m = m
        self.n_pca = n_pca

    def fit(self, Y: np.ndarray) -> "OutputTransform":
        Y = np.asarray(Y, float)
        self.col_mean = np.nanmean(Y, 0)
        self.col_mean = np.where(np.isnan(self.col_mean), 0.0, self.col_mean)
        sd = np.nanstd(Y, 0)
        self.col_std = np.where((sd > 1e-12) & ~np.isnan(sd), sd, 1.0)
        Z = self._standardize(Y)
        R = Z[:, self.m:]
        self.components = None
        if self.n_pca and R.shape[1] > 0:
            k = int(min(self.n_pca, R.shape[0] - 1 if R.shape[0] > 1 else 1, R.shape[1]))
            k = max(k, 1)
            _, s, vt = np.linalg.svd(R, full_matrices=False)
            self.components = vt[:k]  # (k, r)
            self.explained = float((s[:k] ** 2).sum() / max((s**2).sum(), 1e-12))
        return self

    def _standardize(self, Y: np.ndarray) -> np.ndarray:
        Z = (Y - self.col_mean) / self.col_std
        return np.where(np.isnan(Z), 0.0, Z)  # missing response entries -> column mean

    def forward(self, Y: np.ndarray) -> np.ndarray:
        Z = self._standardize(np.asarray(Y, float))
        if self.components is None:
            return Z
        return np.hstack([Z[:, : self.m], Z[:, self.m:] @ self.components.T])

    def inverse(self, Zc: np.ndarray, Sc: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Map (mean, std) in model space back to physical Y space (independent-component approximation)."""
        if self.components is not None:
            Z = np.hstack([Zc[:, : self.m], Zc[:, self.m:] @ self.components])
            S = None if Sc is None else np.hstack([Sc[:, : self.m], np.sqrt((Sc[:, self.m:] ** 2) @ (self.components**2))])
        else:
            Z, S = Zc, Sc
        Y = Z * self.col_std + self.col_mean
        return Y, (None if S is None else S * self.col_std)


class Surrogate(ABC):
    """Abstract surrogate f_hat(x) -> (mu(x), sigma(x))."""

    name: str = "abstract"
    supports_std: bool = False
    use_pca: bool = True

    def __init__(self, features: FeatureExtractor, encoder: TargetEncoder, seed: int = 0, n_pca: int = 8, **kwargs: Any):
        self.features = features
        self.encoder = encoder
        self.seed = seed
        self.n_pca = n_pca if self.use_pca else None
        self.params: Dict[str, Any] = dict(kwargs)
        self.fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray, X_colloc: Optional[np.ndarray] = None) -> "Surrogate":
        Phi = self.features.transform(X)
        self.out = OutputTransform(self.encoder.m, self.n_pca).fit(Y)
        Z = self.out.forward(Y)
        mask = ~np.isnan(np.asarray(Y, float))
        self._fit(Phi, Z, mask, None if X_colloc is None else self.features.transform(X_colloc))
        self.fitted = True
        self.n_train = len(X)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.fitted:
            raise RuntimeError(f"{self.name} is not fitted")
        Zm, Zs = self._predict(self.features.transform(X))
        return self.out.inverse(Zm, Zs)

    @abstractmethod
    def _fit(self, Phi: np.ndarray, Z: np.ndarray, mask: np.ndarray, Phi_colloc: Optional[np.ndarray]) -> None: ...

    @abstractmethod
    def _predict(self, Phi: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]: ...

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "seed": self.seed, "supports_std": self.supports_std, "n_pca": self.n_pca, "params": self.params}
