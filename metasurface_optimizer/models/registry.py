"""Factory for surrogate models by name (used by configs, ablations and the API)."""
from __future__ import annotations

from typing import List, Optional

from metasurface_optimizer.geometry.geometry_generator import GeometryGenerator
from metasurface_optimizer.models.baseline_mlp import BaselineMLP
from metasurface_optimizer.models.cnn_surrogate import CNNSurrogate
from metasurface_optimizer.models.ensemble_uncertainty import EnsembleSurrogate
from metasurface_optimizer.models.features import FeatureExtractor
from metasurface_optimizer.models.gp_surrogate import GPSurrogate
from metasurface_optimizer.models.physics_surrogate import PhysicsInformedMLP
from metasurface_optimizer.models.random_forest import RandomForestSurrogate
from metasurface_optimizer.models.target_encoding import TargetEncoder
from metasurface_optimizer.models.torch_common import TrainConfig
from metasurface_optimizer.models.transformer_surrogate import TransformerSurrogate
from metasurface_optimizer.models.xgboost_model import XGBoostSurrogate
from metasurface_optimizer.physics.config import PhysicsConfig

SURROGATES = ["random_forest", "xgboost", "mlp", "gp", "cnn", "transformer", "physics_mlp", "ensemble_no_physics", "physics_ensemble"]


def build_surrogate(name: str, features: FeatureExtractor, encoder: TargetEncoder, seed: int = 0,
                    physics: Optional[PhysicsConfig] = None, generator: Optional[GeometryGenerator] = None,
                    train: Optional[TrainConfig] = None):
    physics = physics if physics is not None else PhysicsConfig()
    if name == "random_forest":
        return RandomForestSurrogate(features, encoder, seed)
    if name == "xgboost":
        return XGBoostSurrogate(features, encoder, seed)
    if name == "mlp":
        return BaselineMLP(features, encoder, seed, train=train)
    if name == "gp":
        return GPSurrogate(features, encoder, seed)
    if name == "cnn":
        return CNNSurrogate(features, encoder, seed, generator=generator, physics=PhysicsConfig.none())
    if name == "transformer":
        return TransformerSurrogate(features, encoder, seed)
    if name == "physics_mlp":
        return PhysicsInformedMLP(features, encoder, seed, physics=physics, train=train)
    if name in ("physics_ensemble", "ensemble_no_physics"):
        phys = physics if name == "physics_ensemble" else PhysicsConfig.none()
        members: List = [
            PhysicsInformedMLP(features, encoder, seed, physics=phys, train=train),
            GPSurrogate(features, encoder, seed),
            XGBoostSurrogate(features, encoder, seed),
        ]
        return EnsembleSurrogate(members, physics=phys, name=name)
    raise ValueError(f"Unknown surrogate {name!r}; choose from {SURROGATES}")
