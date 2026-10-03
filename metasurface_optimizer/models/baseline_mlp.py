"""Plain MLP deep ensemble (no physics terms) — the 'ML surrogate without physics' baseline."""
from __future__ import annotations

from metasurface_optimizer.models.physics_surrogate import PhysicsInformedMLP
from metasurface_optimizer.physics.config import PhysicsConfig


class BaselineMLP(PhysicsInformedMLP):
    name = "mlp"

    def __init__(self, features, encoder, seed: int = 0, train=None, **kw):
        kw.pop("physics", None)
        super().__init__(features, encoder, seed, physics=PhysicsConfig.none(), train=train, **kw)
