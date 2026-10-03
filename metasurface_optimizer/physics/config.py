"""Switchable physics configuration (each term can be ablated independently)."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class PhysicsConfig:
    """Which physics terms are enabled, and the assumptions that justify them.

    ``passive``/``reciprocal`` are *statements about the structure* supplied
    by the user; the corresponding constraints are only applied when true.
    """

    passive: bool = True
    reciprocal: bool = True
    use_passivity: bool = True
    use_reciprocity: bool = True
    use_smoothness: bool = True
    use_phase_continuity: bool = False
    use_geometry: bool = True
    lambda_p: float = 1.0       # weight of passivity (+ reciprocity) loss
    lambda_s: float = 1e-3      # weight of smoothness loss
    lambda_phase: float = 1e-2
    lambda_g: float = 10.0      # weight of geometry penalty in the optimisation objective
    smoothness_robust: bool = True
    resonance_preserving: bool = True
    collocation_points: int = 32  # unlabeled designs on which physics losses are also enforced

    @property
    def any_enabled(self) -> bool:
        return any([self.passivity_active, self.reciprocity_active, self.use_smoothness, self.use_phase_continuity, self.use_geometry])

    @property
    def passivity_active(self) -> bool:
        return self.use_passivity and self.passive

    @property
    def reciprocity_active(self) -> bool:
        return self.use_reciprocity and self.reciprocal

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passivity_active"] = self.passivity_active
        d["reciprocity_active"] = self.reciprocity_active
        return d

    @classmethod
    def none(cls) -> "PhysicsConfig":
        return cls(use_passivity=False, use_reciprocity=False, use_smoothness=False, use_phase_continuity=False, use_geometry=False)
