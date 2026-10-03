"""Configurable stopping rules (Eq. T1-T4); every decision is recorded.

* T1 objective convergence:  |J*_t - J*_{t-k}| < eps_J                (best validated composite)
* T2 hypervolume:            Delta HV < eps_HV for `patience` iterations
* T3 budget:                 N_HF >= N_max
* T4 uncertainty:            max sigma of the selected batch < sigma_min
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StoppingConfig:
    rules: List[str] = field(default_factory=lambda: ["budget", "objective", "hypervolume"])
    max_hf_evaluations: int = 40
    objective_window: int = 3
    objective_eps: float = 1e-3
    hv_eps: float = 1e-4
    hv_patience: int = 3
    sigma_min: float = 1e-3
    max_iterations: int = 50


class StoppingMonitor:
    def __init__(self, cfg: StoppingConfig):
        self.cfg = cfg
        self.best_history: List[float] = []
        self.hv_history: List[float] = []
        self.reason: Optional[str] = None

    def update(self, best: float, hv: Optional[float], n_hf: int, batch_sigma_max: Optional[float], iteration: int) -> Optional[str]:
        c = self.cfg
        self.best_history.append(best)
        if hv is not None:
            self.hv_history.append(hv)
        r = None
        if "budget" in c.rules and n_hf >= c.max_hf_evaluations:
            r = f"budget: N_HF={n_hf} >= N_max={c.max_hf_evaluations}"
        elif "objective" in c.rules and len(self.best_history) > c.objective_window and \
                abs(self.best_history[-1] - self.best_history[-1 - c.objective_window]) < c.objective_eps:
            r = f"objective convergence: |J_t - J_(t-{c.objective_window})| < {c.objective_eps}"
        elif "hypervolume" in c.rules and len(self.hv_history) > c.hv_patience and \
                all(abs(self.hv_history[-i] - self.hv_history[-i - 1]) < c.hv_eps for i in range(1, c.hv_patience + 1)):
            r = f"hypervolume: Delta HV < {c.hv_eps} for {c.hv_patience} iterations"
        elif "uncertainty" in c.rules and batch_sigma_max is not None and batch_sigma_max < c.sigma_min:
            r = f"uncertainty: max candidate sigma {batch_sigma_max:.2e} < {c.sigma_min}"
        elif iteration >= c.max_iterations:
            r = f"max iterations {c.max_iterations}"
        self.reason = r
        return r
