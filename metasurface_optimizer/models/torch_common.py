"""Shared PyTorch training loop with the physics-aware loss (Eq. L1).

    L = L_data + lambda_p (L_pass + L_rec) + lambda_s L_smooth + lambda_phase L_phase

* L_data   = masked mean squared error on the standardised targets;
* L_pass   = passivity penalty on every predicted complex reflection /
             transmission pair (only if the structure is declared passive);
* L_rec    = reciprocity penalty |S21 - S12|^2 (only if declared reciprocal
             and both entries are part of the target);
* L_smooth = (resonance-weighted, optionally Huber) squared second difference
             of every predicted spectrum along frequency;
* L_phase  = optional phase-continuity term.

Physics terms are evaluated in *physical* units (de-standardised outputs), on
the training designs and on unlabeled collocation designs (semi-supervised
physics regularisation; no labels are needed there). The geometry term of
Eq. L1 acts in the optimisation objective, not here (all training designs are
valid geometries by construction).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from metasurface_optimizer.models.target_encoding import TargetEncoder
from metasurface_optimizer.physics.config import PhysicsConfig
from metasurface_optimizer.physics.passivity import passivity_loss_torch
from metasurface_optimizer.physics.phase_consistency import phase_continuity_loss_torch
from metasurface_optimizer.physics.reciprocity import reciprocity_loss_torch
from metasurface_optimizer.physics.spectral_smoothness import resonance_weights, smoothness_loss_torch

# Small full-batch problems: multithreading overhead dominates, so use one thread.
torch.set_num_threads(1)


@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 3e-3
    weight_decay: float = 1e-4
    hidden: int = 64
    depth: int = 3
    n_members: int = 3
    dropout: float = 0.0
    j_weight: float = 3.0  # extra weight of the objective block in L_data (it drives optimisation)


class PhysicsLoss:
    """Evaluates the physics terms on de-standardised predictions.

    The response block of the target vector is laid out as B contiguous
    spectra of F samples (see TargetEncoder), so it is viewed as an
    (n, B, F) tensor and every term is computed without gathers.
    """

    def __init__(self, encoder: TargetEncoder, cfg: PhysicsConfig, mean: np.ndarray, std: np.ndarray, Y_train: Optional[np.ndarray] = None):
        self.enc, self.cfg = encoder, cfg
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.m, self.F = encoder.m, len(encoder.frequency)
        names = [n for n in encoder.slices if n != "J"]
        self.B = len(names)
        block = {n: b for b, n in enumerate(names)}
        pairs = {k: (block[f"S{k[0]}{k[1]}_{k[2]}_re"], block[f"S{k[0]}{k[1]}_{k[2]}_im"]) for _, _, k in encoder.complex_pairs()}
        refl, trans, has_t = [], [], []
        for (i, j, p), (bre, bim) in pairs.items():
            if i == j:
                t = pairs.get((3 - i, i, p)) if i in (1, 2) else None
                refl.append([bre, bim]); trans.append(list(t) if t else [bre, bim]); has_t.append(1.0 if t else 0.0)
        self.refl = torch.tensor(refl, dtype=torch.long) if refl else None  # (G, 2)
        self.trans = torch.tensor(trans, dtype=torch.long) if refl else None
        self.has_t = torch.tensor(has_t)[None, :, None, None] if refl else None
        self.rec = [(torch.tensor(pairs[(2, 1, p)]), torch.tensor(pairs[(1, 2, p)])) for (i, j, p) in pairs if (i, j) == (2, 1) and (1, 2, p) in pairs]
        self.pair_blocks = list(pairs.values())
        self.spec_scale = torch.tensor([0.1 if n.startswith("RCSred") else 1.0 for n in names])[None, :, None] if names else None
        self.spec_w = None
        if names and cfg.resonance_preserving and Y_train is not None:
            W = []
            for n in names:
                ref = Y_train[:, encoder.slices[n]]
                ref = ref[~np.isnan(ref).any(1)]
                W.append(resonance_weights(ref) if len(ref) else np.ones(self.F - 2))
            self.spec_w = torch.tensor(np.array(W), dtype=torch.float32)[None]

    def __call__(self, z: torch.Tensor, report: bool = False) -> Tuple[torch.Tensor, Dict[str, float]]:
        y = z * self.std + self.mean
        total = torch.zeros(())
        parts: Dict[str, float] = {}
        cfg = self.cfg
        if self.B == 0:
            return total, parts
        spec = y[:, self.m:].reshape(len(y), self.B, self.F)
        if cfg.passivity_active and self.refl is not None:
            r = spec[:, self.refl]  # (n, G, 2, F)
            t = spec[:, self.trans] * self.has_t
            pw = (r**2).sum(2) + (t**2).sum(2)
            lp = torch.relu(pw - 1.0).pow(2).mean()
            if cfg.reciprocity_active:
                for a, b in self.rec:
                    lp = lp + ((spec[:, a] - spec[:, b]) ** 2).sum(1).mean()
            total = total + cfg.lambda_p * lp
            if report:
                parts["physics"] = float(lp.detach())
        if cfg.use_smoothness:
            seq = spec * self.spec_scale
            d2 = seq[..., 2:] - 2 * seq[..., 1:-1] + seq[..., :-2]
            pen = torch.nn.functional.huber_loss(d2, torch.zeros_like(d2), reduction="none", delta=0.05) if cfg.smoothness_robust else d2**2
            if self.spec_w is not None:
                pen = pen * self.spec_w
            ls = pen.mean() * self.B
            total = total + cfg.lambda_s * ls
            if report:
                parts["smooth"] = float(ls.detach())
        if cfg.use_phase_continuity and self.pair_blocks:
            lph = sum(phase_continuity_loss_torch(spec[:, a], spec[:, b]) for a, b in self.pair_blocks)
            total = total + cfg.lambda_phase * lph
            if report:
                parts["phase"] = float(lph.detach())
        return total, parts


def make_mlp(n_in: int, n_out: int, hidden: int, depth: int, dropout: float = 0.0) -> nn.Module:
    layers: List[nn.Module] = []
    d = n_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = hidden
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def train_network(
    net: nn.Module, Phi: np.ndarray, Z: np.ndarray, mask: np.ndarray, m: int, tc: TrainConfig,
    physics: Optional[PhysicsLoss] = None, Phi_colloc: Optional[np.ndarray] = None, seed: int = 0,
    input_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    """Full-batch Adam training; returns final loss components."""
    torch.manual_seed(seed)
    X = torch.tensor(Phi, dtype=torch.float32)
    T = torch.tensor(Z, dtype=torch.float32)
    M = torch.tensor(mask, dtype=torch.float32)
    w = torch.ones(T.shape[1])
    w[:m] = tc.j_weight
    Xc = torch.tensor(Phi_colloc, dtype=torch.float32) if Phi_colloc is not None and physics is not None else None
    opt = torch.optim.Adam(net.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.epochs)
    f = input_fn or (lambda t: t)
    last: Dict[str, float] = {}
    net.train()
    for ep in range(tc.epochs):
        opt.zero_grad()
        out = net(f(X))
        ldata = ((out - T) ** 2 * M * w).sum() / (M * w).sum().clamp_min(1.0)
        loss = ldata
        report = ep == tc.epochs - 1
        parts = {"data": float(ldata.detach())} if report else {}
        if physics is not None:
            lp, pp = physics(out, report)
            if Xc is not None:
                lpc, _ = physics(net(f(Xc)))
                lp = 0.5 * (lp + lpc)
            loss = loss + lp
            parts.update(pp)
        loss.backward()
        opt.step()
        sched.step()
        if ep == tc.epochs - 1:
            last = parts
    net.eval()
    return last


class DeepEnsembleMixin:
    """Trains ``n_members`` independently initialised networks; mean/std over members."""

    def _members_predict(self, Phi: np.ndarray, input_fn=None) -> Tuple[np.ndarray, np.ndarray]:
        X = torch.tensor(Phi, dtype=torch.float32)
        f = input_fn or (lambda t: t)
        with torch.no_grad():
            P = np.stack([net(f(X)).numpy() for net in self.nets])
        return P.mean(0), (P.std(0) if len(self.nets) > 1 else None)
