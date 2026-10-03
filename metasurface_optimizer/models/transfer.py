"""Transfer learning from prior (public) data through a cell-level response model.

Public FSS/metasurface datasets describe *single unit cells* with their own
parameterisations and frequency grids; they rarely match the user's final
stealth problem. We therefore transfer at the level where the physics is
shared: a pre-trained **cell prior model**

    g_theta : (L/P, W/P, P f / c, h f / c, eps_r, tan_delta, cos 2psi, sin 2psi) -> (Re Gamma, Im Gamma) per polarization

operating on dimensionless, frequency-explicit inputs (so any frequency grid
or physical scale is compatible). For a target design, the cells declared in
``schema.metadata['cells']`` are evaluated with g_theta at ``K`` frequencies
in the objective band, and compact *prior-informed features* are appended to
phi(x) (Eq. F1, block p(x)):

    [cos dphi_AB(f_k), sin dphi_AB(f_k)]_k , mean |Gamma_c|   (two-cell designs)
    [cos phi_c(f_k), sin phi_c(f_k)]_k , mean |Gamma_c|        (single-cell designs)

The downstream surrogate (any member type) then learns the array-level
objectives on top. With no prior data the block is simply absent (ablation A1).
Optionally the cell model is fine-tuned on the target's own cell responses
(S11 of each state at normal incidence) after pre-training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from metasurface_optimizer.geometry.parameter_schema import ParameterSchema
from metasurface_optimizer.models.torch_common import make_mlp

C0 = 299_792_458.0
CELL_INPUTS = ["L/P", "W/P", "P*f/c", "h*f/c", "eps_r", "tan_delta", "cos2psi", "sin2psi"]


def cell_inputs(P, L, W, h, eps, tand, rot_deg, f, length_unit_m: float = 1e-3) -> np.ndarray:
    """Dimensionless cell-model inputs; all arrays broadcast; f in Hz."""
    psi = np.radians(rot_deg)
    Pm, hm = P * length_unit_m, h * length_unit_m
    cols = np.broadcast_arrays(L / P, W / P, Pm * f / C0, hm * f / C0, eps, tand, np.cos(2 * psi), np.sin(2 * psi))
    return np.stack([np.ravel(c) for c in cols], -1)


@dataclass
class CellSample:
    inputs: np.ndarray  # (n, 8)
    targets: np.ndarray  # (n, 2*n_pol) Re/Im per polarization
    polarizations: Tuple[str, ...]


class CellPriorModel:
    """Small MLP ensemble g_theta for cell reflection; pre-trained on prior data."""

    def __init__(self, polarizations: Sequence[str] = ("TE", "TM"), hidden: int = 96, depth: int = 3, n_members: int = 2, seed: int = 0):
        self.pols = tuple(polarizations)
        self.hidden, self.depth, self.n_members, self.seed = hidden, depth, n_members, seed
        self.nets: List[torch.nn.Module] = []
        self.history: List[Dict[str, float]] = []

    def _train(self, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float, init: bool, batch: int = 4096) -> float:
        Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        if init:
            self.mu, self.sd = X.mean(0), np.where(X.std(0) > 1e-9, X.std(0), 1.0)
            self.nets = []
            for k in range(self.n_members):
                torch.manual_seed(self.seed * 100 + k)
                self.nets.append(make_mlp(X.shape[1], Y.shape[1], self.hidden, self.depth))
        Xn = (Xt - torch.tensor(self.mu, dtype=torch.float32)) / torch.tensor(self.sd, dtype=torch.float32)
        last = 0.0
        g = torch.Generator().manual_seed(self.seed)
        for net in self.nets:
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            for ep in range(epochs):
                perm = torch.randperm(len(Xn), generator=g)
                for i in range(0, len(Xn), batch):
                    idx = perm[i:i + batch]
                    opt.zero_grad()
                    out = net(Xn[idx])
                    # passivity of the prior model output: |Gamma| <= 1
                    re, im = out[:, 0::2], out[:, 1::2]
                    loss = ((out - Yt[idx]) ** 2).mean() + torch.relu(re**2 + im**2 - 1).pow(2).mean()
                    loss.backward()
                    opt.step()
                last = float(loss)
        return last

    def pretrain(self, data: CellSample, epochs: int = 60, lr: float = 2e-3) -> "CellPriorModel":
        if tuple(data.polarizations) != self.pols:
            raise ValueError("Polarization mismatch between prior data and cell model")
        loss = self._train(data.inputs, data.targets, epochs, lr, init=True)
        self.history.append({"stage": "pretrain", "n": len(data.inputs), "loss": loss})
        return self

    def finetune(self, data: CellSample, epochs: int = 100, lr: float = 5e-4) -> "CellPriorModel":
        loss = self._train(data.inputs, data.targets, epochs, lr, init=False, batch=1024)
        self.history.append({"stage": "finetune", "n": len(data.inputs), "loss": loss})
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        X = torch.tensor((inputs - self.mu) / self.sd, dtype=torch.float32)
        with torch.no_grad():
            return np.mean([n(X).numpy() for n in self.nets], 0)

    def predict_complex(self, inputs: np.ndarray, pol: str) -> np.ndarray:
        out = self.predict(inputs)
        k = self.pols.index(pol)
        return out[:, 2 * k] + 1j * out[:, 2 * k + 1]


class PriorFeatureMap:
    """Turns a target design x into prior-informed features using a CellPriorModel."""

    def __init__(self, model: CellPriorModel, schema: ParameterSchema, band: Tuple[float, float], n_freq: int = 5, pol: str = "TE"):
        cells = schema.metadata.get("cells")
        if not cells:
            raise ValueError("schema.metadata['cells'] must map cell variables (period, length, width, substrate_thickness, eps_r, tan_delta, rotation) to schema parameters")
        self.model, self.schema, self.cells = model, schema, cells
        self.freqs = np.linspace(band[0], band[1], n_freq)
        self.pol = pol
        self.unit = schema.length_scale_to_mm() * 1e-3

    def _get(self, vals: Dict[str, float], cell: Dict[str, str], key: str, default: float) -> np.ndarray:
        name = cell.get(key)
        return np.asarray(vals[name], float) if name is not None else np.full_like(vals[next(iter(vals))], default, dtype=float)

    def cell_gamma(self, X: np.ndarray, cell: Dict[str, str]) -> np.ndarray:
        X = np.atleast_2d(X)
        vals = {n: X[:, i] for i, n in enumerate(self.schema.names)}
        g = lambda k, d: self._get(vals, cell, k, d)[:, None]
        inp = cell_inputs(g("period", 1), g("length", 1), g("width", 1), g("substrate_thickness", 1), g("eps_r", 1), g("tan_delta", 0.0), g("rotation", 0.0),
                          self.freqs[None, :], self.unit)
        return self.model.predict_complex(inp, self.pol).reshape(len(X), len(self.freqs))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        gam = [self.cell_gamma(X, c) for c in self.cells]
        feats = []
        if len(gam) >= 2:
            dphi = np.angle(gam[0]) - np.angle(gam[1])
            feats += [np.cos(dphi), np.sin(dphi)]
        else:
            ph = np.angle(gam[0])
            feats += [np.cos(ph), np.sin(ph)]
        feats.append(np.column_stack([np.abs(gm).mean(1) for gm in gam]))
        return np.hstack(feats)


def target_cell_samples(schema: ParameterSchema, X: np.ndarray, responses, pols: Sequence[str] = ("TE", "TM"),
                        state_suffix: Sequence[str] = ("", "_B")) -> Optional[CellSample]:
    """Build fine-tuning samples from the target's own normal-incidence cell S11 data (if present)."""
    cells = schema.metadata.get("cells", [])
    unit = schema.length_scale_to_mm() * 1e-3
    ins, outs = [], []
    for x, resp in zip(np.atleast_2d(X), responses):
        if resp is None or resp.sparams is None:
            continue
        vals = schema.to_dict_values(x)
        for cell, suf in zip(cells, state_suffix):
            keys = [(1, 1, p + suf) for p in pols]
            if not all(k in resp.sparams.data for k in keys):
                continue
            f = resp.sparams.frequency
            gv = lambda k, d: float(vals[cell[k]]) if k in cell else d
            ins.append(cell_inputs(gv("period", 1), gv("length", 1), gv("width", 1), gv("substrate_thickness", 1), gv("eps_r", 1),
                                   gv("tan_delta", 0.0), gv("rotation", 0.0), f, unit))
            outs.append(np.column_stack([v for k in keys for v in (resp.sparams.data[k].real, resp.sparams.data[k].imag)]))
    if not ins:
        return None
    return CellSample(np.vstack(ins), np.vstack(outs), tuple(pols))
