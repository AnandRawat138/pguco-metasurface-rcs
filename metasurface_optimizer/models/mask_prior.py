"""Mask-based prior learned from FSSDataBase (generic geometry -> spectrum representation).

FSSDataBase (``data/public_datasets/fssdatabase.py``) holds 5,607 HFSS unit cells: up to
three 500 x 500 metal masks on a fixed 10 mm cell (F4BM348 substrate, total thickness 1 or
2 mm) with co-polarised S11/S21 (TE, TM) at 101 points over 10-20 GHz. This module

1. builds a cached, split copy of the 0-deg data (:func:`build_fssdb_cache`): masks block-averaged
   to ``MASK_GRID`` x ``MASK_GRID`` (500 = 10 x 50, so every model pixel is the exact metal fraction
   of a 10 x 10 block, thresholded at 0.5; 0.2 mm per pixel, the 0.4-0.6 mm wires stay 2-3 pixels
   wide), and an 80/10/10 train/val/test split by sample id with a fixed seed;
2. trains a small CNN ``h_theta(masks, scalars) -> (Re, Im)[S11_TE, S11_TM, S21_TE, S21_TM](f)``
   (:class:`MaskPriorModel`). Convolutions use circular padding because the cell is periodic, and
   training uses random joint lateral shifts of all layers (a shifted cell is the same infinite
   array, so the 0-deg co-polarised response is unchanged). Loss (Eq. M1):

       L = mean (h - y)^2 + lambda_p * sum_{p in TE,TM} mean relu(|S11_p|^2 + |S21_p|^2 - 1)^2

   with the passivity term of ``physics/passivity.py``. Model selection uses the validation split
   only; the test split is touched once, by :func:`evaluate_split`;
3. turns a candidate design of the optimiser into prior features (:class:`MaskPriorFeatureMap`).

Scope and limits (stated, not hidden):

* FSSDataBase cells are free-standing FSS layers; most optimiser templates (e.g. the demo
  checkerboard) are ground-backed reflectors. The prior is therefore a *generic geometry
  representation*: it says how a metal pattern of that shape scatters as an FSS, not what the
  target's reflection phase is. It is not a physics transfer.
* The cell period is fixed (10 mm) and only F4BM348 (eps_r 3.48) substrates of 1-2 mm occur. A target
  cell of period P is mapped by electrical-size scaling: the cell is drawn on the model's 10 mm
  grid and the model is queried at f' = f * P / 10 mm, substrate thickness h' = h * 10 mm / P. f' is
  clamped to 10-20 GHz and h' to 1-2 mm; the clamped fraction is recorded
  (``out_of_range_fraction``). Scaling is exact only for a scale-invariant structure, which a
  dispersive, fixed-permittivity substrate is not; permittivity differences are not modelled.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from metasurface_optimizer.core.logging_utils import get_logger
from metasurface_optimizer.physics.passivity import passivity_loss_torch

log = get_logger(__name__)

MASK_GRID = 50
FSSDB_PERIOD_MM = 10.0
FSSDB_FREQ_HZ = np.linspace(10e9, 20e9, 101)
CHANNELS = ["S11_TE", "S11_TM", "S21_TE", "S21_TM"]
SCALARS = ["h_total_mm", "n_layers", "wire_width_mm"]
DB_FLOOR = -60.0


# ------------------------------------------------------------------ data
def build_fssdb_cache(root: str | Path, path: str | Path, mask_size: int = MASK_GRID, seed: int = 0,
                      fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1), max_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load FSSDataBase at 0 deg (NaN and non-passive samples dropped by the adapter), split by id, save ``.npz``."""
    from metasurface_optimizer.data.public_datasets.fssdatabase import FSSDataBaseAdapter

    ad = FSSDataBaseAdapter(root, angle=0.0, mask_size=mask_size, max_samples=max_samples)
    s = ad.load()
    ids = np.array([m["id"] for m in s.meta])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_tr, n_va = int(round(fractions[0] * len(ids))), int(round(fractions[1] * len(ids)))
    split = np.empty(len(ids), dtype="<U5")
    split[perm[:n_tr]], split[perm[n_tr:n_tr + n_va]], split[perm[n_tr + n_va:]] = "train", "val", "test"
    data = {
        "ids": ids, "masks": s.masks.astype(np.uint8), "S": s.responses.astype(np.complex64), "frequency_hz": np.asarray(s.frequency, float),
        "n_layers": np.array([m["n_layers"] for m in s.meta], np.uint8),
        "period_mm": np.array([m.get("period_mm", np.nan) for m in s.meta], np.float32),
        "wire_width_mm": np.array([m.get("wire_width_mm", np.nan) for m in s.meta], np.float32),
        "h_total_mm": np.array([np.nan if m.get("db_height_mm") is None else m["db_height_mm"] for m in s.meta], np.float32),
        "split": split, "split_seed": np.array(seed),
        "skipped": np.array(json.dumps({"nan": ad.skipped_nan, "nonpassive": ad.skipped_active, "angle": ad.skipped_angle, "failures": ad.failures})),
        "readme": np.array(
            f"FSSDataBase 0-deg cache. masks (n,3,{mask_size},{mask_size}) uint8: 500x500 masks block-averaged, metal if fraction >= 0.5, "
            "absent layers zero. S complex64 (n,101,4) channels S11_TE,S11_TM,S21_TE,S21_TM = 10^(dB/20) exp(j phase). "
            "frequency_hz (101,). h_total_mm = samples.height. split: train/val/test by sample id (seeded permutation)."),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    log.info("FSSDataBase cache: %d samples (%s) -> %s", len(ids), {k: int((split == k).sum()) for k in ("train", "val", "test")}, path)
    return data


def load_cache(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def scalar_inputs(data: Dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([data["h_total_mm"], data["n_layers"].astype(np.float32), data["wire_width_mm"]]).astype(np.float32)


# ------------------------------------------------------------------ model
class MaskPriorNet(nn.Module):
    def __init__(self, n_layers: int = 3, n_scalars: int = len(SCALARS), n_freq: int = 101, n_ch: int = 4, width: int = 32, emb: int = 64):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, padding_mode="circular"), nn.BatchNorm2d(o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1, padding_mode="circular"), nn.BatchNorm2d(o), nn.SiLU(), nn.MaxPool2d(2))

        self.conv = nn.Sequential(block(n_layers, width), block(width, 2 * width), block(2 * width, 3 * width), nn.AdaptiveAvgPool2d(3), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(27 * width + n_scalars, 256), nn.SiLU(), nn.Linear(256, emb), nn.SiLU())
        self.out = nn.Linear(emb, n_freq * n_ch * 2)
        self.shape = (n_freq, n_ch, 2)

    def embed(self, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.conv(m), s], 1))

    def forward(self, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.out(self.embed(m, s)).view(-1, *self.shape)


def _to_ri(S: np.ndarray) -> np.ndarray:
    return np.stack([S.real, S.imag], -1).astype(np.float32)


class MaskPriorModel:
    """CNN prior h_theta on FSSDataBase masks; predicts complex S on the 10-20 GHz, 101-point grid."""

    def __init__(self, width: int = 32, emb: int = 64, lambda_passive: float = 1.0, seed: int = 0):
        self.width, self.emb, self.lambda_passive, self.seed = width, emb, lambda_passive, seed
        self.net: Optional[MaskPriorNet] = None
        self.frequency = FSSDB_FREQ_HZ
        self.history: List[Dict[str, float]] = []

    def _loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = ((out - y) ** 2).mean()
        if self.lambda_passive:
            re, im = out[..., 0], out[..., 1]
            for a, b in ((0, 2), (1, 3)):  # (S11, S21) per polarization
                loss = loss + self.lambda_passive * passivity_loss_torch(re[:, :, a], im[:, :, a], re[:, :, b], im[:, :, b])
        return loss

    def fit(self, masks: np.ndarray, scalars: np.ndarray, S: np.ndarray, val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
            epochs: int = 40, batch: int = 64, lr: float = 2e-3, shift_augment: bool = True, threads: Optional[int] = None) -> "MaskPriorModel":
        """Adam + cosine schedule; keeps the epoch with the lowest validation loss (training loss if no val)."""
        prev = torch.get_num_threads()
        torch.set_num_threads(threads or max(1, min(8, (os.cpu_count() or 2) // 2)))
        try:
            torch.manual_seed(self.seed)
            self.net = MaskPriorNet(masks.shape[1], scalars.shape[1], S.shape[1], S.shape[2], self.width, self.emb)
            self.frequency = FSSDB_FREQ_HZ if S.shape[1] == len(FSSDB_FREQ_HZ) else np.linspace(10e9, 20e9, S.shape[1])
            self.s_mu, self.s_sd = scalars.mean(0), np.where(scalars.std(0) > 1e-9, scalars.std(0), 1.0)
            M = torch.tensor(masks, dtype=torch.float32)
            X = torch.tensor((scalars - self.s_mu) / self.s_sd, dtype=torch.float32)
            Y = torch.tensor(_to_ri(S))
            opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            g = torch.Generator().manual_seed(self.seed)
            best, best_state = np.inf, None
            for ep in range(epochs):
                self.net.train()
                perm = torch.randperm(len(M), generator=g)
                tot = 0.0
                for i in range(0, len(M), batch):
                    idx = perm[i:i + batch]
                    m = M[idx]
                    if shift_augment:
                        dy, dx = (int(v) for v in torch.randint(0, m.shape[-1], (2,), generator=g))
                        m = torch.roll(m, (dy, dx), (-2, -1))
                    opt.zero_grad()
                    loss = self._loss(self.net(m, X[idx]), Y[idx])
                    loss.backward()
                    opt.step()
                    tot += float(loss.detach()) * len(idx)
                sched.step()
                rec = {"epoch": ep, "train_loss": tot / len(M)}
                score = rec["train_loss"]
                if val is not None:
                    rec["val_loss"] = score = float(self._loss(torch.tensor(_to_ri(self.predict(val[0], val[1]))), torch.tensor(_to_ri(val[2]))))
                self.history.append(rec)
                if score < best:
                    best, best_state = score, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            self.net.load_state_dict(best_state)
            self.net.eval()
            log.info("mask prior: %d epochs, best %s loss %.5f", epochs, "val" if val is not None else "train", best)
        finally:
            torch.set_num_threads(prev)
        return self

    def _inputs(self, masks: np.ndarray, scalars: np.ndarray):
        return torch.tensor(np.asarray(masks), dtype=torch.float32), torch.tensor((np.asarray(scalars, np.float32) - self.s_mu) / self.s_sd, dtype=torch.float32)

    def predict(self, masks: np.ndarray, scalars: np.ndarray, batch: int = 512) -> np.ndarray:
        """Complex S (n, F, 4), channels ``CHANNELS``."""
        self.net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(masks), batch):
                o = self.net(*self._inputs(masks[i:i + batch], scalars[i:i + batch])).numpy()
                outs.append(o[..., 0] + 1j * o[..., 1])
        return np.concatenate(outs) if outs else np.zeros((0, len(self.frequency), 4), complex)

    def embed(self, masks: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return self.net.embed(*self._inputs(masks, scalars)).numpy()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "s_mu": self.s_mu, "s_sd": self.s_sd, "frequency": self.frequency, "history": self.history,
                    "cfg": {"width": self.width, "emb": self.emb, "lambda_passive": self.lambda_passive, "seed": self.seed},
                    "shape": {"layers": self.net.conv[0][0].in_channels, "scalars": self.net.head[0].in_features - 27 * self.width,
                              "freq": self.net.shape[0], "ch": self.net.shape[1]}}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MaskPriorModel":
        d = torch.load(path, weights_only=False)
        m = cls(**d["cfg"])
        sh = d["shape"]
        m.net = MaskPriorNet(sh["layers"], sh["scalars"], sh["freq"], sh["ch"], m.width, m.emb)
        m.net.load_state_dict(d["state"])
        m.net.eval()
        m.s_mu, m.s_sd, m.frequency, m.history = d["s_mu"], d["s_sd"], d["frequency"], d["history"]
        return m


# ------------------------------------------------------------------ evaluation
def spectrum_errors(pred: np.ndarray, true: np.ndarray, phase_floor_db: float = -20.0) -> Dict[str, Dict[str, float]]:
    """Per channel: |S| dB MAE (both clipped at DB_FLOOR) and circular phase MAE (deg) where true |S| > phase_floor_db."""
    out = {}
    with np.errstate(divide="ignore"):
        dp = np.maximum(20 * np.log10(np.abs(pred)), DB_FLOOR)
        dt = np.maximum(20 * np.log10(np.abs(true)), DB_FLOOR)
    dphi = np.degrees(np.abs(np.angle(pred * np.conj(true))))
    for k, c in enumerate(CHANNELS):
        sel = dt[..., k] > phase_floor_db
        out[c] = {"mag_db_mae": float(np.mean(np.abs(dp[..., k] - dt[..., k]))),
                  "phase_deg_mae": float(np.mean(dphi[..., k][sel])) if sel.any() else None,
                  "complex_rmse": float(np.sqrt(np.mean(np.abs(pred[..., k] - true[..., k]) ** 2))),
                  "passband_agreement": float(np.mean((dp[..., k] > -5) == (dt[..., k] > -5)))}
    return out


def fill_features(data: Dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([data["masks"].mean(axis=(2, 3)), data["h_total_mm"]]).astype(float)


def evaluate_split(model: MaskPriorModel, data: Dict[str, np.ndarray], split: str = "test") -> Dict[str, object]:
    """Model vs. two baselines on one split: training-mean complex spectrum, and nearest training neighbour by per-layer fill factor + thickness."""
    tr, te = data["split"] == "train", data["split"] == split
    true = data["S"][te]
    pred = model.predict(data["masks"][te], scalar_inputs(data)[te])
    mean = np.broadcast_to(data["S"][tr].mean(0), true.shape)
    F = fill_features(data)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-12
    Ft, Fe = (F[tr] - mu) / sd, (F[te] - mu) / sd
    nn_idx = np.argmin(((Fe[:, None, :] - Ft[None, :, :]) ** 2).sum(-1), 1)
    nn_pred = data["S"][tr][nn_idx]
    return {"split": split, "n": int(te.sum()), "n_train": int(tr.sum()),
            "model": spectrum_errors(pred, true), "baseline_train_mean": spectrum_errors(mean, true), "baseline_nn_fill_factor": spectrum_errors(nn_pred, true)}


# ------------------------------------------------------------------ optimiser features
class MaskPriorFeatureMap:
    """Prior features for a target design from the FSSDataBase mask prior (Eq. F1 block p(x)).

    Every cell in ``schema.metadata['cells']`` (period, length, width, rotation, substrate_thickness) is
    drawn as a centred (rotated) rectangle on the model grid, rasterised with
    :func:`~metasurface_optimizer.geometry.geometry_generator.rasterize_geometry` over the scaled
    10 mm cell, and evaluated at ``n_freq`` band frequencies with electrical-size scaling (module
    docstring). Features follow :class:`~metasurface_optimizer.models.transfer.PriorFeatureMap`
    for ``channel`` (default S11_TE): [cos dphi_AB, sin dphi_AB](f_k) for two cells (else cos/sin phi),
    plus mean |S| per cell. ``mode='embedding'`` instead returns the model's penultimate embedding
    (mean over cells, first ``n_embed`` dims), a pure geometry representation."""

    def __init__(self, model: MaskPriorModel, schema, band: Tuple[float, float], n_freq: int = 5, channel: str = "S11_TE",
                 mode: str = "spectrum", n_embed: int = 8, grid: int = MASK_GRID):
        cells = schema.metadata.get("cells")
        if not cells:
            raise ValueError("schema.metadata['cells'] must map cell variables (period, length, width, rotation, substrate_thickness) to schema parameters")
        if mode not in ("spectrum", "embedding"):
            raise ValueError("mode must be 'spectrum' or 'embedding'")
        self.model, self.schema, self.cells, self.mode, self.n_embed, self.grid = model, schema, cells, mode, n_embed, grid
        self.freqs = np.linspace(band[0], band[1], n_freq)
        self.ch = CHANNELS.index(channel)
        self.mm = schema.length_scale_to_mm()
        self.wire = float(model.s_mu[SCALARS.index("wire_width_mm")])
        self.out_of_range_fraction = 0.0

    def cell_masks(self, X: np.ndarray, cell: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """(n, 3, grid, grid) masks on the scaled 10 mm cell and scale factors 10 mm / P."""
        from shapely.geometry import MultiPolygon, box

        from metasurface_optimizer.geometry.geometry_generator import CellGeometry, Layer, centered_rect, rasterize_geometry

        X = np.atleast_2d(X)
        g = lambda k, d: X[:, self.schema.index(cell[k])] if k in cell else np.full(len(X), d, float)
        P, L, W, rot = g("period", FSSDB_PERIOD_MM) * self.mm, g("length", 0.0) * self.mm, g("width", 0.0) * self.mm, g("rotation", 0.0)
        k = FSSDB_PERIOD_MM / P
        out = np.zeros((len(X), 3, self.grid, self.grid), np.float32)
        half = FSSDB_PERIOD_MM / 2
        for i in range(len(X)):
            poly = centered_rect(L[i] * k[i], W[i] * k[i], angle_deg=rot[i]).intersection(box(-half, -half, half, half))
            top = MultiPolygon([poly]) if poly.geom_type == "Polygon" and not poly.is_empty else MultiPolygon()
            geom = CellGeometry(FSSDB_PERIOD_MM, FSSDB_PERIOD_MM, [Layer("top_metal", 0.0, 0.0, "metal", polygons=top)], "fssdb_scaled_cell")
            out[i, 0] = rasterize_geometry(geom, self.grid)
        return out, k

    def _cell_response(self, X: np.ndarray, cell: Dict[str, str]):
        X = np.atleast_2d(X)
        masks, k = self.cell_masks(X, cell)
        h = (X[:, self.schema.index(cell["substrate_thickness"])] * self.mm if "substrate_thickness" in cell else np.full(len(X), 1.0)) * k
        s = np.column_stack([np.clip(h, 1.0, 2.0), np.ones(len(X)), np.full(len(X), self.wire)]).astype(np.float32)
        if self.mode == "embedding":
            return self.model.embed(masks, s)
        S = self.model.predict(masks, s)[:, :, self.ch]
        fq = self.freqs[None, :] / k[:, None]  # f' = f * P / 10 mm
        f0, f1 = self.model.frequency[0], self.model.frequency[-1]
        self.out_of_range_fraction = float(np.mean((fq < f0) | (fq > f1)))
        fq = np.clip(fq, f0, f1)
        return np.stack([np.interp(fq[i], self.model.frequency, S[i].real) + 1j * np.interp(fq[i], self.model.frequency, S[i].imag) for i in range(len(X))])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        resp = [self._cell_response(X, c) for c in self.cells]
        if self.mode == "embedding":
            return np.mean(resp, 0)[:, : self.n_embed]
        feats = []
        if len(resp) >= 2:
            dphi = np.angle(resp[0]) - np.angle(resp[1])
            feats += [np.cos(dphi), np.sin(dphi)]
        else:
            ph = np.angle(resp[0])
            feats += [np.cos(ph), np.sin(ph)]
        feats.append(np.column_stack([np.abs(r).mean(1) for r in resp]))
        return np.hstack(feats)


def load_or_train(t: Dict[str, object], root: Path) -> MaskPriorModel:
    """Model for ``transfer.prior: fssdatabase_mask_prior``. Uses ``model_path`` if it exists, else trains from
    ``cache_path`` (built from ``dataset_root`` or $FSSDATABASE_ROOT if missing) and saves to ``model_path``."""
    rel = lambda p: Path(p) if Path(p).is_absolute() else root / p
    mp = rel(t.get("model_path", "outputs/models/fssdb_mask_prior.pt"))
    if mp.exists():
        return MaskPriorModel.load(mp)
    cp = rel(t.get("cache_path", "outputs/cache/fssdb_mask50.npz"))
    if not cp.exists():
        ds = t.get("dataset_root") or os.environ.get("FSSDATABASE_ROOT")
        if not ds:
            raise FileNotFoundError(f"{mp} and {cp} are missing; set transfer.dataset_root or FSSDATABASE_ROOT to build them")
        build_fssdb_cache(ds, cp, seed=int(t.get("split_seed", 0)))
    data = load_cache(cp)
    tr, va = data["split"] == "train", data["split"] == "val"
    sc = scalar_inputs(data)
    model = MaskPriorModel(seed=int(t.get("seed", 0))).fit(data["masks"][tr], sc[tr], data["S"][tr], (data["masks"][va], sc[va], data["S"][va]),
                                                          epochs=int(t.get("epochs", 40)))
    model.save(mp)
    return model
