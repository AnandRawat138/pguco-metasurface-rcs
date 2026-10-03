"""Adapter for FSSDataBase (Hugging Face: ``kkking789/FSSDataBase``; arXiv:2609.30787).

Layout (checked against a partial local copy: the complete ``database/DataBase.db``
plus 614 of the 5,607 sample folders):

* ``database/DataBase.db`` (SQLite)
    - ``samples(id TEXT, sample_path TEXT, size REAL, height REAL)``; ``sample_path``
      is relative to the dataset root (``datasets/<id>``). IDs stay strings: legacy
      IDs look like floats (``1786866648.2374983``) and would lose digits as numbers.
    - ``responses(id, direct, angle, p000..p299)``: TE-only +1/-1 labels on the padded
      grid ``linspace(1, 30, 300)`` GHz, 0 outside the simulated band (in practice p093-p195,
      10.02-19.91 GHz; these labels agree with the raw S21_TE data at 97-100% of points); ``direct`` 0 = S11,
      1 = S21. ``responses_angle``: matching TE phase (float16-rounded, may be NULL).
* ``datasets/<id>/``
    - ``raw_result.csv`` (no header, 10 columns): angle_deg, freq_GHz, S11_TE_dB, S11_TM_dB,
      S21_TE_dB, S21_TM_dB, then the four phases (deg) in the same order. One block of
      rows per incidence angle (0 deg, optionally 30 deg); 10-20 GHz in 0.1 GHz steps.
      NaN appears as literal ``nan`` (in the local copy: only in 30-deg rows).
    - ``label_result.csv``: same shape, +1 where |S| > -5 dB else -1 (also -1 for NaN, so
      it is *not* a validity flag; this adapter recomputes nothing from it).
    - ``mask_<n>.csv``: 500 x 500 0/1 grid, one per metal layer (1-3 layers).
    - ``data.json``: ``unit = [wire_width_mm, size]`` with cell period ``2 * size`` mm,
      ``sub`` = substrate layers {material, h}, a material catalogue (not usage), and HFSS
      ``build`` expressions (never evaluated here).

FSSDataBase cells are generic HFSS FSS results. They are used only as *generic
electromagnetic prior knowledge* (mask -> spectrum representation learning), never as
data about the user's target design.
"""
from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from metasurface_optimizer.core.logging_utils import get_logger
from metasurface_optimizer.data.public_datasets.base import MaskSample, PublicDatasetAdapter

log = get_logger(__name__)

RAW_COLUMNS = ["angle_deg", "freq_GHz", "S11_TE_dB", "S11_TM_dB", "S21_TE_dB", "S21_TM_dB",
               "S11_TE_phase", "S11_TM_phase", "S21_TE_phase", "S21_TM_phase"]
CHANNELS = ["S11_TE", "S11_TM", "S21_TE", "S21_TM"]
DB_GRID_GHZ = np.linspace(1.0, 30.0, 300)


def _block_downsample(mask: np.ndarray, size: int) -> np.ndarray:
    """Area-preserving downsample of a binary mask to ``size x size`` (metal fraction per block >= 0.5)."""
    H, W = mask.shape
    if size >= H:
        return mask.astype(np.uint8)
    if H % size or W % size:
        from metasurface_optimizer.geometry.mask_encoder import resample_mask

        return resample_mask(mask, size).astype(np.uint8)
    return (mask.reshape(size, H // size, size, W // size).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)


class FSSDataBaseAdapter(PublicDatasetAdapter):
    name = "FSSDataBase"

    def __init__(self, root: str | Path, angle: float = 0.0, nan_policy: str = "drop_sample", max_layers: int = 3,
                 mask_size: Optional[int] = 100, max_samples: Optional[int] = None, ids: Optional[Sequence[str]] = None):
        """``angle``: incidence angle block to read (0 or 30 deg).
        ``nan_policy``: ``drop_sample`` (skip samples with any NaN at that angle; recorded) or ``keep``.
        ``mask_size``: downsample 500 x 500 masks to this grid (None keeps full resolution).
        Masks are returned as (n, max_layers, H, W); absent layers are all-zero and
        ``meta[i]['n_layers']`` says how many are real."""
        self.root = Path(root)
        self.angle = float(angle)
        if nan_policy not in ("drop_sample", "keep"):
            raise ValueError("nan_policy must be 'drop_sample' or 'keep'")
        self.nan_policy = nan_policy
        self.max_layers = max_layers
        self.mask_size = mask_size
        self.max_samples = max_samples
        self.ids = list(ids) if ids is not None else None
        self.failures: List[Dict[str, str]] = []
        self.skipped_nan: List[str] = []

    # ------------------------------------------------------------ discovery
    @property
    def db_path(self) -> Path:
        return self.root / "database" / "DataBase.db"

    def available(self) -> bool:
        return self.db_path.exists() or (self.root / "datasets").is_dir()

    def sample_table(self) -> List[Dict[str, object]]:
        """Rows of ``samples`` (ids kept as TEXT), or a directory scan when the database is absent."""
        if self.db_path.exists():
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            try:
                rows = con.execute("SELECT CAST(id AS TEXT), sample_path, size, height FROM samples ORDER BY id").fetchall()
            finally:
                con.close()
            return [{"id": r[0], "path": self.root / Path(str(r[1]).replace("\\", "/")), "size": r[2], "height": r[3]} for r in rows]
        return [{"id": d.name, "path": d, "size": None, "height": None}
                for d in sorted((self.root / "datasets").iterdir()) if d.is_dir() and d.name != "cache"]

    # ------------------------------------------------------------ per-sample readers
    @staticmethod
    def read_raw(path: Path) -> np.ndarray:
        import pandas as pd

        arr = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if arr.ndim != 2 or arr.shape[1] != len(RAW_COLUMNS):
            raise ValueError(f"{path.name}: expected {len(RAW_COLUMNS)} columns, got {arr.shape}")
        return arr

    def read_spectrum(self, path: Path):
        """Complex S-parameters at ``self.angle``: (frequency_Hz (F,), S (F, 4)) with channels ``CHANNELS``."""
        arr = self.read_raw(path)
        rows = arr[np.isclose(arr[:, 0], self.angle, atol=1e-6)]
        if not len(rows):
            raise ValueError(f"no rows at {self.angle} deg (angles present: {sorted(set(np.round(arr[:, 0], 6)))})")
        rows = rows[np.argsort(rows[:, 1])]
        mag = 10.0 ** (rows[:, 2:6] / 20.0)
        S = mag * np.exp(1j * np.radians(rows[:, 6:10]))
        return rows[:, 1] * 1e9, S

    def read_masks(self, d: Path) -> np.ndarray:
        import pandas as pd

        files = sorted(d.glob("mask_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
        if not files:
            raise FileNotFoundError("no mask_<n>.csv")
        if len(files) > self.max_layers:
            raise ValueError(f"{len(files)} layers > max_layers={self.max_layers}")
        layers = []
        for f in files:
            m = (np.nan_to_num(pd.read_csv(f, header=None).to_numpy(float)) > 0.5).astype(np.uint8)
            layers.append(_block_downsample(m, self.mask_size) if self.mask_size else m)
        return np.stack(layers)

    @staticmethod
    def read_meta(d: Path) -> Dict[str, object]:
        p = d / "data.json"
        if not p.exists():
            return {}
        js = json.loads(p.read_text(encoding="utf-8"), parse_float=Decimal)  # keeps float-looking ids exact
        out: Dict[str, object] = {}
        unit = js.get("unit")
        if isinstance(unit, list) and len(unit) >= 2:
            out["wire_width_mm"] = float(unit[0])
            out["period_mm"] = 2.0 * float(unit[1])  # generator uses d = 2 * size
        sub = js.get("sub")
        if isinstance(sub, dict):
            out["substrate"] = [{"name": k, "material": v.get("material"), "h_mm": float(v["h"]) if "h" in v else None}
                                for k, v in sub.items() if isinstance(v, dict)]
        if "idx" in js:
            out["idx"] = str(js["idx"])
        return out

    # ------------------------------------------------------------ dataset
    def load(self, **kw) -> MaskSample:
        rows = self.sample_table()
        if self.ids is not None:
            want = set(self.ids)
            rows = [r for r in rows if r["id"] in want]
        masks, resps, metas = [], [], []
        freq = None
        for r in rows:
            if self.max_samples is not None and len(masks) >= self.max_samples:
                break
            d = Path(r["path"])
            try:
                if not (d / "raw_result.csv").exists():
                    raise FileNotFoundError("raw_result.csv missing (sample not downloaded or download interrupted)")
                f, S = self.read_spectrum(d / "raw_result.csv")
                if np.isnan(S).any():
                    if self.nan_policy == "drop_sample":
                        self.skipped_nan.append(str(r["id"]))
                        continue
                if freq is None:
                    freq = f
                elif len(f) != len(freq) or not np.allclose(f, freq, rtol=1e-9, atol=1.0):
                    raise ValueError("frequency grid differs from the first sample")
                m = self.read_masks(d)
                pad = np.zeros((self.max_layers,) + m.shape[1:], np.uint8)
                pad[: len(m)] = m
                meta = {"id": str(r["id"]), "n_layers": int(len(m)), "db_size": r["size"], "db_height_mm": r["height"]}
                meta.update(self.read_meta(d))
                masks.append(pad); resps.append(S); metas.append(meta)
            except Exception as exc:
                self.failures.append({"id": str(r["id"]), "error": f"{type(exc).__name__}: {exc}"})
        if not masks:
            raise RuntimeError(f"No FSSDataBase samples parsed under {self.root}; failures: {self.failures[:3]}")
        log.info("FSSDataBase: %d samples loaded, %d skipped for NaN at %g deg, %d unavailable/failed",
                 len(masks), len(self.skipped_nan), self.angle, len(self.failures))
        return MaskSample(np.stack(masks), freq, np.stack(resps),
                          f"S_complex[{','.join(CHANNELS)}] at theta={self.angle:g} deg (HFSS, generic FSS cells)", metas)

    def database_labels(self, direct: int = 1, angle: float = 0.0) -> Dict[str, object]:
        """TE pass/stop labels from ``responses`` for every sample in the database (including ones whose folders
        were not downloaded). Returns ids, the in-band frequency grid (Hz) and labels (n, F) in {-1, +1}."""
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            cols = [c[1] for c in con.execute("PRAGMA table_info(responses)") if c[1].startswith("p")]
            q = f"SELECT CAST(id AS TEXT), {','.join(cols)} FROM responses WHERE direct = ? AND abs(angle - ?) < 1e-6 ORDER BY id"
            rows = con.execute(q, (direct, angle)).fetchall()
        finally:
            con.close()
        ids = [r[0] for r in rows]
        L = np.array([r[1:] for r in rows], float) if rows else np.zeros((0, len(cols)))
        used = np.any(L != 0, axis=0)  # 0 = sentinel outside the simulated band
        return {"ids": ids, "frequency": DB_GRID_GHZ[: len(cols)][used] * 1e9, "labels": L[:, used],
                "columns": [c for c, u in zip(cols, used) if u], "kind": f"{'S21' if direct else 'S11'}_TE label (+1: |S| > -5 dB)"}

    def compatibility_report(self) -> Dict[str, object]:
        return {
            "name": self.name, "used_for": "generic EM prior (mask -> spectrum representation learning)",
            "not_used_for": "target-domain training or evaluation",
            "verified_against_files": "format checked against a partial local copy (614/5607 samples + full DataBase.db)",
            "parse_failures": len(self.failures), "skipped_for_nan": len(self.skipped_nan),
        }
