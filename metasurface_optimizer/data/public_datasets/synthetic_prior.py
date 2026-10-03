"""SYNTHETIC prior dataset of single patch cells (stand-in when no public data is reachable).

Generated with the same analytic cell model as the synthetic oracle, over a
deliberately *different and wider* domain (2-20 GHz, broad period / substrate
ranges, both polarizations, normal and oblique incidence). Because it shares
the oracle's physics family, any transfer-learning gain measured with it is
optimistic; real experiments should replace it with public or in-house
full-wave cell data (e.g. via :class:`FSSDataBaseAdapter`).
"""
from __future__ import annotations

import numpy as np

from metasurface_optimizer.core.provenance import Fidelity
from metasurface_optimizer.data.public_datasets.base import PublicDatasetAdapter
from metasurface_optimizer.models.transfer import CellSample, cell_inputs
from metasurface_optimizer.solvers.analytic_oracle import cell_reflection


class SyntheticCellPrior(PublicDatasetAdapter):
    name = "synthetic_cell_prior"
    version = "1.0"
    fidelity = Fidelity.SYNTHETIC_ORACLE

    def __init__(self, n_cells: int = 600, n_freq: int = 40, f_range=(2e9, 20e9), seed: int = 0):
        self.n_cells, self.n_freq, self.f_range, self.seed = n_cells, n_freq, f_range, seed

    def available(self) -> bool:
        return True

    def load(self, **kw) -> CellSample:
        rng = np.random.default_rng(self.seed)
        f = np.linspace(*self.f_range, self.n_freq)
        ins, outs = [], []
        for _ in range(self.n_cells):
            P = rng.uniform(3.0, 20.0)
            L = rng.uniform(0.1, 0.97) * P
            W = rng.uniform(0.1, 0.97) * P
            h = rng.uniform(0.2, 6.0)
            er = rng.uniform(1.0, 12.0)
            td = rng.uniform(0.0, 0.03)
            rot = rng.uniform(-45, 45)
            # oblique incidence variety is folded in as data augmentation
            th = rng.choice([0.0, 0.0, 15.0, 30.0])
            # keep geometrically valid rotated patches only
            from metasurface_optimizer.geometry.geometry_generator import rotated_extent

            ex, ey = rotated_extent(L, W, rot)
            if P - max(ex, ey) < 0.05 * P:
                continue
            g = [cell_reflection(f, P * 1e-3, L * 1e-3, W * 1e-3, rot, h * 1e-3, er, td, th, pol) for pol in ("TE", "TM")]
            ins.append(cell_inputs(P, L, W, h, er, td, rot, f))
            outs.append(np.column_stack([g[0].real, g[0].imag, g[1].real, g[1].imag]))
        return CellSample(np.vstack(ins), np.vstack(outs), ("TE", "TM"))

    def compatibility_report(self):
        return {"name": self.name, "fidelity": self.fidelity.value,
                "warning": "Shares the analytic oracle's physics family: transfer gains measured with it are optimistic."}
