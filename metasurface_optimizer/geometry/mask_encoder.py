"""Binary mask utilities: resampling to a fixed grid and compact geometric descriptors.

Descriptors (all dimensionless): fill factor, normalised second moments,
number of components and holes, perimeter-to-area ratio and the magnitudes of
the lowest 2-D Fourier coefficients (periodic-structure friendly). They give
mask-based datasets (e.g. FSSDataBase) a fixed-length representation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def resample_mask(mask: np.ndarray, size: int) -> np.ndarray:
    m = np.asarray(mask, float)
    zoom = (size / m.shape[0], size / m.shape[1])
    return ndimage.zoom(m, zoom, order=1) > 0.5


def mask_descriptors(mask: np.ndarray, n_fourier: int = 4) -> np.ndarray:
    m = np.asarray(mask, bool)
    H, W = m.shape
    ff = m.mean()
    ys, xs = np.nonzero(m)
    if len(xs):
        cx, cy = xs.mean() / W, ys.mean() / H
        mxx, myy = ((xs / W - cx) ** 2).mean(), ((ys / H - cy) ** 2).mean()
        mxy = ((xs / W - cx) * (ys / H - cy)).mean()
    else:
        mxx = myy = mxy = 0.0
    n_comp = ndimage.label(m)[1]
    n_holes = ndimage.label(~m)[1] - 1
    perim = (m ^ ndimage.binary_erosion(m)).sum() / max(m.sum(), 1)
    F = np.abs(np.fft.fft2(m.astype(float))) / m.size
    four = F[:n_fourier, :n_fourier].ravel()[1:]
    return np.concatenate([[ff, mxx, myy, mxy, n_comp, max(n_holes, 0), perim], four])
