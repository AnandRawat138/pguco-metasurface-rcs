"""Deterministic seeding utilities."""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if installed) PyTorch; return a NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:  # pragma: no cover
        pass
    return np.random.default_rng(seed)


def child_seed(seed: int, *tags: object) -> int:
    """Derive a reproducible child seed from a parent seed and tags."""
    # hashlib (not hash()) so the result does not depend on PYTHONHASHSEED.
    tag_ints = [int(hashlib.sha256(str(t).encode()).hexdigest()[:8], 16) for t in tags]
    ss = np.random.SeedSequence([seed] + tag_ints)
    return int(ss.generate_state(1)[0])
