import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metasurface_optimizer.geometry.parameter_schema import ParameterSchema  # noqa: E402


@pytest.fixture
def schema():
    return ParameterSchema.load(ROOT / "configs/schemas/checkerboard_demo.yaml")


@pytest.fixture
def oracle():
    from metasurface_optimizer.solvers.analytic_oracle import AnalyticCheckerboardOracle

    return AnalyticCheckerboardOracle({"n_freq": 21, "theta_inc": [0.0, 30.0], "theta_s_step": 6.0, "phi_s_step": 30.0})


@pytest.fixture
def objectives():
    from metasurface_optimizer.optimization.objectives import ObjectiveSet

    band = (8e9, 12e9)
    return ObjectiveSet.from_config([
        dict(name="specular", kind="specular_rcs", band=band, incidence=[0.0], scale="reduction_db"),
        dict(name="diffusion", kind="angular_entropy", band=band, incidence=[0.0], weight=0.5),
    ])


@pytest.fixture
def rng():
    return np.random.default_rng(0)
