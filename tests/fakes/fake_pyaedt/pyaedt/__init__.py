"""FAKE legacy-namespace PyAEDT (tests only; see tests/fakes/fake_aedt.py). Not a solver."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fake_aedt import make_module  # noqa: E402

_m = make_module(legacy=True)
Hfss = _m.Hfss
__version__ = _m.__version__
