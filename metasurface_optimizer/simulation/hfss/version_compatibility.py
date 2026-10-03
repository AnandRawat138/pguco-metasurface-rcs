"""PyAEDT version-compatibility layer (re-export).

The implementation lives in :mod:`hfss_worker.runner.version_compatibility`
because it is inlined into every ``run_hfss.py`` and must run on the HFSS PC
with nothing but PyAEDT installed. Version differences are isolated there.
"""
from hfss_worker.runner.version_compatibility import *  # noqa: F401,F403
from hfss_worker.runner.version_compatibility import COMPAT_LOG, import_pyaedt, normalize_aedt_version, open_hfss, versions  # noqa: F401
