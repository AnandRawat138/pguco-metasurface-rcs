"""High-fidelity simulation automation (two-PC HFSS/PyAEDT workflow).

PC-1 (this framework, no HFSS) turns candidates into solver-independent job
packages; PC-2 (HFSS 2024 R2 + PyAEDT) runs them with ``hfss_worker``; the
results come back as validated result packages. The optimizer never imports
PyAEDT: everything solver-specific sits behind :class:`SolverAdapter`.

Modes: DRY_RUN (packages only), MOCK (software test, never physics), HFSS.
"""
from hfss_worker.protocol import JobState  # noqa: F401
