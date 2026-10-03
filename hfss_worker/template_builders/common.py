"""Version-tolerant PyAEDT calls used by the template builders (AEDT 2024 R2 / PyAEDT 0.8-0.12 spellings).

The builders run once, by hand, on the HFSS PC. They are not used by the worker. Every call that differs
between PyAEDT releases goes through :func:`call`, which passes each keyword under the spelling the
installed function accepts. Anything the builder cannot set reliably is reported in ``NOTES`` and printed,
and the 12-step verification (``worker.py --verify-template``) is the gate that decides whether the
template may be used.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hfss_worker.runner.version_compatibility import _kw, import_pyaedt, open_hfss  # noqa: E402

NOTES = []


def note(msg):
    NOTES.append(msg)
    print("NOTE:", msg)


def call(func, *args, **named):
    """``named``: canonical keyword -> (value, [alternative spellings])."""
    kw = _kw(func, [((k,) + tuple(alts), v) for k, (v, alts) in named.items()])
    return func(*args, **kw)


def open_new(project, design, version="2024.2", non_graphical=False):
    if os.path.exists(project):
        raise SystemExit("%s exists; delete it or choose another --out (templates are never overwritten)" % project)
    mod, ns, ver = import_pyaedt()
    print("PyAEDT %s %s" % (ns, ver))
    app = open_hfss(mod, project, design, version, non_graphical=non_graphical, new_desktop=True, close_on_exit=False)
    try:
        app.solution_type = "Modal"
    except Exception as e:
        note("could not set solution type Modal: %s" % e)
    app.modeler.model_units = "mm"
    return app


def set_vars(app, values):
    for k, v in values.items():
        app[k] = v


def set_props(obj, props, label):
    """Set props on an AEDT object (excitation, setup, far-field sphere) and read them back."""
    for k, v in props.items():
        obj.props[k] = v
    try:
        obj.update()
    except Exception as e:
        note("%s: update() failed: %s" % (label, e))
    bad = {k: (v, obj.props.get(k)) for k, v in props.items() if str(obj.props.get(k)).replace(" ", "") != str(v).replace(" ", "")}
    if bad:
        note("%s: properties read back differently (requested, got): %s" % (label, bad))


def perfect_e(app, sheets, name):
    return call(app.assign_perfecte_to_sheets, sheets, name=(name, ["sourcename"]))


def setup_and_sweep(app, freq_ghz, f_start, f_stop, n, sweep_type, max_passes, delta_key, delta):
    setup = call(app.create_setup, name=("Setup1", ["setupname"]))
    set_props(setup, {"Frequency": "%gGHz" % freq_ghz, "MaximumPasses": max_passes, delta_key: delta}, "Setup1")
    call(app.create_linear_count_sweep, setup=("Setup1", ["setupname"]), units=("GHz", ["unit"]),
         start_frequency=(f_start, ["freqstart"]), stop_frequency=(f_stop, ["freqstop"]), num_of_freq_points=(n, []),
         name=("Sweep1", ["sweepname"]), sweep_type=(sweep_type, []), save_fields=(False, []))
    return setup


def finish(app, project):
    try:
        ok = app.validate_simple()
        print("validate_simple:", ok)
        if not ok:
            note("AEDT design validation reported problems; open the project and check the message window")
    except Exception as e:
        note("validate_simple failed: %s" % e)
    app.save_project()
    print("saved", project)
    if NOTES:
        print("\n%d note(s) - check them in the AEDT GUI before verifying:" % len(NOTES))
        for n in NOTES:
            print("  -", n)
    try:
        app.release_desktop(close_projects=True, close_desktop=True)
    except TypeError:
        app.release_desktop()
