"""Build HFSS Template B for Song et al. (2016): finite 30 x 30 ring+disc array (or the PEC plate) for backscatter RCS.

    python hfss_worker/template_builders/song2016_array.py --layout baseline_paper.csv --out D:/hfss_templates/song2016_array.aedt
    python hfss_worker/template_builders/song2016_array.py --plate --out D:/hfss_templates/song2016_plate.aedt

Array model (mm): substrate FR4_epoxy n*p x n*p x h on a PEC ground sheet; per cell (row i from +y,
column j from -x) a PEC ring (outer r1_ii_jj, inner k2*r1_ii_jj) and disc (k3*r1_ii_jj). One design variable
per cell, initialised from the layout CSV (outputs/song2016/layouts/*.csv). Plate model: one PEC sheet
plate_L x plate_L. Both: vacuum box padded by ``pad`` with radiation boundary Rad1; incident plane wave
IncPWave_TE travelling with theta = 180deg - theta_inc, phi = 0 (arrives from (theta_inc, 180 deg)), E along
phi-hat (= y, TE); infinite sphere 'Backscatter' at (theta_inc, 180 deg); Setup1 at 12 GHz, Sweep1 discrete.

The worker sets theta_inc, the r1_* variables and the sweep per job. Large model: expect hours per job
on a workstation; raise max_runtime_minutes in hfss_worker/config.yaml (e.g. 1440). FE-BI instead of the
radiation boundary shrinks the air volume; if you switch, keep the boundary name Rad1 or edit the registration.
Check in the GUI before verifying: the plane-wave angles/polarization and the sphere angles read back as
expressions of theta_inc (the builder prints a NOTE if not), and that 'RCSTotal' is offered in Far Fields.
"""
import argparse
import csv

from common import call, finish, note, open_new, perfect_e, set_props, set_vars, setup_and_sweep


def read_layout(path, n):
    r1 = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            r1[(int(row["row"]), int(row["col"]))] = float(row["r1_mm"])
    if len(r1) != n * n:
        raise SystemExit("%s: expected %d cells, found %d" % (path, n * n, len(r1)))
    return r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", help="layout CSV (row,col,...,r1_mm); required unless --plate")
    ap.add_argument("--plate", action="store_true", help="build the PEC plate reference instead")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--version", default="2024.2")
    ap.add_argument("--non-graphical", action="store_true")
    a = ap.parse_args()
    n = a.n
    app = open_new(a.out, "Plate" if a.plate else "Array", a.version, a.non_graphical)
    set_vars(app, {"p": "10mm", "h": "3mm", "k2": "0.92", "k3": "0.8", "pad": "12mm", "theta_inc": "0deg", "L": "%d*p" % n})
    m = app.modeler
    if a.plate:
        set_vars(app, {"plate_L": "%dmm" % (10 * n)})
        call(m.create_rectangle, "XY", ["-plate_L/2", "-plate_L/2", "0mm"], ["plate_L", "plate_L"], name=("Plate", []))
        perfect_e(app, ["Plate"], "PEC_plate")
        box = (["-plate_L/2-pad", "-plate_L/2-pad", "-pad"], ["plate_L+2*pad", "plate_L+2*pad", "2*pad"])
    else:
        if not a.layout:
            raise SystemExit("--layout is required for the array")
        r1 = read_layout(a.layout, n)
        set_vars(app, {"r1_%02d_%02d" % ij: "%.4fmm" % v for ij, v in sorted(r1.items())})
        call(m.create_box, ["-L/2", "-L/2", "0mm"], ["L", "L", "h"], name=("Substrate", []), material=("FR4_epoxy", ["matname"]))
        call(m.create_rectangle, "XY", ["-L/2", "-L/2", "0mm"], ["L", "L"], name=("GroundSheet", []))
        metal = []
        for i in range(n):
            for j in range(n):
                v = "r1_%02d_%02d" % (i, j)
                c = ["(%g)*p" % (j - (n - 1) / 2), "(%g)*p" % ((n - 1) / 2 - i), "h"]
                ring = call(m.create_circle, "XY", c, v, name=("Ring_%02d_%02d" % (i, j), []))
                hole = call(m.create_circle, "XY", c, "k2*" + v, name=("Hole_%02d_%02d" % (i, j), []))
                m.subtract(ring, hole, False)
                call(m.create_circle, "XY", c, "k3*" + v, name=("Disc_%02d_%02d" % (i, j), []))
                metal += ["Ring_%02d_%02d" % (i, j), "Disc_%02d_%02d" % (i, j)]
            print("row %d/%d drawn" % (i + 1, n), flush=True)
        perfect_e(app, metal, "Metal")
        perfect_e(app, ["GroundSheet"], "Ground")
        box = (["-L/2-pad", "-L/2-pad", "-pad"], ["L+2*pad", "L+2*pad", "h+2*pad"])
    air = call(m.create_box, box[0], box[1], name=("AirBox", []), material=("vacuum", ["matname"]))
    call(app.assign_radiation_boundary_to_objects, air, name=("Rad1", ["boundary_name"]))
    # incident plane wave (created with plain numbers, then switched to theta_inc expressions)
    if hasattr(app, "plane_wave"):
        pw = call(app.plane_wave, vector_format=("Spherical", []), polarization=("Horizontal", []),
                  propagation_vector=([["0deg", "0deg", 1], ["180deg", "180deg", 1]], []), name=("IncPWave_TE", []))
    else:
        pw = call(app.create_incident_plane_wave, "IncPWave_TE")
    set_props(pw, {"ThetaStart": "180deg-theta_inc", "ThetaStop": "180deg-theta_inc", "ThetaPoints": 1,
                   "PhiStart": "0deg", "PhiStop": "0deg", "PhiPoints": 1, "EoPhi": "1", "EoTheta": "0"}, "IncPWave_TE")
    sph = call(app.insert_infinite_sphere, definition=("Theta-Phi", []), x_start=(0, []), x_stop=(0, []), x_step=(1, []),
               y_start=(180, []), y_stop=(180, []), y_step=(1, []), name=("Backscatter", []))
    set_props(sph, {"ThetaStart": "theta_inc", "ThetaStop": "theta_inc", "PhiStart": "180deg", "PhiStop": "180deg"}, "Backscatter")
    setup_and_sweep(app, 12, 7, 12, 11, "Discrete", 10, "MaximumDeltaE", 0.05)
    if not a.plate:
        note("the r1_* variables hold the layout from %s; the worker overwrites all 900 per job" % a.layout)
    finish(app, a.out)


if __name__ == "__main__":
    main()
