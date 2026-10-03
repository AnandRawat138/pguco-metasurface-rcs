"""Build HFSS Template A for Song et al. (2016): ring+disc unit cell with a Floquet port (run once on the HFSS PC).

    python hfss_worker/template_builders/song2016_unit_cell.py --out D:/hfss_templates/song2016_unit_cell.aedt

Model (mm): 10 mm cell, FR4_epoxy (eps_r 4.4, tan d 0.02) 3 mm thick on a PEC ground sheet; on top a PEC
ring (outer R1, inner k2*R1) and a PEC disc (k3*R1). Air box to h + air_h with lattice pairs on the side
walls and FloquetPort1 (2 modes) on the top face, de-embedded by air_h to the substrate surface (the
reflection-phase reference plane of the paper's Fig. 2). Setup1 at 13 GHz (adaptive), Sweep1 interpolating
6-13 GHz. Variables: p, h, R1, k2, k3, air_h.

Then verify:  python hfss_worker/worker.py --verify-template hfss_jobs/song2016/templates/song2016_unit_cell.json
Check in the GUI before verifying: the Floquet mode table (mode 1 must be TE, mode 2 TM for the registered
expressions S(FloquetPort1:1,FloquetPort1:1) / :2), and that the de-embed distance equals air_h.
"""
import argparse

from common import call, finish, note, open_new, perfect_e, set_vars, setup_and_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="2024.2")
    ap.add_argument("--non-graphical", action="store_true")
    a = ap.parse_args()
    app = open_new(a.out, "UnitCell", a.version, a.non_graphical)
    set_vars(app, {"p": "10mm", "h": "3mm", "R1": "3mm", "k2": "0.92", "k3": "0.8", "air_h": "15mm"})
    m = app.modeler
    call(m.create_box, ["-p/2", "-p/2", "0mm"], ["p", "p", "h"], name=("Substrate", []), material=("FR4_epoxy", ["matname"]))
    air = call(m.create_box, ["-p/2", "-p/2", "0mm"], ["p", "p", "h+air_h"], name=("AirBox", []), material=("vacuum", ["matname"]))
    ring = call(m.create_circle, "XY", ["0mm", "0mm", "h"], "R1", name=("Ring", []))
    hole = call(m.create_circle, "XY", ["0mm", "0mm", "h"], "k2*R1", name=("RingHole", []))
    m.subtract(ring, hole, False)
    call(m.create_circle, "XY", ["0mm", "0mm", "h"], "k3*R1", name=("Disc", []))
    call(m.create_rectangle, "XY", ["-p/2", "-p/2", "0mm"], ["p", "p"], name=("GroundSheet", []))
    perfect_e(app, ["Ring", "Disc"], "Metal")
    perfect_e(app, ["GroundSheet"], "Ground")
    pairs = app.auto_assign_lattice_pairs(air)
    print("lattice pairs:", pairs)
    if list(pairs or []) != ["LatticePair1", "LatticePair2"]:
        note("lattice pairs are named %s; edit 'boundaries' in song2016_unit_cell.json to match" % pairs)
    call(app.create_floquet_port, air.top_face_z, modes=(2, ["nummodes"]), name=("FloquetPort1", ["portname"]),
         deembed_distance=("air_h", ["deembed_dist"]))
    setup_and_sweep(app, 13, 6, 13, 71, "Interpolating", 20, "MaximumDeltaS", 0.01)
    finish(app, a.out)


if __name__ == "__main__":
    main()
