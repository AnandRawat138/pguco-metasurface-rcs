import json

import numpy as np
from PIL import Image

from metasurface_optimizer.geometry.geometry_generator import GeometryGenerator
from metasurface_optimizer.geometry.image_parser import analyze_mask, load_mask
from metasurface_optimizer.geometry.stl_exporter import export_stl
from metasurface_optimizer.geometry.stl_loader import load_stl
from metasurface_optimizer.tracking.experiment import ExperimentRecord, SQLiteRepository


def test_stl_export_validates_and_reimports(schema, tmp_path):
    g = GeometryGenerator(schema)
    x = schema.default_vector(); x[schema.index("rotation")] = 4.0
    val = export_stl(g.build(x), tmp_path / "c.stl")
    assert val.ok and all(b["watertight"] for b in val.bodies)
    side = json.loads((tmp_path / "c.json").read_text())
    assert side["unit"] == "mm" and len(side["layers"]) == 3
    a = load_stl(tmp_path / "c.stl", resolution=200)
    comps = a.inferred["components"]
    assert len(comps) == 4
    L = max(c["major_px"] for c in comps) * a.inferred["pixel_size"]
    assert abs(L - 8.5) < 0.3  # recovered within ~2 pixels
    assert abs(comps[0]["orientation_deg"] - 4.0) < 1.0
    assert "approximate" in a.to_dict()["statement"]


def test_image_parser(schema, tmp_path):
    g = GeometryGenerator(schema)
    m = g.rasterize(schema.default_vector(), 256)
    Image.fromarray((~m * 255).astype(np.uint8)).save(tmp_path / "m.png")
    r = analyze_mask(load_mask(tmp_path / "m.png"), period_mm=20.0)
    assert r.n_components == 4
    assert abs(r.estimates["length_mm"] - 8.5) < 0.2
    assert r.to_dict()["source"] == "image_estimate"


def test_experiment_record_and_sqlite(tmp_path, schema):
    rec = ExperimentRecord("t", 0, schema.to_dict(), [], {}, {"name": "x"}, {}, {}, {}, results={"best_validated": {"composite": -1.0}})
    p = rec.save(tmp_path / "e.json")
    d = json.loads(p.read_text())
    for k in ("experiment_id", "timestamp", "dataset_version", "model_version", "random_seed", "parameter_schema", "objective_definition",
              "normalization", "optimizer", "acquisition_function", "physics_constraints", "training_configuration", "candidate_pool_size",
              "number_of_HFSS_evaluations", "results"):
        assert k in d
    repo = SQLiteRepository(tmp_path / "x.sqlite")
    repo.add(rec)
    assert repo.list()[0]["experiment_id"] == rec.experiment_id
    assert repo.get(rec.experiment_id)["name"] == "t"


def test_csv_roundtrip_preserves_objectives(tmp_path):
    """A response written to the documented CSV layouts and read back gives the same objectives (Mode B upload path)."""
    from metasurface_optimizer.core.provenance import Fidelity
    from metasurface_optimizer.data.loaders.target_loader import load_response, write_rcs_csv, write_sparams_csv
    from metasurface_optimizer.experiment import ExperimentBuilder, load_config

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    b = ExperimentBuilder(load_config(root / "configs/experiments/demo_checkerboard.yaml"), root)
    resp = b.solver().evaluate(b.schema.to_dict_values(b.x0()))
    write_sparams_csv(resp.sparams, tmp_path / "s.csv")
    write_rcs_csv(resp.rcs, tmp_path / "r.csv")
    back = load_response(Fidelity.SYNTHETIC_ORACLE, tmp_path / "s.csv", tmp_path / "r.csv", freq_unit="hz")
    assert set(back.sparams.data) == set(resp.sparams.data)
    prob = b.problem()
    np.testing.assert_allclose(prob.objectives.evaluate(back), prob.objectives.evaluate(resp), rtol=1e-6, atol=1e-9)
