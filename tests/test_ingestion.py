"""Ingestion tests on a synthetic, self-generated PDF (no real paper is used)."""
import numpy as np
import pytest

from metasurface_optimizer.core.provenance import Confidence, ValueSource
from metasurface_optimizer.ingestion.pdf_extractor import extract_from_text, extract_paper
from metasurface_optimizer.ingestion.plot_digitizer import AxisCalibration, detect_axes, extract_curve, load_rgb
from metasurface_optimizer.ingestion.reconstruction import reconstruct


def test_text_regex():
    vals = extract_from_text("The unit cell has P = 10 mm and w1=0.5mm on FR4 with εr = 4.4 and tan δ = 0.02, operating 8–12 GHz under TE incidence.")
    names = {v.name: v for v in vals}
    assert names["dim:P"].value == 10.0 and names["dim:w1"].value == 0.5
    assert names["eps_r"].value == 4.4 and names["tan_delta"].value == 0.02
    assert names["frequency_range"].value == [8e9, 12e9]
    assert names["dim:P"].source is ValueSource.TEXT and names["dim:P"].confidence is Confidence.MEDIUM


@pytest.fixture
def plot_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = np.linspace(8, 12, 200); y = -10 - 10 * np.exp(-((f - 10) / 0.5) ** 2)
    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(f, y, color=(1, 0, 0), lw=1.5)
    ax.set_xlim(8, 12); ax.set_ylim(-25, -5); ax.grid(False)
    for s in ax.spines.values():
        s.set_color("black")
    p = tmp_path / "plot.png"; fig.savefig(p); plt.close(fig)
    return p, (f, y)


def test_digitizer_recovers_curve(plot_png):
    p, (f, y) = plot_png
    img = load_rgb(p)
    H, W = img.shape[:2]
    # axes occupy [0.1, 0.9] of the figure in both directions
    cal = AxisCalibration((0.1 * W, 0.9 * W), (8, 12), (0.9 * H, 0.1 * H), (-25, -5))
    c = extract_curve(img, cal, color=(1, 0, 0), tol=0.3, n_points=41)
    yi = np.interp(c.x, f, y)
    assert np.max(np.abs(c.y - yi)) < 4 * c.y_uncertainty + 0.3
    d = c.to_dict()
    assert d["source"] == "plot_digitized" and d["confidence"] == "LOW"
    ax = detect_axes(img)
    assert ax["x_axis_row"] is not None


def test_pdf_extraction_and_reconstruction(tmp_path, plot_png, schema):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Design: period P = 10 mm, patch length La = 8.5 mm, Lb = 5 mm.", fontsize=10)
    page.insert_text((50, 90), "Substrate F4B with relative permittivity 4.4, loss tangent 0.02, h = 3 mm.", fontsize=10)
    page.insert_text((50, 108), "RCS reduction is measured from 8 to 12 GHz.", fontsize=10)
    page.insert_image(fitz.Rect(50, 130, 450, 430), filename=str(plot_png[0]))
    page.insert_text((50, 450), "Fig. 3. Simulated monostatic RCS versus frequency.", fontsize=10)
    pdf = tmp_path / "paper.pdf"; doc.save(pdf)
    ex = extract_paper(pdf, tmp_path / "figs")
    names = {v.name for v in ex.values}
    assert {"dim:P", "dim:La", "dim:Lb", "dim:h", "eps_r", "tan_delta", "frequency_range"} <= names
    assert ex.figures and ex.figures[0].kind_guess == "plot"
    assert ex.to_dict()["status"].startswith("UNVERIFIED")
    rec = reconstruct(schema, ex.values, {"period": "dim:P", "length_a": "dim:La", "length_b": "dim:Lb", "substrate_thickness": "dim:h", "eps_r": "eps_r"},
                      user_values={"width_a": 8.5, "width_b": 5.0, "rotation": 0.0})
    assert rec.values["period"] == 10.0 and rec.provenance["width_a"]["source"] == "user"
    assert rec.requires_user_verification and rec.confidence.value in ("MEDIUM", "LOW")
