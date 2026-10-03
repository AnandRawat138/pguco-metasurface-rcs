"""Research-paper ingestion for the Evaluation / Existing Paper tab (an engineering feature).

The scientific method never requires a paper: it needs an existing design and
its response. This module only helps recover them. Every extracted value is an
:class:`ExtractedValue` with ``source`` (text / table / plot_digitized / ...) and
``confidence``, and must be verified by the user before optimisation.

Extraction:
* text with PyMuPDF; regular expressions for dimensions (``P = 10 mm``),
  substrate permittivity / loss tangent / thickness, frequency ranges,
  polarisation and incidence angle;
* tables via PyMuPDF's table finder (values from tables get HIGH confidence);
* figures (embedded images) with their page and nearby caption, saved as PNG
  for the semi-automatic plot digitiser and the unit-cell image parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from metasurface_optimizer.core.provenance import Confidence, ExtractedValue, ValueSource

_NUM = r"([0-9]+(?:\.[0-9]+)?)"
_UNIT = r"(mm|cm|um|µm|mil|m)\b"
# symbol = value unit, e.g. "P = 10 mm", "w1=0.5mm", "period of 12 mm"
_DIM_PATTERNS = [
    re.compile(r"\b([A-Za-z][A-Za-z0-9_]{0,6})\s*(?:=|:)\s*" + _NUM + r"\s*" + _UNIT),
    re.compile(r"\b(period|periodicity|length|width|gap|thickness|radius|spacing)\s+(?:of\s+|is\s+)?" + _NUM + r"\s*" + _UNIT, re.I),
]
_EPS = re.compile(r"(?:ε\s*r|εr|epsilon_?r|\\varepsilon_?r|dielectric constant|relative permittivity|permittivity)\s*(?:of|=|:|is)?\s*" + _NUM, re.I)
_TAND = re.compile(r"(?:tan\s*δ|tanδ|tan\s*delta|loss tangent)\s*(?:of|=|:|is)?\s*" + r"([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)", re.I)
_FREQ = re.compile(r"" + _NUM + r"\s*(?:-|–|—|to)\s*" + _NUM + r"\s*(GHz|MHz)", re.I)
_SUBSTRATE = re.compile(r"\b(FR-?4|Rogers\s*RO?\s*\d{4}[A-Z]?|RT/?duroid\s*\d{4}|F4B|Taconic\s*\w+|polyimide|PTFE)\b", re.I)
_ANGLE = re.compile(r"(?:incidence|incident)\s+angles?\s*(?:of|up to|from)?\s*" + _NUM + r"\s*°?", re.I)
_POL = re.compile(r"\b(TE|TM|x-polari[sz]ed|y-polari[sz]ed|LHCP|RHCP|dual-polari[sz]ed|polari[sz]ation[- ]insensitive)\b", re.I)


@dataclass
class FigureInfo:
    page: int
    path: str
    caption: str
    kind_guess: str  # "unit_cell" | "plot" | "unknown"


@dataclass
class PaperExtraction:
    source_file: str
    n_pages: int
    values: List[ExtractedValue] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    figures: List[FigureInfo] = field(default_factory=list)
    captions: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"source_file": self.source_file, "n_pages": self.n_pages, "values": [v.to_dict() for v in self.values],
                "tables": self.tables, "figures": [f.__dict__ for f in self.figures], "captions": self.captions,
                "warnings": self.warnings, "status": "UNVERIFIED - user must confirm every value before optimisation"}


def _to_mm(v: float, unit: str) -> float:
    return v * {"mm": 1, "cm": 10, "um": 1e-3, "µm": 1e-3, "mil": 0.0254, "m": 1000}[unit.lower()]


def extract_from_text(text: str, source: ValueSource = ValueSource.TEXT, page: Optional[int] = None) -> List[ExtractedValue]:
    out: List[ExtractedValue] = []
    note = f"page {page}" if page is not None else ""
    for pat in _DIM_PATTERNS:
        for m in pat.finditer(text):
            sym, val, unit = m.group(1), float(m.group(2)), m.group(3)
            out.append(ExtractedValue(f"dim:{sym}", _to_mm(val, unit), "mm", source, note=f"{note} '{m.group(0).strip()}'"))
    for m in _EPS.finditer(text):
        v = float(m.group(1))
        if 1.0 <= v <= 100:
            out.append(ExtractedValue("eps_r", v, "1", source, note=f"{note} '{m.group(0).strip()}'"))
    for m in _TAND.finditer(text):
        v = float(m.group(1))
        if 0 <= v < 1:
            out.append(ExtractedValue("tan_delta", v, "1", source, note=f"{note} '{m.group(0).strip()}'"))
    for m in _FREQ.finditer(text):
        s = 1e9 if m.group(3).lower() == "ghz" else 1e6
        out.append(ExtractedValue("frequency_range", [float(m.group(1)) * s, float(m.group(2)) * s], "Hz", source, note=f"{note} '{m.group(0)}'"))
    for m in _SUBSTRATE.finditer(text):
        out.append(ExtractedValue("substrate_material", m.group(1), None, source, note=note))
    for m in _ANGLE.finditer(text):
        out.append(ExtractedValue("incidence_angle_max", float(m.group(1)), "deg", source, Confidence.LOW, note=f"{note} '{m.group(0)}'"))
    for m in _POL.finditer(text):
        out.append(ExtractedValue("polarization", m.group(1), None, source, Confidence.LOW, note=note))
    return out


def _dedupe(values: List[ExtractedValue]) -> List[ExtractedValue]:
    seen, out = set(), []
    for v in values:
        k = (v.name, str(v.value), v.source.value)
        if k not in seen:
            seen.add(k); out.append(v)
    return out


def extract_paper(pdf_path: str | Path, figure_dir: Optional[str | Path] = None, min_image_px: int = 120,
                  min_vector_paths: int = 20, render_dpi: int = 110) -> PaperExtraction:
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    ex = PaperExtraction(str(pdf_path), doc.page_count)
    fig_dir = Path(figure_dir) if figure_dir else None
    if fig_dir:
        fig_dir.mkdir(parents=True, exist_ok=True)
    cap_re = re.compile(r"^(Fig\.?|Figure|TABLE|Table)\s*([0-9IVX]+)[.:]?\s*(.*)", re.I)
    for pno in range(doc.page_count):
        page = doc[pno]
        text = page.get_text()
        ex.values += extract_from_text(text, ValueSource.TEXT, pno + 1)
        page_caps = []
        for block in page.get_text("blocks"):
            t = block[4].strip().replace("\n", " ")
            m = cap_re.match(t)
            if m:
                cap = {"page": pno + 1, "label": f"{m.group(1)} {m.group(2)}", "text": t[:400], "bbox": list(block[:4])}
                ex.captions.append(cap); page_caps.append(cap)
        try:
            for ti, tab in enumerate(page.find_tables().tables):
                rows = tab.extract()
                ex.tables.append({"page": pno + 1, "index": ti, "rows": rows})
                flat = " ".join(" = ".join(str(c) for c in r if c) for r in rows if r)
                ex.values += extract_from_text(flat, ValueSource.TABLE, pno + 1)
        except Exception as exc:  # pragma: no cover - depends on PyMuPDF build
            ex.warnings.append(f"table extraction failed on page {pno + 1}: {exc}")
        if fig_dir:
            for ii, img in enumerate(page.get_images(full=True)):
                try:
                    pix = fitz.Pixmap(doc, img[0])
                    if pix.width < min_image_px or pix.height < min_image_px:
                        continue
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    p = fig_dir / f"page{pno + 1}_img{ii}.png"
                    pix.save(p)
                    cap = page_caps[0]["text"] if page_caps else ""
                    kind = "plot" if re.search(r"S11|S21|RCS|reflection|transmission|dB|frequency|monostatic|bistatic", cap, re.I) else \
                        ("unit_cell" if re.search(r"unit cell|geometry|element|structure|layout|schematic", cap, re.I) else "unknown")
                    ex.figures.append(FigureInfo(pno + 1, str(p), cap, kind))
                except Exception as exc:  # pragma: no cover
                    ex.warnings.append(f"image {ii} on page {pno + 1}: {exc}")
            # Vector-drawn plots carry no embedded image: render the page so it can still be digitised.
            if not page.get_images() and len(page.get_drawings()) >= min_vector_paths:
                p = fig_dir / f"page{pno + 1}_render.png"
                page.get_pixmap(dpi=render_dpi).save(p)
                cap = page_caps[0]["text"] if page_caps else ""
                ex.figures.append(FigureInfo(pno + 1, str(p), cap or "full-page render (vector graphics)", "page_render"))
    ex.values = _dedupe(ex.values)
    if not ex.values:
        ex.warnings.append("No parameter values recognised in the text; scanned PDFs need OCR, or enter values manually.")
    return ex
