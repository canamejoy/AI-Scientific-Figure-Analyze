"""Tests for the layout-model figure-extraction path.

Part 1 uses a stub detector (no model) to verify the full mapping chain:
page render → regions → PDF points → caption text from the text layer →
figure number parsing → high-res crop.

Part 2 loads the REAL DocLayout-YOLO model (downloads weights on first run)
and checks it initializes and runs inference on a rendered page without
crashing. It is skipped automatically when the dependency is unavailable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import pytest

from test_extraction import build_pdf  # shared one-page PDF builder

from src.models import LayoutRegion, PixelBox
from src.parsing.pdf_parser import ScientificPDFParser


class StubLayoutDetector:
    """Returns hand-placed figure/caption regions for the synthetic page.

    The synthetic page is 612x792 pt; at the 144-DPI detection render this is
    1224x1584 px (scale 2.0). ``build_pdf`` draws the figure at x 100–450 pt /
    y ~192–372 pt, with the caption line near y ~394 pt.
    """

    def detect(self, page_image) -> List[LayoutRegion]:
        return [
            LayoutRegion(
                label="figure",
                box=PixelBox(x0=195, y0=375, x1=915, y1=750),
                confidence=0.93,
            ),
            LayoutRegion(
                label="figure_caption",
                box=PixelBox(x0=195, y0=780, x1=1160, y1=820),
                confidence=0.91,
            ),
            # Noise the parser must ignore.
            LayoutRegion(
                label="plain_text",
                box=PixelBox(x0=140, y0=150, x1=1100, y1=200),
                confidence=0.99,
            ),
        ]


def test_layout_path_with_stub() -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf(b"Figure 1: Accuracy versus training epochs."))
        pdf_path = Path(tmp.name)
    try:
        parser = ScientificPDFParser(
            use_layout_model=True, layout_detector=StubLayoutDetector()
        )
        document = parser.parse(pdf_path)
        assert len(document.figures) == 1
        figure = document.figures[0]
        assert figure.detection_method == "layout-model"
        assert figure.figure_id == "figure-1"
        assert figure.label == "Figure 1"
        # Caption read from the PDF text layer inside the detected region.
        assert figure.caption.startswith("Figure 1:")
        assert figure.image.width > 100
    finally:
        pdf_path.unlink(missing_ok=True)


def test_real_model_loads() -> None:
    import pdfplumber

    try:
        from src.parsing.layout_detector import LayoutFigureDetector
    except ImportError:
        pytest.skip("doclayout-yolo not installed")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf(b"Figure 1: A plot."))
        pdf_path = Path(tmp.name)
    try:
        detector = LayoutFigureDetector()
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_image = pdf.pages[0].to_image(resolution=144).original
        regions = detector.detect(page_image)
        # Content on a synthetic page is not asserted — just that it ran.
        assert isinstance(regions, list)
    finally:
        pdf_path.unlink(missing_ok=True)
