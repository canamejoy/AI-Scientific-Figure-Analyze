"""Tests for figure + caption extraction (heuristic path, no model)."""

from __future__ import annotations

import tempfile
from pathlib import Path


def build_pdf(caption: bytes) -> bytes:
    """Builds a one-page PDF with a vector figure and the given caption line.

    Args:
        caption: The caption text to place below the figure (raw PDF string
            bytes, e.g. ``b"Figure 1: A plot."``).
    """
    content = (
        b"BT /F1 11 Tf 72 700 Td (As shown in Figure 1, the accuracy increases "
        b"steadily with the number of training epochs.) Tj ET\n"
        b"1 w\n100 420 350 180 re S\n120 440 m 200 500 l S\n"
        b"300 560 m 430 580 l S\n"
        b"BT /F1 10 Tf 100 390 Td (" + caption + b") Tj ET\n"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return bytes(out)


def _parse(caption: bytes):
    from src.parsing.pdf_parser import ScientificPDFParser

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf(caption))
        pdf_path = Path(tmp.name)
    try:
        return ScientificPDFParser(use_layout_model=False).parse(pdf_path), pdf_path
    finally:
        pass  # caller unlinks


def test_figure_and_caption_extraction() -> None:
    document, pdf_path = _parse(b"Figure 1: Accuracy versus training epochs.")
    try:
        assert len(document.pages) == 1
        assert "Figure 1" in document.pages[0].text
        assert len(document.figures) == 1, [f.figure_id for f in document.figures]

        figure = document.figures[0]
        assert figure.figure_id == "figure-1"
        assert figure.label == "Figure 1"
        assert figure.number == "1"
        assert figure.detection_method == "vector-cluster"
        assert figure.caption.startswith("Figure 1:")
        assert figure.dpi == 300
        # 350pt-wide region + padding at 300 DPI → well over 1000 px wide.
        assert figure.image.width > 1000
        assert figure.image.mode == "RGB"
    finally:
        pdf_path.unlink(missing_ok=True)


def test_dotted_figure_number() -> None:
    document, pdf_path = _parse(b"Figure 3.2: A book-style numbered figure.")
    try:
        assert len(document.figures) == 1
        assert document.figures[0].figure_id == "figure-3.2"
        assert document.figures[0].label == "Figure 3.2"
    finally:
        pdf_path.unlink(missing_ok=True)


def test_no_caption_no_figure() -> None:
    """A page whose text is not a caption anchor yields no figures."""
    document, pdf_path = _parse(
        b"This paragraph mentions Figure 1 but is not a caption."
    )
    try:
        assert document.figures == []
    finally:
        pdf_path.unlink(missing_ok=True)


def test_dataset_export() -> None:
    import json
    import shutil

    from src.extraction.exporter import DatasetExporter
    from src.parsing.pdf_parser import ScientificPDFParser

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf(b"Figure 1: Exported figure."))
        pdf_path = Path(tmp.name)
    out_dir = Path(tempfile.mkdtemp(prefix="dataset_"))
    try:
        document = ScientificPDFParser(use_layout_model=False).parse(pdf_path)
        paper_dir = DatasetExporter(out_dir).export(document, paper_name="paper")

        for name in ("figure_001.png", "figure_001.txt", "figure_001.json", "index.json"):
            assert (paper_dir / name).is_file(), f"missing {name}"

        metadata = json.loads((paper_dir / "figure_001.json").read_text(encoding="utf-8"))
        assert metadata["figure_id"] == "figure-1"
        assert metadata["caption"].startswith("Figure 1:")
        assert metadata["image_size_px"][0] > 1000
        assert metadata["files"]["image"] == "figure_001.png"

        index = json.loads((paper_dir / "index.json").read_text(encoding="utf-8"))
        assert len(index["figures"]) == 1
        assert index["figures"][0]["stem"] == "figure_001"
        assert "Exported figure" in (paper_dir / "figure_001.txt").read_text(encoding="utf-8")
    finally:
        pdf_path.unlink(missing_ok=True)
        shutil.rmtree(out_dir, ignore_errors=True)
