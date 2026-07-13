"""Structured on-disk dataset export.

Produces, per figure::

    <output_dir>/<paper_name>/
        figure_001.png     # the complete figure, original resolution
        figure_001.txt     # the full caption, verbatim
        figure_001.json    # figure metadata

    <output_dir>/<paper_name>/index.json   # one entry per figure

Images are written losslessly (PNG) at their extracted resolution — no
resizing, no recompression beyond PNG encoding.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models import ExtractedFigure, ParsedDocument

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    """Reduces an arbitrary string to a filesystem-safe directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "paper"


class DatasetExporter:
    """Writes an extracted document to a structured dataset directory.

    Args:
        output_dir: Root directory datasets are written under (created on
            demand).
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def export(
        self, document: ParsedDocument, paper_name: Optional[str] = None
    ) -> Path:
        """Exports every figure of a parsed document.

        Args:
            document: The parsed document.
            paper_name: Dataset folder name; defaults to the PDF's stem.

        Returns:
            The path of the written paper directory.
        """
        name = _safe_name(paper_name or Path(document.source_path).stem)
        paper_dir = self.output_dir / name
        paper_dir.mkdir(parents=True, exist_ok=True)

        index: List[Dict[str, Any]] = []
        for position, figure in enumerate(document.figures, start=1):
            stem = f"figure_{position:03d}"
            figure.image.save(paper_dir / f"{stem}.png", format="PNG")
            (paper_dir / f"{stem}.txt").write_text(figure.caption, encoding="utf-8")

            metadata = self._figure_metadata(figure)
            metadata["files"] = {"image": f"{stem}.png", "caption": f"{stem}.txt"}
            (paper_dir / f"{stem}.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            index.append({"stem": stem, **metadata})
            logger.info("Exported %s -> %s.png", figure.label, stem)

        (paper_dir / "index.json").write_text(
            json.dumps(
                {"source_pdf": document.source_path, "figures": index},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(
            "Dataset written: %s (%d figure(s))", paper_dir, len(document.figures)
        )
        return paper_dir

    @staticmethod
    def _figure_metadata(figure: ExtractedFigure) -> Dict[str, Any]:
        """JSON-serializable metadata for a figure."""
        return {
            "figure_id": figure.figure_id,
            "label": figure.label,
            "number": figure.number,
            "page_number": figure.page_number,
            "caption": figure.caption,
            "detection_method": figure.detection_method,
            "dpi": figure.dpi,
            "bbox_pdf_points": figure.bbox.model_dump(),
            "image_size_px": [figure.image.width, figure.image.height],
        }
