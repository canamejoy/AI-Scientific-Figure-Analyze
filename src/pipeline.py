"""Extraction orchestration for the viewer.

A thin facade the Streamlit viewer talks to: parse a PDF into figures and
captions, hold the result, and optionally export it as a dataset. No vision
model is involved — extraction is fully deterministic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from src.extraction.exporter import DatasetExporter
from src.models import ExtractedFigure, ParsedDocument
from src.parsing.pdf_parser import ScientificPDFParser

logger = logging.getLogger(__name__)


class Orchestrator:
    """Parses PDFs into figures + captions and holds the latest document.

    Args:
        parser: Custom PDF parser; a default 300-DPI parser is created when
            omitted.
    """

    def __init__(self, parser: Optional[ScientificPDFParser] = None) -> None:
        self._parser = parser or ScientificPDFParser()
        self._document: Optional[ParsedDocument] = None

    def process_document(self, pdf_path: str) -> None:
        """Parses a PDF, replacing any previously processed document.

        Args:
            pdf_path: Filesystem path to the PDF.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the PDF cannot be parsed at all.
        """
        logger.info("Processing document: %s", pdf_path)
        self._document = self._parser.parse(pdf_path)
        logger.info(
            "Document ready — %d page(s), %d figure(s)",
            len(self._document.pages),
            len(self._document.figures),
        )

    def export(
        self, output_dir: Union[str, Path], paper_name: Optional[str] = None
    ) -> Path:
        """Exports the processed document as a structured dataset.

        Raises:
            RuntimeError: If no document has been processed yet.
        """
        if self._document is None:
            raise RuntimeError("No document has been processed yet.")
        return DatasetExporter(Path(output_dir)).export(
            self._document, paper_name=paper_name
        )

    @property
    def figures(self) -> List[ExtractedFigure]:
        """Every extracted figure in document order (empty before processing)."""
        return list(self._document.figures) if self._document else []

    @property
    def document(self) -> Optional[ParsedDocument]:
        """The currently processed document, if any."""
        return self._document
