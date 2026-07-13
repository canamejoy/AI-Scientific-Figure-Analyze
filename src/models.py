"""Typed data models for figure extraction.

All models are Pydantic v2 models so the pipeline stays fully validated and
serializable (except the in-memory ``PIL.Image`` payloads, allowed through
``arbitrary_types_allowed``).
"""

from __future__ import annotations

from typing import List, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

# How a figure's bounding box was determined on the page.
DetectionMethod = Literal[
    "layout-model", "embedded-image", "vector-cluster", "caption-fallback"
]


class BoundingBox(BaseModel):
    """Axis-aligned rectangle in PDF *point* space with a top-left origin.

    pdfplumber reports geometry with ``y`` increasing downwards, so
    ``top < bottom`` for every valid box. One PDF point equals 1/72 inch, so a
    box of width ``w`` points renders to ``w * dpi / 72`` pixels at a given
    resolution.
    """

    x0: float = Field(..., description="Left edge in PDF points.")
    top: float = Field(..., description="Top edge in PDF points (smaller = higher).")
    x1: float = Field(..., description="Right edge in PDF points.")
    bottom: float = Field(..., description="Bottom edge in PDF points.")

    @property
    def width(self) -> float:
        """Horizontal extent in points (may be negative for degenerate boxes)."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Vertical extent in points (may be negative for degenerate boxes)."""
        return self.bottom - self.top

    @property
    def area(self) -> float:
        """Non-negative area in square points."""
        return max(self.width, 0.0) * max(self.height, 0.0)

    @property
    def vertical_center(self) -> float:
        """The ``y`` coordinate halfway between ``top`` and ``bottom``."""
        return (self.top + self.bottom) / 2.0

    def padded(self, pad: float) -> "BoundingBox":
        """Returns a copy grown by ``pad`` points on every side."""
        return BoundingBox(
            x0=self.x0 - pad,
            top=self.top - pad,
            x1=self.x1 + pad,
            bottom=self.bottom + pad,
        )

    def clamped(self, x0: float, top: float, x1: float, bottom: float) -> "BoundingBox":
        """Returns the intersection of this box with the given bounds.

        May be degenerate (zero or negative area) when this box lies entirely
        outside the bounds — callers should check :attr:`width`/:attr:`height`.
        """
        return BoundingBox(
            x0=max(self.x0, x0),
            top=max(self.top, top),
            x1=min(self.x1, x1),
            bottom=min(self.bottom, bottom),
        )

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Returns the smallest box containing both this box and ``other``."""
        return BoundingBox(
            x0=min(self.x0, other.x0),
            top=min(self.top, other.top),
            x1=max(self.x1, other.x1),
            bottom=max(self.bottom, other.bottom),
        )

    def horizontal_overlap(self, other: "BoundingBox") -> float:
        """Length (in points) of the shared horizontal span with ``other``.

        Returns ``0.0`` when the boxes do not overlap horizontally.
        """
        return max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))


class PixelBox(BaseModel):
    """Axis-aligned rectangle in *image pixel* space (origin top-left).

    Used for regions returned by the page-layout detection model.
    """

    x0: int = Field(..., ge=0)
    y0: int = Field(..., ge=0)
    x1: int = Field(..., ge=0)
    y1: int = Field(..., ge=0)

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def clamped(self, width: int, height: int) -> "PixelBox":
        """Returns a copy confined to an image of the given dimensions."""
        return PixelBox(
            x0=max(0, min(self.x0, width)),
            y0=max(0, min(self.y0, height)),
            x1=max(0, min(self.x1, width)),
            y1=max(0, min(self.y1, height)),
        )


class LayoutRegion(BaseModel):
    """One region found by a page-layout detection model.

    Attributes:
        label: Normalized region type (``"figure"``, ``"figure_caption"``, ...).
        box: The region's extent in the rendered page image's pixel space.
        confidence: Detector confidence in [0, 1].
    """

    label: str
    box: PixelBox
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class PageText(BaseModel):
    """Full extracted text of a single PDF page.

    Attributes:
        page_number: 1-based page index, matching pdfplumber's convention.
        text: The page text (may be empty for image-only pages).
    """

    page_number: int = Field(..., ge=1)
    text: str


class ExtractedFigure(BaseModel):
    """A complete figure detected in the document, with its rendered crop.

    Attributes:
        figure_id: Stable identifier derived from the caption number
            (``"figure-3"``), or a page-local id when no number was parsed.
        label: Human-readable label, e.g. ``"Figure 3"``.
        number: Raw figure number from the caption (``"3"``, ``"4.1"``, or a
            synthesized ``"p2-1"`` when none was found).
        caption: The full caption paragraph, whitespace-normalized.
        page_number: 1-based page the figure was found on.
        bbox: The figure region in PDF point coordinates (top-left origin).
        image: The rendered high-resolution figure as an RGB ``PIL.Image``.
        detection_method: How the region was located.
        dpi: The effective resolution the figure was rendered at.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    figure_id: str
    label: str
    number: str
    caption: str
    page_number: int = Field(..., ge=1)
    bbox: BoundingBox
    image: Image.Image
    detection_method: DetectionMethod
    dpi: int = Field(..., ge=36)


class ParsedDocument(BaseModel):
    """The complete result of parsing one PDF.

    Attributes:
        source_path: Filesystem path of the parsed PDF.
        pages: Per-page extracted text, in page order.
        figures: Every extracted figure, in document order.
    """

    source_path: str
    pages: List[PageText]
    figures: List[ExtractedFigure]

    @property
    def full_text(self) -> str:
        """The whole document text with pages joined by blank lines."""
        return "\n\n".join(page.text for page in self.pages)

    def page_text(self, page_number: int) -> str:
        """Returns the text of a specific page.

        Raises:
            KeyError: If the page does not exist in this document.
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page.text
        raise KeyError(f"Page {page_number} not found in '{self.source_path}'.")
