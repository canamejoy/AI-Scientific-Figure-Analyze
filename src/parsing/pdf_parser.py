"""Scientific PDF figure and caption extraction.

Walks every page of a PDF with ``pdfplumber``, locates each complete figure,
and pairs it with its caption. Figure detection uses two strategies:

1. **Page-layout model (primary).** The rendered page is analyzed by
   DocLayout-YOLO, which locates ``figure`` and ``figure_caption`` regions
   *visually* — robust across journal layouts, including two-column papers.
   Each figure's caption text is read from the PDF text layer inside the
   detected caption region.

2. **Caption-anchor heuristic (fallback).** When the layout model is
   unavailable, captions are found by an anchor regex ("Figure 3:", "Fig. 2.",
   "FIGURE 4 —") and each caption is mapped to the graphical region it
   describes (the union of raster/vector objects in a search band above or
   below the caption, or a caption-relative window when none is found).

Each detected figure is rendered into a high-resolution RGB ``PIL.Image``.
No panel splitting is performed — a downstream consumer receives the complete
figure and its full caption.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import pdfplumber
from PIL import Image

from src.models import (
    BoundingBox,
    DetectionMethod,
    ExtractedFigure,
    LayoutRegion,
    PageText,
    ParsedDocument,
    PixelBox,
)

logger = logging.getLogger(__name__)

# A text line as returned by pdfplumber's ``page.extract_text_lines()``.
_LineDict = Dict[str, Any]

# Caption anchor: "Figure 3:", "Fig. 12.", "FIGURE 2 -", "Figure 4a —",
# "Figure 3.2: ..." (book-style dotted numbering). The separator after the
# number is *required* so in-prose sentences such as "Figure 3 shows the
# results" are not mistaken for captions.
_CAPTION_ANCHOR: "re.Pattern[str]" = re.compile(
    r"^\s*(?:fig(?:ure)?\.?)\s*"
    r"(?P<number>\d{1,3}(?:\.\d{1,2})?[a-z]?)"
    r"\s*(?P<sep>[:.\-–—|])\s*\S",
    re.IGNORECASE,
)

# Figure number inside caption prose (layout-model path, where the caption
# text may start anywhere): "FIG. 3.", "Figure 4:", "Fig 2a".
_CAPTION_NUMBER: "re.Pattern[str]" = re.compile(
    r"\bfig(?:ure)?\.?\s*(?P<number>\d{1,3}(?:\.\d{1,2})?[a-z]?)",
    re.IGNORECASE,
)


class _CaptionMatch(NamedTuple):
    """A caption anchor located on a page, with its full paragraph text."""

    number: str
    label: str
    text: str
    bbox: BoundingBox


def _line_bbox(line: _LineDict) -> BoundingBox:
    """Builds a :class:`BoundingBox` from a pdfplumber text-line dict.

    Coordinates are cast to ``float`` because pdfplumber occasionally yields
    ``Decimal`` values depending on the underlying pdfminer objects.
    """
    return BoundingBox(
        x0=float(line["x0"]),
        top=float(line["top"]),
        x1=float(line["x1"]),
        bottom=float(line["bottom"]),
    )


class ScientificPDFParser:
    """Parses scientific PDFs into per-page text and high-resolution figures.

    Stateless between :meth:`parse` calls and safe to reuse for multiple
    documents.

    Args:
        resolution: Target render resolution in DPI (minimum 72). 300 DPI
            keeps axis annotations legible.
        max_render_edge_px: Safety cap on the longest rendered edge in pixels;
            the effective DPI is reduced for very large regions.
        bbox_padding_pt: Padding (PDF points) around detected regions so
            anti-aliased strokes at the border are not clipped.
        min_figure_width_pt: Minimum plausible figure width (heuristic path).
        min_figure_height_pt: Minimum plausible figure height (heuristic path).
        fallback_region_height_pt: Height of the caption-relative window used
            when no graphic can be associated with a caption.
        use_layout_model: Whether to use DocLayout-YOLO as the primary figure
            locator. ``None`` reads ``LAYOUT_DETECTION`` (on unless set to
            ``off``/``false``/``0``). When the model or its dependencies are
            unavailable, the parser silently falls back to the heuristic path.
        layout_detector: Injected detector instance (mainly for tests); built
            lazily from the environment when omitted.
        layout_detect_dpi: Resolution pages are rendered at for layout
            detection (final figures still render at ``resolution``).
        layout_min_confidence: Minimum confidence for a detected figure region.

    Raises:
        ValueError: If ``resolution`` is below 72 DPI.
    """

    DEFAULT_RESOLUTION: int = 300

    def __init__(
        self,
        resolution: int = DEFAULT_RESOLUTION,
        max_render_edge_px: int = 4000,
        bbox_padding_pt: float = 6.0,
        min_figure_width_pt: float = 60.0,
        min_figure_height_pt: float = 40.0,
        fallback_region_height_pt: float = 320.0,
        use_layout_model: Optional[bool] = None,
        layout_detector: Optional[object] = None,
        layout_detect_dpi: int = 144,
        layout_min_confidence: float = 0.3,
    ) -> None:
        if resolution < 72:
            raise ValueError(f"resolution must be >= 72 DPI, got {resolution}.")
        self.resolution = resolution
        self.max_render_edge_px = max_render_edge_px
        self.bbox_padding_pt = bbox_padding_pt
        self.min_figure_width_pt = min_figure_width_pt
        self.min_figure_height_pt = min_figure_height_pt
        self.fallback_region_height_pt = fallback_region_height_pt
        if use_layout_model is None:
            use_layout_model = (
                os.getenv("LAYOUT_DETECTION") or "on"
            ).strip().lower() not in {"0", "off", "false", "no"}
        self.use_layout_model = use_layout_model
        self.layout_detect_dpi = layout_detect_dpi
        self.layout_min_confidence = layout_min_confidence
        self._layout_detector = layout_detector
        self._layout_detector_failed = False

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def parse(self, pdf_path: Union[str, Path]) -> ParsedDocument:
        """Parses a PDF into page texts and extracted figures.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            A :class:`~src.models.ParsedDocument` holding every page's text
            (page order preserved) and all extracted figures.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.
            RuntimeError: If the PDF cannot be opened or read at all.
                Page-level failures are logged and skipped instead of
                aborting the whole document.
        """
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        pages: List[PageText] = []
        figures: List[ExtractedFigure] = []
        seen_ids: Set[str] = set()

        logger.info("Parsing '%s' at %d DPI", path.name, self.resolution)
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        logger.exception(
                            "Text extraction failed on page %d", page.page_number
                        )
                        text = ""
                    pages.append(PageText(page_number=page.page_number, text=text))

                    # Primary: visual layout detection on the rendered page.
                    # Fallback: caption-anchor heuristics over PDF coordinates.
                    page_figures: List[ExtractedFigure] = []
                    detector = self._get_layout_detector()
                    if detector is not None:
                        try:
                            page_figures = self._extract_with_layout_model(
                                page, detector, seen_ids
                            )
                        except Exception:
                            logger.exception(
                                "Layout-model extraction failed on page %d",
                                page.page_number,
                            )
                    if not page_figures:
                        try:
                            page_figures = self._extract_page_figures(page, seen_ids)
                        except Exception:
                            logger.exception(
                                "Figure extraction failed on page %d",
                                page.page_number,
                            )
                    figures.extend(page_figures)

                    # Release pdfplumber's per-page caches so large documents
                    # do not accumulate memory across pages.
                    try:
                        page.flush_cache()
                    except Exception:  # pragma: no cover - defensive only
                        pass
        except Exception as exc:
            raise RuntimeError(f"Unable to parse PDF '{path}': {exc}") from exc

        logger.info(
            "Parsed %d page(s) from '%s'; extracted %d figure(s)",
            len(pages),
            path.name,
            len(figures),
        )
        return ParsedDocument(source_path=str(path), pages=pages, figures=figures)

    # ------------------------------------------------------------------ #
    # Layout-model extraction (primary)                                   #
    # ------------------------------------------------------------------ #

    def _get_layout_detector(self) -> Optional[object]:
        """Returns the layout detector, building it lazily.

        Returns ``None`` when layout detection is disabled, its dependencies
        are missing, or a previous construction attempt failed — the caller
        then uses the heuristic path.
        """
        if not self.use_layout_model or self._layout_detector_failed:
            return None
        if self._layout_detector is not None:
            return self._layout_detector
        try:
            from src.parsing.layout_detector import LayoutFigureDetector

            self._layout_detector = LayoutFigureDetector(
                confidence=self.layout_min_confidence
            )
        except Exception as exc:
            logger.warning(
                "Layout model unavailable (%s) — using caption-anchor "
                "heuristics instead. Install with: pip install doclayout-yolo "
                "huggingface-hub",
                exc,
            )
            self._layout_detector_failed = True
            return None
        return self._layout_detector

    def _extract_with_layout_model(
        self,
        page: "pdfplumber.page.Page",
        detector: object,
        seen_ids: Set[str],
    ) -> List[ExtractedFigure]:
        """Extracts figures using visual layout detection on the page image.

        Pipeline: render the page → detect ``figure`` / ``figure_caption``
        regions → map boxes back to PDF points → read each caption's text
        from the PDF text layer → crop the figure at full resolution.

        Args:
            page: The pdfplumber page.
            detector: A detector exposing ``detect(image) -> List[LayoutRegion]``.
            seen_ids: Global figure-id registry.

        Returns:
            Figures found on this page; empty when the detector reports
            nothing, which triggers the heuristic fallback.
        """
        rendered_page = page.to_image(resolution=self.layout_detect_dpi).original
        regions: List[LayoutRegion] = detector.detect(rendered_page)  # type: ignore[attr-defined]

        figure_regions = [
            r
            for r in regions
            if r.label == "figure" and r.confidence >= self.layout_min_confidence
        ]
        caption_regions = [r for r in regions if r.label == "figure_caption"]
        if not figure_regions:
            return []

        # Pixel (detect render) → PDF point mapping.
        scale = 72.0 / float(self.layout_detect_dpi)
        page_x0, page_top, page_x1, page_bottom = (float(v) for v in page.bbox)

        def to_points(box: PixelBox) -> BoundingBox:
            return BoundingBox(
                x0=page_x0 + box.x0 * scale,
                top=page_top + box.y0 * scale,
                x1=page_x0 + box.x1 * scale,
                bottom=page_top + box.y1 * scale,
            ).clamped(page_x0, page_top, page_x1, page_bottom)

        figure_regions.sort(key=lambda r: (r.box.y0, r.box.x0))  # reading order
        extracted: List[ExtractedFigure] = []

        for index, figure_region in enumerate(figure_regions, start=1):
            region_pts = to_points(figure_region.box)
            caption_region = self._nearest_caption(figure_region.box, caption_regions)
            caption_text = (
                self._extract_caption_text(page, to_points(caption_region.box))
                if caption_region
                else ""
            )

            number_match = _CAPTION_NUMBER.search(caption_text)
            if number_match:
                number = number_match.group("number")
            else:
                # No parseable number: synthesize a stable page-local id.
                number = f"p{page.page_number}-{index}"
            label = f"Figure {number}"
            figure_id = f"figure-{number.lower()}"
            if figure_id in seen_ids:
                logger.debug(
                    "Skipping duplicate layout-model figure %s on page %d",
                    figure_id,
                    page.page_number,
                )
                continue

            image, dpi = self._render_region(page, region_pts)
            if image is None:
                continue

            caption = caption_text or f"({label} — no caption text detected)"
            seen_ids.add(figure_id)
            extracted.append(
                ExtractedFigure(
                    figure_id=figure_id,
                    label=label,
                    number=number,
                    caption=caption,
                    page_number=page.page_number,
                    bbox=region_pts,
                    image=image,
                    detection_method="layout-model",
                    dpi=dpi,
                )
            )
            logger.info(
                "Extracted %s via layout model (page %d, conf=%.2f, %dx%d px)",
                label,
                page.page_number,
                figure_region.confidence,
                image.width,
                image.height,
            )
        return extracted

    @staticmethod
    def _nearest_caption(
        figure_box: PixelBox, caption_regions: List[LayoutRegion]
    ) -> Optional[LayoutRegion]:
        """Pairs a figure with its most plausible caption region.

        Candidates must overlap the figure horizontally; the winner minimizes
        the vertical gap, with a bias toward captions *below* the figure (the
        dominant convention in scientific journals).
        """
        best: Optional[LayoutRegion] = None
        best_score = float("inf")
        for candidate in caption_regions:
            overlap = min(figure_box.x1, candidate.box.x1) - max(
                figure_box.x0, candidate.box.x0
            )
            if overlap <= 0:
                continue
            if candidate.box.y0 >= figure_box.y1:
                gap = candidate.box.y0 - figure_box.y1  # below the figure
            elif candidate.box.y1 <= figure_box.y0:
                gap = (figure_box.y0 - candidate.box.y1) + 40  # above: biased
            else:
                gap = 0  # overlapping (caption inside the figure box)
            if gap < best_score:
                best_score = gap
                best = candidate
        return best

    def _extract_caption_text(
        self, page: "pdfplumber.page.Page", bbox: BoundingBox
    ) -> str:
        """Reads a caption's text from the PDF text layer within a region.

        Some PDFs use kerning so tight that pdfplumber's default word
        segmentation drops the inter-word spaces ("Theenergiesare..."). Several
        ``x_tolerance`` values are tried and the variant recovering the most
        spaces wins — as long as it does not overshoot into shredded single
        letters ("T h e"), which a space-ratio ceiling filters out.
        """
        page_x0, page_top, page_x1, page_bottom = (float(v) for v in page.bbox)
        clamped = bbox.padded(2.0).clamped(page_x0, page_top, page_x1, page_bottom)
        if clamped.width <= 1 or clamped.height <= 1:
            return ""
        try:
            region = page.within_bbox(
                (clamped.x0, clamped.top, clamped.x1, clamped.bottom)
            )
            text = " ".join((region.extract_text() or "").split())
        except Exception:
            logger.exception("Caption text extraction failed")
            return ""

        if len(text) <= 40:
            return text
        candidates = [text]
        for tolerance in (1.5, 1.0, 0.7):
            try:
                variant = " ".join(
                    (region.extract_text(x_tolerance=tolerance) or "").split()
                )
            except Exception:  # pragma: no cover - defensive only
                continue
            if variant:
                candidates.append(variant)

        def space_ratio(candidate: str) -> float:
            return candidate.count(" ") / max(len(candidate), 1)

        # Normal English prose sits around a 0.12-0.20 space ratio; beyond
        # ~0.25 the tokenizer started splitting inside words.
        plausible = [c for c in candidates if space_ratio(c) <= 0.25] or candidates
        return max(plausible, key=lambda c: c.count(" "))

    # ------------------------------------------------------------------ #
    # Heuristic extraction (fallback)                                     #
    # ------------------------------------------------------------------ #

    def _extract_page_figures(
        self, page: "pdfplumber.page.Page", seen_ids: Set[str]
    ) -> List[ExtractedFigure]:
        """Finds and crops every captioned figure on one page (heuristic path).

        Args:
            page: The pdfplumber page object.
            seen_ids: Figure ids already extracted (first match wins).

        Returns:
            The figures extracted from this page, possibly empty.
        """
        lines: List[_LineDict] = page.extract_text_lines() or []
        captions = self._find_captions(lines)
        if not captions:
            return []

        graphics = self._collect_graphics(page)
        extracted: List[ExtractedFigure] = []

        for match in captions:
            figure_id = f"figure-{match.number.lower()}"
            if figure_id in seen_ids:
                logger.debug(
                    "Skipping duplicate caption for %s on page %d",
                    figure_id,
                    page.page_number,
                )
                continue

            region, method = self._locate_figure_region(
                page, match.bbox, lines, graphics
            )
            image, dpi = self._render_region(page, region)
            if image is None:
                logger.warning(
                    "Could not render a crop for %s on page %d — skipping",
                    figure_id,
                    page.page_number,
                )
                continue

            seen_ids.add(figure_id)
            extracted.append(
                ExtractedFigure(
                    figure_id=figure_id,
                    label=match.label,
                    number=match.number,
                    caption=match.text,
                    page_number=page.page_number,
                    bbox=region,
                    image=image,
                    detection_method=method,
                    dpi=dpi,
                )
            )
            logger.info(
                "Extracted %s (page %d, %s, %dx%d px @ %d DPI)",
                match.label,
                page.page_number,
                method,
                image.width,
                image.height,
                dpi,
            )
        return extracted

    def _find_captions(self, lines: List[_LineDict]) -> List[_CaptionMatch]:
        """Detects caption anchors and gathers their full paragraph text.

        A caption starts at a line matching :data:`_CAPTION_ANCHOR` and
        continues over subsequent lines while (a) the vertical gap stays below
        ~0.8 of a line height (same paragraph) and (b) no new caption anchor
        begins.
        """
        matches: List[_CaptionMatch] = []
        for idx, line in enumerate(lines):
            text = (line.get("text") or "").strip()
            anchor = _CAPTION_ANCHOR.match(text)
            if not anchor:
                continue

            number = anchor.group("number")
            caption_lines: List[_LineDict] = [line]
            prev = line
            for nxt in lines[idx + 1 :]:
                nxt_text = (nxt.get("text") or "").strip()
                if _CAPTION_ANCHOR.match(nxt_text):
                    break  # the next figure's caption starts here
                prev_box = _line_bbox(prev)
                nxt_box = _line_bbox(nxt)
                line_height = max(prev_box.height, 6.0)
                if nxt_box.top - prev_box.bottom > 0.8 * line_height:
                    break
                caption_lines.append(nxt)
                prev = nxt

            bbox = _line_bbox(caption_lines[0])
            for extra in caption_lines[1:]:
                bbox = bbox.union(_line_bbox(extra))
            caption_text = " ".join(
                (part.get("text") or "").strip() for part in caption_lines
            ).strip()

            matches.append(
                _CaptionMatch(
                    number=number,
                    label=f"Figure {number}",
                    text=caption_text,
                    bbox=bbox,
                )
            )
        return matches

    def _locate_figure_region(
        self,
        page: "pdfplumber.page.Page",
        caption_bbox: BoundingBox,
        lines: List[_LineDict],
        graphics: List[Tuple[BoundingBox, str]],
    ) -> Tuple[BoundingBox, DetectionMethod]:
        """Maps a caption to the graphical region it most plausibly describes.

        Returns the detected region and the detection method used.
        """
        page_x0, page_top, page_x1, page_bottom = (float(v) for v in page.bbox)

        # The caption's horizontal extent (slightly widened) approximates the
        # column the figure lives in — crucial for two-column papers.
        column = BoundingBox(
            x0=max(page_x0, caption_bbox.x0 - 15.0),
            top=page_top,
            x1=min(page_x1, caption_bbox.x1 + 15.0),
            bottom=page_bottom,
        )

        # Captions usually sit below their figure, so search "above" first.
        for direction in ("above", "below"):
            band = self._content_band(page, caption_bbox, column, lines, direction)
            if band is None:
                continue
            detected = self._detect_in_band(page, band, graphics)
            if detected is not None:
                return detected

        # Fallback: no graphic could be associated with this caption. Use a
        # fixed-height window above the caption, or below when the caption sits
        # too close to the top of the page.
        top = max(page_top, caption_bbox.top - self.fallback_region_height_pt)
        fallback = BoundingBox(
            x0=column.x0,
            top=top,
            x1=column.x1,
            bottom=max(top + 1.0, caption_bbox.top - 2.0),
        )
        if fallback.height < self.min_figure_height_pt:
            bottom = min(
                page_bottom, caption_bbox.bottom + self.fallback_region_height_pt
            )
            fallback = BoundingBox(
                x0=column.x0,
                top=min(caption_bbox.bottom + 2.0, bottom - 1.0),
                x1=column.x1,
                bottom=bottom,
            )
        logger.debug(
            "Using caption-relative fallback region on page %d", page.page_number
        )
        return fallback, "caption-fallback"

    def _content_band(
        self,
        page: "pdfplumber.page.Page",
        caption_bbox: BoundingBox,
        column: BoundingBox,
        lines: List[_LineDict],
        direction: str,
    ) -> Optional[BoundingBox]:
        """Computes the vertical band where the figure could live.

        The band spans from the caption edge to the nearest *paragraph-like*
        text line in the given direction (or the page margin). Short text lines
        are ignored on purpose: axis tick labels, legend entries, and in-plot
        annotations are text too, and treating them as boundaries would
        truncate the figure. A line only counts as a boundary when it is at
        least half the caption's width and overlaps the caption column.

        Returns the search band, or ``None`` if the available space is too
        small to contain a plausible figure.
        """
        page_x0, page_top, page_x1, page_bottom = (float(v) for v in page.bbox)
        min_paragraph_width = 0.5 * max(caption_bbox.width, 1.0)

        if direction == "above":
            boundary = page_top
            for line in lines:
                line_box = _line_bbox(line)
                if line_box.bottom > caption_bbox.top - 1.0:
                    continue  # not strictly above the caption
                if line_box.width < min_paragraph_width:
                    continue  # short line — likely part of the figure itself
                if line_box.horizontal_overlap(column) <= 0.0:
                    continue  # belongs to the other column of the layout
                boundary = max(boundary, line_box.bottom)
            band = BoundingBox(
                x0=column.x0, top=boundary, x1=column.x1, bottom=caption_bbox.top - 1.0
            )
        else:
            boundary = page_bottom
            for line in lines:
                line_box = _line_bbox(line)
                if line_box.top < caption_bbox.bottom + 1.0:
                    continue  # not strictly below the caption
                if line_box.width < min_paragraph_width:
                    continue
                if line_box.horizontal_overlap(column) <= 0.0:
                    continue
                boundary = min(boundary, line_box.top)
            band = BoundingBox(
                x0=column.x0,
                top=caption_bbox.bottom + 1.0,
                x1=column.x1,
                bottom=boundary,
            )

        if band.height < 25.0:
            return None
        return band

    def _detect_in_band(
        self,
        page: "pdfplumber.page.Page",
        band: BoundingBox,
        graphics: List[Tuple[BoundingBox, str]],
    ) -> Optional[Tuple[BoundingBox, DetectionMethod]]:
        """Looks for a figure-sized graphic inside a search band.

        The region is the union of *every* raster image and vector drawing
        primitive found in the band, so a multi-panel figure composed of many
        separate objects is captured whole. The band is already bounded by
        paragraph-like text and the caption, which keeps the union from
        swallowing surrounding prose.

        Returns ``(region, method)`` when a plausible figure region is found,
        otherwise ``None``. The method is ``"embedded-image"`` when at least
        one raster contributed, else ``"vector-cluster"``.
        """
        page_x0 = float(page.bbox[0])
        page_x1 = float(page.bbox[2])
        page_width = page_x1 - page_x0

        candidates: List[BoundingBox] = []
        has_raster = False
        for box, kind in graphics:
            if not (band.top <= box.vertical_center <= band.bottom):
                continue
            if box.horizontal_overlap(band) <= 0.0:
                continue
            if kind == "image":
                candidates.append(box)
                has_raster = True
            else:
                # Skip page-wide thin rules (header/footer/section separators).
                if box.width > 0.9 * page_width and box.height < 3.0:
                    continue
                candidates.append(box)

        if not candidates:
            return None

        union = candidates[0]
        for box in candidates[1:]:
            union = union.union(box)
        if (
            union.width < self.min_figure_width_pt
            or union.height < self.min_figure_height_pt
        ):
            return None

        clipped = union.clamped(page_x0, band.top, page_x1, band.bottom)
        return clipped, ("embedded-image" if has_raster else "vector-cluster")

    # ------------------------------------------------------------------ #
    # Graphics collection & rendering                                     #
    # ------------------------------------------------------------------ #

    def _collect_graphics(
        self, page: "pdfplumber.page.Page"
    ) -> List[Tuple[BoundingBox, str]]:
        """Collects every graphic object on the page with a coarse kind tag.

        Returns ``(bbox, kind)`` pairs where ``kind`` is ``"image"`` for
        embedded rasters and ``"vector"`` for drawing primitives.
        """
        graphics: List[Tuple[BoundingBox, str]] = []
        for obj in page.images or []:
            box = self._object_bbox(obj)
            if box is not None:
                graphics.append((box, "image"))
        for attr in ("rects", "lines", "curves"):
            for obj in getattr(page, attr, None) or []:
                box = self._object_bbox(obj)
                if box is not None:
                    graphics.append((box, "vector"))
        return graphics

    @staticmethod
    def _object_bbox(obj: Dict[str, Any]) -> Optional[BoundingBox]:
        """Safely builds a bounding box from a pdfplumber object dict.

        Returns the box, or ``None`` if the object lacks usable coordinates.
        """
        try:
            return BoundingBox(
                x0=float(obj["x0"]),
                top=float(obj["top"]),
                x1=float(obj["x1"]),
                bottom=float(obj["bottom"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _render_region(
        self, page: "pdfplumber.page.Page", region: BoundingBox
    ) -> Tuple[Optional[Image.Image], int]:
        """Renders a page region into an in-memory high-resolution RGB image.

        Returns ``(image, dpi)`` where ``image`` is ``None`` if rendering
        failed or the region was degenerate, and ``dpi`` is the effective
        resolution used (possibly reduced for very large regions — see
        ``max_render_edge_px``).
        """
        page_x0, page_top, page_x1, page_bottom = (float(v) for v in page.bbox)
        clamped = region.padded(self.bbox_padding_pt).clamped(
            page_x0, page_top, page_x1, page_bottom
        )
        if clamped.width < 4.0 or clamped.height < 4.0:
            logger.warning(
                "Degenerate crop region on page %d — skipping", page.page_number
            )
            return None, self.resolution

        # A region of W x H points renders to (W * dpi / 72) x (H * dpi / 72)
        # pixels. Cap the longest edge so a full-page region cannot allocate a
        # multi-hundred-megabyte bitmap.
        longest_pt = max(clamped.width, clamped.height)
        dpi = self.resolution
        if longest_pt * dpi / 72.0 > self.max_render_edge_px:
            dpi = max(72, int(self.max_render_edge_px * 72.0 / longest_pt))
            logger.debug(
                "Reducing render DPI to %d for a %.0f pt region on page %d",
                dpi,
                longest_pt,
                page.page_number,
            )

        try:
            cropped = page.crop((clamped.x0, clamped.top, clamped.x1, clamped.bottom))
            rendered = cropped.to_image(resolution=dpi).original.convert("RGB")
        except Exception:
            logger.exception("Failed to render crop on page %d", page.page_number)
            return None, dpi
        return rendered, dpi
