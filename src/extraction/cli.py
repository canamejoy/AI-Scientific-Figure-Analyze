"""Command-line interface: extract figures + captions from a PDF.

Examples::

    # Extract into ./dataset/<pdf-stem>/
    python -m src.extraction.cli paper.pdf

    # Custom output directory and DPI
    python -m src.extraction.cli paper.pdf -o out --dpi 400

    # Force the caption-anchor heuristics (skip the layout model)
    python -m src.extraction.cli paper.pdf --no-layout-model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.extraction.exporter import DatasetExporter
from src.parsing.pdf_parser import ScientificPDFParser


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.extraction.cli",
        description="Extract figures and their captions from a scientific PDF.",
    )
    parser.add_argument("pdf", type=Path, help="Path to the scientific PDF.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("dataset"),
        help="Dataset output directory (default: ./dataset).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Dataset folder name (default: the PDF's file stem).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render resolution for extracted figures (default: 300).",
    )
    parser.add_argument(
        "--no-layout-model",
        action="store_true",
        help="Skip DocLayout-YOLO; use caption-anchor heuristics only.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def main(argv: list = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code (0 on success).
    """
    args = build_arg_parser().parse_args(argv)
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    logger = logging.getLogger("extraction.cli")

    # None → respect the LAYOUT_DETECTION env var; --no-layout-model forces off.
    parser = ScientificPDFParser(
        resolution=args.dpi,
        use_layout_model=False if args.no_layout_model else None,
    )
    try:
        document = parser.parse(args.pdf)
        paper_dir = DatasetExporter(args.output).export(document, paper_name=args.name)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Extraction failed")
        return 1

    logger.info(
        "Done — %d figure(s) at %s", len(document.figures), paper_dir
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
