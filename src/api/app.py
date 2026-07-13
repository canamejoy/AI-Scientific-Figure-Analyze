"""Streamlit viewer for the figure extractor.

Run from the repository root with::

    streamlit run src/api/app.py

Upload a scientific PDF → see every extracted figure with its caption and
detection metadata → optionally export the dataset to disk. No vision model
is involved; extraction is fully deterministic.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import streamlit as st

# Make the project root importable so absolute "src.*" imports resolve when
# the app is launched via "streamlit run src/api/app.py".
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import ExtractedFigure  # noqa: E402
from src.pipeline import Orchestrator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Scientific Figure Extractor", layout="wide")


def _orchestrator() -> Orchestrator:
    """Returns the per-session orchestrator, creating it on demand."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = Orchestrator()
        st.session_state.processed_digest = None
    return st.session_state.orchestrator


def _process_upload(uploaded_file) -> None:
    """Runs extraction on a newly uploaded PDF (hashed to avoid re-processing)."""
    payload = uploaded_file.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    if st.session_state.get("processed_digest") == digest:
        return

    orchestrator = _orchestrator()
    tmp_path: Optional[Path] = None
    try:
        with st.spinner("Detecting and rendering figures…"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(payload)
                tmp_path = Path(tmp.name)
            orchestrator.process_document(str(tmp_path))
    except Exception as exc:
        logger.exception("Failed to process uploaded PDF")
        st.error(f"Could not process this PDF: {exc}")
        return
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    st.session_state.processed_digest = digest
    st.session_state.paper_name = Path(uploaded_file.name).stem
    count = len(orchestrator.figures)
    if count:
        st.success(f"Extracted {count} figure(s).")
    else:
        st.warning("No figures were detected in this document.")


def _render_figures(figures: List[ExtractedFigure]) -> None:
    """Renders every extracted figure with its caption and metadata."""
    for index, figure in enumerate(figures, start=1):
        st.markdown(f"### {figure.label} · page {figure.page_number}")
        image_col, meta_col = st.columns([3, 2], gap="large")
        with image_col:
            st.image(figure.image, width="stretch")
        with meta_col:
            st.markdown("**Caption**")
            st.write(figure.caption)
            st.caption(
                f"`{figure.figure_id}` · detection `{figure.detection_method}` · "
                f"{figure.dpi} DPI · {figure.image.width}×{figure.image.height} px"
            )
        st.divider()


def main() -> None:
    """Entry point executed on every Streamlit rerun."""
    orchestrator = _orchestrator()

    st.title("Scientific Figure Extractor")
    st.caption("Extract complete figures and their captions from a scientific PDF.")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded is not None:
        _process_upload(uploaded)

    figures = orchestrator.figures
    if not figures:
        st.info("Upload a paper to extract its figures.")
        return

    with st.sidebar:
        st.header("Export")
        default_dir = str(_PROJECT_ROOT / "dataset")
        out_dir = st.text_input("Output directory", value=default_dir)
        if st.button("Export dataset", width="stretch"):
            try:
                paper_dir = orchestrator.export(
                    out_dir, paper_name=st.session_state.get("paper_name")
                )
                st.success(f"Dataset written to {paper_dir}")
            except Exception as exc:
                logger.exception("Export failed")
                st.error(f"Export failed: {exc}")

    _render_figures(figures)


main()
