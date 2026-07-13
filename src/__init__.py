"""Scientific figure extractor.

Turns a scientific PDF into a structured dataset of complete figures and
their captions — no LLM involved. A downstream analysis agent receives each
figure whole, at high resolution, with its full caption.

Modules:

* ``src.parsing``    — figure/caption detection (DocLayout-YOLO + heuristic
  fallback) and high-resolution rendering.
* ``src.extraction`` — the dataset exporter and CLI.
* ``src.pipeline``   — a thin facade for the viewer.
* ``src.api``        — the Streamlit viewer.
"""
