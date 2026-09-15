"""Shared helpers of the spatial_clustering scripts: environment, figure saving. One of four
sibling copies (one per analysis folder, each folder self-contained); the function bodies are
identical across copies, each copy keeps only what its folder uses.

Import order matters. Call `setup()` FIRST, before numpy or matplotlib are imported anywhere in
the process: it pins the BLAS thread count (reduction order changes results) and fixes matplotlib's
SVG hash salt + SOURCE_DATE_EPOCH (otherwise every rerun rewrites SVG ids and embedded dates and a
byte comparison cannot tell "nothing moved" from "a number changed").
"""
import os
import re
from pathlib import Path

SVG_HASHSALT = "protvl-paper-v2"          # seeds matplotlib's SVG element ids (random per process otherwise)


# ------------------------------------------------------------------ environment
def setup(threads: int = 1) -> None:
    """Deterministic figure emission + BLAS width. Must run before matplotlib/numpy import."""
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")          # 2024-01-01
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(threads)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = SVG_HASHSALT


# ------------------------------------------------------------------ figure saving
def save_figure(fig, stem, dpi=300, bbox="tight", close=True) -> list[Path]:
    """The one output convention: stem.svg with real, Illustrator-editable <text> (no glyph paths,
    explicit font properties) and stem.png at `dpi`. `stem` may carry a .png/.svg suffix, which is
    dropped. Returns [png, svg]; optionally closes the figure."""
    import matplotlib.pyplot as plt
    stem = Path(stem)
    if stem.suffix in (".png", ".svg"):
        stem = stem.with_suffix("")
    svg, png = stem.with_suffix(".svg"), stem.with_suffix(".png")
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(svg, bbox_inches=bbox)
    _explicit_font_props(svg)
    fig.savefig(png, dpi=dpi, bbox_inches=bbox)
    if close:
        plt.close(fig)
    return [png, svg]


_SHORTHAND = re.compile(r"font:\s*([\d.]+)px\s*([^;\"]+)")


def _explicit_font_props(svg_path) -> None:
    """Expand matplotlib's `font: <size>px <stack>` shorthand into explicit properties, so the
    SVG stays editable in Illustrator."""
    src = open(svg_path).read()
    out = _SHORTHAND.sub(lambda m: f"font-size: {m.group(1)}px; font-family: {m.group(2).strip()}", src)
    open(svg_path, "w").write(out)
