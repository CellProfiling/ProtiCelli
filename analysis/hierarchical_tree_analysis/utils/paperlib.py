"""Shared helpers of the hierarchical_tree_analysis scripts: environment, figure style + saving.
One of four sibling copies (one per analysis folder, each folder self-contained); the function
bodies are identical across copies, each copy keeps only what its folder uses.

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


def rc_reset() -> None:
    """Back to matplotlib's defaults (as if a fresh process), keeping the fixed SVG hash salt."""
    import matplotlib
    matplotlib.rc_file_defaults()
    matplotlib.rcParams["svg.hashsalt"] = SVG_HASHSALT


# ------------------------------------------------------------------ figure style + saving
def apply_figure_style(*, frame="open", sizes=(8, 7, 6), dpi=200, bbox="tight") -> None:
    """rcParams for publication output (font-size ladder, outward ticks, frameless legends,
    300-dpi save, Type-42 fonts). frame: 'open' | 'boxed'."""
    import matplotlib as mpl
    base, secondary, tick = sizes
    boxed = frame == "boxed"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": base,
        "axes.labelsize": base, "axes.titlesize": base, "legend.fontsize": secondary,
        "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.grid": False, "legend.frameon": False,
        "figure.dpi": dpi, "savefig.dpi": 300, "savefig.bbox": bbox,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.labelweight": "normal",
        "lines.linewidth": 1.2, "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


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
