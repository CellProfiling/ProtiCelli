"""Shared helpers of the embedding_comparison scripts: environment, figure style + saving, cohort
constants. One of four sibling copies (one per analysis folder, each folder self-contained); the
function bodies are identical across copies, each copy keeps only what its folder uses.

Import order matters. Call `setup()` FIRST, before numpy or matplotlib are imported anywhere in
the process: it pins the BLAS thread count (reduction order changes results) and fixes matplotlib's
SVG hash salt + SOURCE_DATE_EPOCH (otherwise every rerun rewrites SVG ids and embedded dates and a
byte comparison cannot tell "nothing moved" from "a number changed").
"""
import os
import re
from pathlib import Path

SVG_HASHSALT = "protvl-paper-v2"          # seeds matplotlib's SVG element ids (random per process otherwise)
RT = dict(float_precision="round_trip")   # pandas' default float parser is not correctly rounded


# ------------------------------------------------------------------ environment
def setup(threads: int = 1) -> None:
    """Deterministic figure emission + BLAS width. Must run before matplotlib/numpy import."""
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")          # 2024-01-01
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[v] = str(threads)
    import matplotlib
    matplotlib.use("Agg")
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


# ------------------------------------------------------------------ cohort constants
# HPA atlas_name -> Proteome2Cell generated-image directory (relative to NORM_V2; see build_arms.py)
LINES = {
    "U2OS": "u2os_p2c", "A-431": "p2c_full/A-431", "U-251MG": "p2c_full/U-251MG",
    "MCF-7": "p2c_full/MCF-7", "HEK293": "p2c_full/HEK293", "PC-3": "p2c_full/PC-3",
    "Rh30": "p2c_full/Rh30", "HeLa": "p2c_full/HeLa", "SH-SY5Y": "p2c_full/SH-SY5Y",
    "CACO-2": "p2c_full/CACO-2", "Hep-G2": "p2c_full/Hep-G2", "BJ [Human fibroblast]": "p2c_full/BJ",
}
SAMP_SEEDS = (0, 1, 2)          # generated-cell sampling seeds (depth-matched draw)
SHORT = {atlas: atlas.split(" ")[0] for atlas in LINES}      # atlas_name -> short label ('BJ')
ATLAS = {short: atlas for atlas, short in SHORT.items()}      # short label -> atlas_name


def stem(atlas: str) -> str:
    """Filesystem-safe atlas_name: 'BJ [Human fibroblast]' -> 'BJ__Human_fibroblast_'."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in atlas)


def canon_gene(g: str) -> str:
    """Annotations spell multi-gene antibody targets with ',' and embeddings with '+'."""
    return str(g).replace(",", "+").replace(" ", "")
