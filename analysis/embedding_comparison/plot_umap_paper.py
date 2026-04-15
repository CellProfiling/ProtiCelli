# -*- coding: utf-8 -*-
"""
Publication-ready UMAP plot (2x3 layout) for protein embeddings.

Creates a clean 2-row x 3-column figure:
  Row 1: Colored by cell line  (Combined | Real | Generated)
  Row 2: Colored by location   (Combined | Real | Generated)

Uses sc.pl.umap() identically to compare_embeddings_visual.py,
then strips titles/labels/ticks/spines for publication readiness.

Author: Konstantin Kahnert
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'  # Keep text as text objects in SVG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from embedding_utils import (
    filter_and_aggregate_real_embeddings_by_cell_line,
    load_multiple_embeddings,
    load_unnormalized_real_embeddings,
    assign_primary_location,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Color palettes (copied from compare_embeddings_visual.py:888-920)
# ------------------------------------------------------------------

def build_palettes(cell_lines_sorted):
    """Build all palette dicts, identical to the original script."""
    tab20 = list(plt.cm.tab20.colors)
    tab20b = list(plt.cm.tab20b.colors)
    extended_palette = tab20 + [tab20b[14], tab20b[15], tab20b[18], tab20b[19]]

    color_map_by_cell_line_generated = {}
    color_map_by_cell_line_real = {}
    color_map_by_cell_line_type = {}

    for idx, cell_line in enumerate(cell_lines_sorted):
        color_map_by_cell_line_generated[cell_line] = extended_palette[idx * 2 + 1]
        color_map_by_cell_line_real[cell_line] = extended_palette[idx * 2]
        color_map_by_cell_line_type[f"{cell_line} (Generated)"] = extended_palette[idx * 2 + 1]
        color_map_by_cell_line_type[f"{cell_line} (Real)"] = extended_palette[idx * 2]

    return color_map_by_cell_line_real, color_map_by_cell_line_generated, color_map_by_cell_line_type


def _strip_axes(ax):
    """Remove titles, labels, ticks, and spines."""
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Publication-ready 2x3 UMAP plot for protein embeddings"
    )
    parser.add_argument("--real-embedding", required=True)
    parser.add_argument("--hpa-csv", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--cell-lines", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--save-svg", action="store_true", help="Also save plots as SVG")
    parser.add_argument(
        "--replot", action="store_true",
        help="Skip recomputation; load cached h5ad files and replot only."
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_path = Path(args.output_path)
    cell_lines = args.cell_lines

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cache paths (saved alongside output)
    cache_dir = output_path.parent / "umap_paper_cache"
    cache_combined = cache_dir / "combined.h5ad"
    cache_real = cache_dir / "real.h5ad"
    cache_gen = cache_dir / "gen.h5ad"
    cache_cell_lines = cache_dir / "cell_lines_sorted.txt"

    logger.info("Publication-ready UMAP (2x3)")
    logger.info("=" * 50)

    if args.replot:
        # ==================================================================
        # Fast path: load pre-computed data
        # ==================================================================
        for p in [cache_combined, cache_real, cache_gen, cache_cell_lines]:
            if not p.exists():
                logger.error("Cache file missing: %s", p)
                logger.error("Run without --replot first to compute and cache the data.")
                sys.exit(1)
        logger.info("Loading cached AnnData objects (--replot mode)...")
        combined_adata = sc.read_h5ad(cache_combined)
        real_adata_global = sc.read_h5ad(cache_real)
        gen_adata_global = sc.read_h5ad(cache_gen)
        cell_lines_sorted = cache_cell_lines.read_text().splitlines()
        logger.info("   Combined: %s, Real: %s, Gen: %s",
                     combined_adata.shape, real_adata_global.shape, gen_adata_global.shape)
        logger.info("   Cell lines: %s", cell_lines_sorted)
    else:
        # ==================================================================
        # 1. Load data — identical to compare_embeddings_visual.py main()
        # ==================================================================
        logger.info("Loading real embeddings...")
        real_adata_raw = load_unnormalized_real_embeddings(Path(args.real_embedding))

        real_embeddings = {}
        for cl in cell_lines:
            try:
                real_embeddings[f"Real_{cl}"] = (
                    filter_and_aggregate_real_embeddings_by_cell_line(
                        real_adata_raw, cl, args.hpa_csv
                    )
                )
            except Exception as e:
                logger.error("Failed real for %s: %s", cl, e)

        if not real_embeddings:
            logger.error("No real embeddings loaded - aborting.")
            sys.exit(1)

        logger.info("Loading generated embeddings...")
        gen_paths = {}
        for cl in cell_lines:
            p = base_dir / cl / "regenerated" / "aggregated" / "embeddings_aggregated.h5ad"
            if p.exists():
                gen_paths[cl] = p
            else:
                logger.warning("Missing generated for %s: %s", cl, p)
        generated_embeddings = load_multiple_embeddings(gen_paths)

        # ==================================================================
        # 2. Tag and combine
        # ==================================================================
        logger.info("Building combined AnnData...")
        all_embeddings = {}

        for name, adata in real_embeddings.items():
            adata_copy = adata.copy()
            adata_copy.obs["embedding_type"] = "Real"
            adata_copy.obs["cell_line"] = name.replace("Real_", "")
            all_embeddings[name] = adata_copy

        for name, adata in generated_embeddings.items():
            adata_copy = adata.copy()
            adata_copy.obs["embedding_type"] = "Generated"
            adata_copy.obs["cell_line"] = name
            all_embeddings[name] = adata_copy

        # ==================================================================
        # 3. Combined UMAP
        # ==================================================================
        logger.info("Computing global UMAP for all embeddings...")
        combined_adata = sc.concat(
            list(all_embeddings.values()), join="outer", fill_value=0
        )
        combined_adata.obs_names_make_unique()

        combined_adata.obs["cell_line_type"] = (
            combined_adata.obs["cell_line"]
            + " ("
            + combined_adata.obs["embedding_type"]
            + ")"
        )

        sc.pp.neighbors(combined_adata, n_neighbors=25, n_pcs=150)
        sc.tl.umap(combined_adata, random_state=42)

        # ==================================================================
        # 4. Subset
        # ==================================================================
        real_mask = combined_adata.obs["embedding_type"] == "Real"
        real_adata_global = combined_adata[real_mask].copy()

        gen_mask = combined_adata.obs["embedding_type"] == "Generated"
        gen_adata_global = combined_adata[gen_mask].copy()

        # ==================================================================
        # 5. Location mapping
        # ==================================================================
        logger.info("Mapping locations from real to generated embeddings...")

        gene_col = (
            "gene_name"
            if "gene_name" in real_adata_global.obs.columns
            else "gene_names"
        )

        gene_to_location = dict(
            zip(real_adata_global.obs[gene_col], real_adata_global.obs["locations"])
        )

        if gene_col in gen_adata_global.obs.columns:
            gen_adata_global.obs["locations"] = gen_adata_global.obs[gene_col].map(
                gene_to_location
            )
        else:
            gen_adata_global.obs["locations"] = gen_adata_global.obs.index.map(
                gene_to_location
            )

        gen_adata_global.obs["locations"] = gen_adata_global.obs["locations"].fillna(
            "Unknown"
        )

        mapped_count = (gen_adata_global.obs["locations"] != "Unknown").sum()
        logger.info("   Mapped locations to %d/%d generated proteins",
                     mapped_count, len(gen_adata_global))

        real_adata_global.obs["location_label"] = real_adata_global.obs[
            "locations"
        ].apply(assign_primary_location)
        gen_adata_global.obs["location_label"] = gen_adata_global.obs["locations"].apply(
            assign_primary_location
        )

        combined_adata.obs["location_label"] = "Unknown"
        combined_adata.obs.loc[real_adata_global.obs.index, "location_label"] = (
            real_adata_global.obs["location_label"]
        )
        combined_adata.obs.loc[gen_adata_global.obs.index, "location_label"] = (
            gen_adata_global.obs["location_label"]
        )

        # Cell lines sorted — derived from successfully loaded real embeddings
        cell_lines_sorted = sorted(
            set(name.replace("Real_", "") for name in real_embeddings.keys())
        )

        # ==================================================================
        # Save cache for --replot
        # ==================================================================
        logger.info("Saving cache to %s ...", cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        def _safe_write(adata, path):
            """Write h5ad, coercing any non-string object columns to str."""
            a = adata.copy()
            for col in a.obs.columns:
                if a.obs[col].dtype == object:
                    a.obs[col] = a.obs[col].astype(str)
            a.write_h5ad(path)

        _safe_write(combined_adata, cache_combined)
        _safe_write(real_adata_global, cache_real)
        _safe_write(gen_adata_global, cache_gen)
        cache_cell_lines.write_text("\n".join(cell_lines_sorted))
        logger.info("   Cache saved.")

    # Palettes — uses cell_lines_sorted set above (either loaded or freshly derived)
    color_map_real, color_map_gen, color_map_type = build_palettes(cell_lines_sorted)

    # ==================================================================
    # 6. Plot 2x4 — cols 0-2: UMAPs (no legends), col 3: legends only
    # ==================================================================
    logger.info("Plotting 2x4 figure...")

    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    # Reorder combined so real points are drawn on top of generated.
    # Two mechanisms needed:
    #   1. Obs order (generated first, real last): controls z-order for location_label,
    #      where both types share the same category names.
    #   2. Category order on cell_line_type (generated categories first, real last):
    #      scanpy iterates over categories and draws each as a separate scatter call,
    #      so the last category listed ends up on top regardless of obs order.
    gen_idx = combined_adata.obs.index[combined_adata.obs["embedding_type"] == "Generated"]
    real_idx = combined_adata.obs.index[combined_adata.obs["embedding_type"] == "Real"]
    combined_plot = combined_adata[list(gen_idx) + list(real_idx)].copy()

    # Set category order: all (Generated) before all (Real)
    gen_cats = [f"{cl} (Generated)" for cl in cell_lines_sorted]
    real_cats = [f"{cl} (Real)" for cl in cell_lines_sorted]
    combined_plot.obs["cell_line_type"] = pd.Categorical(
        combined_plot.obs["cell_line_type"],
        categories=gen_cats + real_cats,
        ordered=False,
    )

    # Compute midpoint dot size for real points (meet-in-the-middle between
    # combined default and separate-real default, so real points look consistent)
    n_combined = len(combined_plot)
    n_real = len(real_adata_global)
    size_combined_default = 120000 / n_combined
    size_real_separate_default = 120000 / n_real
    size_real_mid = (size_combined_default + size_real_separate_default) / 2

    # Size arrays for combined plots: real points → mid size, generated → default
    real_mask_combined = (combined_plot.obs["embedding_type"] == "Real").values
    combined_sizes = np.where(real_mask_combined, size_real_mid, size_combined_default)

    logger.info("   Dot sizes - combined default: %.1f, separate-real default: %.1f, "
                "mid (applied to real): %.1f",
                size_combined_default, size_real_separate_default, size_real_mid)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        width_ratios=[1, 1, 1, 0.45],
        hspace=0.02, wspace=0.0,
    )

    # UMAP axes (cols 0-2)
    umap_axes = []
    for r in range(2):
        row = []
        for c in range(3):
            row.append(fig.add_subplot(gs[r, c]))
        umap_axes.append(row)

    # Legend axes (col 3)
    leg_ax0 = fig.add_subplot(gs[0, 3])
    leg_ax1 = fig.add_subplot(gs[1, 3])
    leg_ax0.axis("off")
    leg_ax1.axis("off")

    # --- Row 0: Colored by cell line (no legends on UMAP panels) ---
    sc.pl.umap(combined_plot, color="cell_line_type", palette=color_map_type,
               size=combined_sizes, ax=umap_axes[0][0], show=False, legend_loc=None)
    sc.pl.umap(real_adata_global, color="cell_line", palette=color_map_real,
               size=size_real_mid, ax=umap_axes[0][1], show=False, legend_loc=None)
    sc.pl.umap(gen_adata_global, color="cell_line", palette=color_map_gen,
               ax=umap_axes[0][2], show=False, legend_loc=None)

    # --- Row 1: Colored by location (no legends on UMAP panels) ---
    sc.pl.umap(combined_plot, color="location_label",
               size=combined_sizes, ax=umap_axes[1][0], show=False, legend_loc=None)
    sc.pl.umap(real_adata_global, color="location_label",
               size=size_real_mid, ax=umap_axes[1][1], show=False, legend_loc=None)
    sc.pl.umap(gen_adata_global, color="location_label",
               ax=umap_axes[1][2], show=False, legend_loc=None)

    # --- Strip decorations and rasterize scatter points ---
    for row in umap_axes:
        for ax in row:
            _strip_axes(ax)
            for coll in ax.collections:
                coll.set_rasterized(True)

    # --- Cell line legend (col 3, row 0) — 2 columns ---
    handles_cl = []
    for cl in cell_lines_sorted:
        handles_cl.append(Line2D(
            [0], [0], marker="o", color="w", linewidth=0, markersize=6,
            markerfacecolor=color_map_type[f"{cl} (Real)"],
            label=f"{cl} (Real)",
        ))
        handles_cl.append(Line2D(
            [0], [0], marker="o", color="w", linewidth=0, markersize=6,
            markerfacecolor=color_map_type[f"{cl} (Generated)"],
            label=f"{cl} (Generated)",
        ))
    leg_ax0.legend(
        handles=handles_cl, loc="center", frameon=False,
        fontsize=7, ncol=2, handletextpad=0.3, columnspacing=1.0,
    )

    # --- Location legend (col 3, row 1) — 1 column ---
    present_locs = set(combined_plot.obs["location_label"].values)
    handles_loc = []
    for loc in ["Nucleoplasm", "Nucleoli", "Nuclear speckles", "Nuclear membrane",
                 "Nucleus", "Cytosol", "Cytoskeleton", "Mitochondria",
                 "Endomembrane system", "Plasma membrane", "Vesicles",
                 "Mitotic structures", "Cytokinetic bridge", "Multi-localizing",
                 "Unknown"]:
        if loc not in present_locs:
            continue
        # Use scanpy's auto-assigned color from one of the plotted panels
        color = None
        for a in [combined_plot, real_adata_global, gen_adata_global]:
            key = "location_label_colors"
            if key in a.uns:
                cats = list(a.obs["location_label"].cat.categories) if hasattr(
                    a.obs["location_label"], "cat") else sorted(a.obs["location_label"].unique())
                if loc in cats:
                    color = a.uns[key][cats.index(loc)]
                    break
        if color is None:
            color = (0.7, 0.7, 0.7)
        handles_loc.append(Line2D(
            [0], [0], marker="o", color="w", linewidth=0, markersize=6,
            markerfacecolor=color, label=loc,
        ))
    leg_ax1.legend(
        handles=handles_loc, loc="center", frameon=False,
        fontsize=8, ncol=1, handletextpad=0.3,
    )

    # ==================================================================
    # 7. Save
    # ==================================================================
    logger.info("Saving to %s ...", output_path)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    logger.info("Saved PNG: %s", output_path)
    if args.save_svg:
        svg_path = output_path.with_suffix(".svg")
        plt.savefig(svg_path, dpi=args.dpi, bbox_inches="tight")
        logger.info("Saved SVG: %s", svg_path)
    # plt.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
