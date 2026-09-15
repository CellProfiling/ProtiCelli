"""Reactome sub-pathway dotplot (Fig 5G): Hep-G2, Metabolism of lipids (R-HSA-556833).

Do the organelle-defined spatial clusters of a Hep-G2 cell differ in the lipid-metabolism
sub-pathways their proteins occupy? For every Reactome sub-pathway two levels below the root and
every cluster, the mean log2 spatial enrichment of the pathway's proteins is tested against zero
(one-sided Wilcoxon signed-rank, BH-corrected over all pairs); pairs with BH p < 0.05 are drawn.
`reactome.py` holds the hierarchy traversal, the statistics and the renderer.

| input | role |
|---|---|
| `inputs/spatial_enrichment.csv` | protein-level spatial enrichment s_{k,g} per cluster (Gene, Cluster_1 .. Cluster_11), from the pixel-level spatial clustering (Methods) |
| `inputs/Hep-G2_clustered_cell_{009,028}.png` | the two clustered cell images stacked beside the plot |
| `data/reactome_cache.json` | the ContentService responses, fetched on first run |

The cluster annotations below are the expert organelle assignment of the 11 clusters (Methods,
"Organelle annotation"), with each cluster's colour in the cluster maps.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "utils")]
from paperlib import setup
setup(threads=1)       # SVG hash salt, SOURCE_DATE_EPOCH, BLAS threads -- before matplotlib is imported

import reactome as R

INPUTS, DATA, FIG = HERE / "inputs", HERE / "data", HERE / "figures"
ROOT, DEPTH = "R-HSA-556833", 2
ANNOTATION = {           # cluster -> (colour name in the cluster maps, organelle label)
    1: ("red", "weak plasma membrane"), 2: ("blue", "Nucleus Speckles"), 3: ("green", "Cytosol, Cytoskeleton"),
    4: ("yellow", "Unassigned"), 5: ("magenta", "Mitochondria, ER"), 6: ("cyan", "Mitochondria, ER"),
    7: ("orange", "Golgi, Intermediate filaments"), 8: ("purple", "Nucleolus Speckles"),
    9: ("teal", "weak Golgi apparatus"), 10: ("sky blue", "Nucleolus"), 11: ("maroon", "Nuclear lamina"),
}

table = R.dotplot_table(ROOT, DEPTH, INPUTS / "spatial_enrichment.csv", DATA / "reactome_cache.json")
sig = table[table.pval < 0.05]
print(f"{table.stId.nunique()} sub-pathways x {len(table.attrs['cluster_cols'])} clusters = {len(table)} tests; "
      f"{len(sig)} significant pairs over {sig.stId.nunique()} sub-pathways")
assert len(sig), "no (sub-pathway, cluster) pair passes BH p < 0.05; nothing to draw"

FIG.mkdir(exist_ok=True)
R.make_dotplot(sig, table.attrs["cluster_cols"], FIG / "F_fig_5G_dotplot", title=f"Metabolism of lipids ({ROOT}) in Hep-G2",
               annotation={k: {"color": R.COLOR_RGB[c], "label": lab} for k, (c, lab) in ANNOTATION.items()},
               img_paths=[INPUTS / "Hep-G2_clustered_cell_009.png", INPUTS / "Hep-G2_clustered_cell_028.png"])
