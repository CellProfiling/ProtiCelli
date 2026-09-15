# Reactome sub-pathway dotplot

Do the organelle-defined spatial clusters of a Hep-G2 cell differ in the lipid-metabolism
sub-pathways their proteins occupy? For every Reactome sub-pathway two levels below
Metabolism of lipids (R-HSA-556833) and every cluster, the mean log2 spatial enrichment of the
pathway's proteins is tested against zero (one-sided Wilcoxon signed-rank, BH-corrected over all
pairs); pairs with BH p < 0.05 are drawn. Produces Fig 5G.

The pixel-level spatial clustering that defines the clusters and the per-protein enrichment
table is `spatial_clustering.ipynb` in this folder.

## Run

```bash
PY=/path/to/python bash run_spatial_clustering.sh
```

One step: `plot_dotplot.py`. `reactome.py` holds the hierarchy traversal through the Reactome
ContentService, the statistics and the renderer.

## Inputs

| input | file | used by |
|---|---|---|
| protein-level spatial enrichment s_{k,g} per cluster (Gene, Cluster_1 .. Cluster_11), from the pixel-level spatial clustering (Methods) | `inputs/spatial_enrichment.csv` (included) | statistics |
| the two clustered-cell images stacked beside the plot | `inputs/Hep-G2_clustered_cell_{009,028}.png` (included) | figure |
| the Reactome ContentService, fetched on first run and cached | `data/reactome_cache.json` | hierarchy and gene sets |

The cluster-to-organelle annotation is the expert assignment of the 11 clusters (Methods), set
at the top of `plot_dotplot.py`.

## Outputs

`figures/F_fig_5G_dotplot.{svg,png}`. The Reactome hierarchy moves between releases; 
delete the cache to refresh it, and expect the sub-pathway set to change slightly with the release.

## Environment

Python >= 3.10 with `requirements.txt`.
