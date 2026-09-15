"""Fig 3C and 3D: the generated hierarchy tree, the subcluster UMAPs (levels 2-4) and the AMI ladder.

U2OS only. Seven levels of Leiden subclustering (targets 1, 2, 6, 14, 22, 38, 86 clusters; per-level
neighbourhoods 125, 100, 90, 55, 40, 25, 10; euclidean; the tuner picks the maximum-modularity
resolution among target-attaining candidates) were run for both arms at 3 sampling x 10 clustering
seeds, and for a real-vs-real split-half ceiling at 3 splits x 10 seeds (`panelCD_cluster.py`: one
JSON per run in `data/{ladders,ceiling}/`, per-protein labels in `data/labels/`, the assembled
tables `data/published_*.csv`, the representative run's renumbered labels from `pick`, the GO terms
per node from `enrich`, the Fig 3D coordinates from `panel-d-umap`). This script draws from those.

| section | output |
|---|---|
| 1 | Fig 3C: `figures/F_tree_generated.{svg,png}`, `F_tree.pick.json` |
| 2 | Fig 3D left: `figures/F_panelD_level{2,3,4}.{svg,png}` |
| 3 | Fig 3D right: `figures/F_ladder.{svg,png}` + `F_ladder.source.csv` |
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "utils")]
from paperlib import setup
setup(threads=4)       # SVG hash salt, SOURCE_DATE_EPOCH, BLAS threads -- before matplotlib is imported

import json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
import paperlib as pl, treelib as tl
from hier import TARGETS
from panelCD_cluster import LABELS, MAX_LEVELS, RES

warnings.filterwarnings("ignore")
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
pick = json.load(open(RES / "cluster_renumbering.json"))
SS, CS = pick["samp_seed"], pick["clust_seed"]
raw_real, rrc = tl.load_cluster_labels(LABELS / f"labels_samp{SS}_seed{CS}_real.tsv.gz")
raw_gen, rgc = tl.load_cluster_labels(LABELS / f"labels_samp{SS}_seed{CS}_gen.tsv.gz")
_, raw_gen = tl.align_datasets(raw_real, raw_gen, rrc, rgc)
df_real, real_cols = tl.load_cluster_labels(LABELS / f"labels_samp{SS}_seed{CS}_real.renumbered.tsv.gz")
df_gen, gen_cols = tl.load_cluster_labels(LABELS / f"labels_samp{SS}_seed{CS}_gen.renumbered.tsv.gz")
df_real, df_gen = tl.align_datasets(df_real, df_gen, real_cols, gen_cols)

# ------------------------------------------------------------------------------
# 1. Fig 3C: the generated hierarchy tree
# ------------------------------------------------------------------------------
# The representative run (`pick`: the real arm's level-2 minority fraction closest to the median
# over the 30 runs) and its renumbered labels. The enrichment cache holds the g:Profiler result
# per node keyed by (level, RAW cluster id), so it is remapped through the same old->new map.
# Node size 400 + n proteins, colour the -log10 p of the top GO:CC term (GO:MF + GO:BP at the
# finest level). The layout is pinned to the raw-id geometry: `dot` derives x positions partly from
# node insertion order, so the raw-id graph is laid out once and its positions, keyed by
# (level, gene set), are reused for the renumbered tree.

pl.rc_reset()
m = pd.read_csv(RES / "cluster_renumbering_map.csv")
m = m[m.arm == "generated"]
key = {(int(r.level), str(r.old_cluster)): int(r.new_cluster) for r in m.itertuples()}
src_cache = pickle.loads((RES / "enrichment" / "enrichment_cache_generated.pkl.rawids").read_bytes())
enrich = {(lvl, key.get((lvl, str(cl)), cl)): val for (lvl, cl), val in src_cache.items()}
assert len(enrich) == len(src_cache), "cache remap collided keys"

G_raw, _ = tl.hierarchy_from_labels(raw_gen, rgc, MAX_LEVELS)
pinned = tl.pin_positions(G_raw, tl.dot_layout(G_raw, 1, 0.7))
G, _ = tl.hierarchy_from_labels(df_gen, gen_cols, MAX_LEVELS)
tl.render_circle_tree(G, enrich, FIG / "F_tree_generated", pos=tl.pinned_layout(G, pinned))
json.dump(dict(line="U2OS", samp_seed=SS, clust_seed=CS, selection_rule=pick["selection_rule"],
               level2_minority_frac=pick["level2_minority_frac"], level2_minority_frac_median=pick["level2_minority_frac_median"],
               n_runs_considered=pick["n_runs_considered"], gen_columns=gen_cols, n_genes=int(len(df_gen)),
               enrichment="g:Profiler GO:CC (GO:MF+GO:BP deepest)"),
          open(FIG / "F_tree.pick.json", "w"), indent=1)

# ------------------------------------------------------------------------------
# 2. Fig 3D left: subcluster UMAPs, levels 2-4
# ------------------------------------------------------------------------------
# Two SEPARATE UMAP fits (one per arm, per-arm centred, k = 125, seed 42) were cached by
# `panelCD_cluster.py panel-d-umap` in `data/umap_cache/` together with the gene list, in sorted
# gene order (the order `align_datasets` gives the labels). For display the generated map is
# rotated/reflected/translated onto the real one by orthogonal Procrustes, then scaled per axis so
# the robust widths match (UMAP output scale is arbitrary per fit); generated cluster ids are
# matched to real ones by maximum overlap (Hungarian). The clustering shown is computed in 1,536-D
# and never from these coordinates.

pl.rc_reset()
cache = RES / "umap_cache"
genes = (cache / "gene_names.txt").read_text().split()
assert genes == list(df_real["gene_name"]), "umap cache gene order != aligned labels; rerun panel-d-umap"
UR = np.load(cache / "umap_coords_real.npy")
UG = np.load(cache / "umap_coords_generated.npy")


def align_umaps(source, target):
    """Procrustes (rotation/reflection/translation) of source onto target, then one scale factor
    per axis so the 0.5-99.5 percentile widths match."""
    tm, sm = target.mean(axis=0), source.mean(axis=0)
    R, _ = orthogonal_procrustes(source - sm, target - tm)
    out = (source - sm) @ R + tm
    lo_s, hi_s = np.percentile(out, (0.5, 99.5), axis=0); lo_t, hi_t = np.percentile(target, (0.5, 99.5), axis=0)
    c = out.mean(axis=0)
    return (out - c) * (hi_t - lo_t) / (hi_s - lo_s) + c


def match_clusters(ref_labels, mov_labels):
    """Relabel mov to maximise overlap with ref (Hungarian on the contingency table)."""
    ru = sorted(set(ref_labels)); mu = sorted(set(mov_labels))
    C = np.zeros((len(mu), len(ru)), dtype=np.int64)
    for i, mm in enumerate(mu):
        sel = mov_labels == mm
        for j, r in enumerate(ru):
            C[i, j] = int((ref_labels[sel] == r).sum())
    ri, ci = linear_sum_assignment(-C)
    remap = {mu[i]: ru[j] for i, j in zip(ri, ci)}
    nxt = len(ru)
    for mm in mu:
        if mm not in remap:
            remap[mm] = f"x{nxt}"; nxt += 1
    return np.asarray([remap[mm] for mm in mov_labels])


UG = align_umaps(UG, UR)
both = np.vstack([UR, UG])
lo, hi = both.min(0), both.max(0)
pad = 0.04 * (hi - lo)
XLIM, YLIM = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])
for level_i in range(1, 4):                      # levels 2..4
    rc, gc = real_cols[level_i], gen_cols[level_i]
    rl = df_real[rc].astype(str).to_numpy()
    gl = match_clusters(rl, df_gen[gc].astype(str).to_numpy())
    tl.subcluster_2x2(genes, UR, UG, rl, gl, level_i + 1, rc, gc, FIG / f"F_panelD_level{level_i + 1}", XLIM, YLIM)
    print(f"  level {level_i + 1}: k_real={len(set(rl))} k_gen={len(set(gl))}")

# ------------------------------------------------------------------------------
# 3. Fig 3D right: the AMI ladder
# ------------------------------------------------------------------------------
# AMI per level, observed (real vs generated, blue) and the split-half ceiling (real vs real,
# green), mean +- 1 SD over the 30 runs each. Every plotted value is written to the source CSV;
# the observed / ceiling fraction the text quotes is in data/published_ladder_with_ceiling.csv.

pl.rc_reset()
pl.apply_figure_style(sizes=(9, 8, 7))
C_CEI, C_OBS = "#047857", "#1D4ED8"
LV = list(range(2, 8))
runs = pd.read_csv(RES / "published_ladder_u2os_runs.csv")
cei = pd.read_csv(RES / "published_ceiling_u2os.csv")

src = []
fig, ax = plt.subplots(figsize=(4.6, 4.3))
for df, col, lab in [(cei, C_CEI, "real vs real (split-half ceiling), ±1 SD"),
                     (runs, C_OBS, "real vs generated (observed), ±1 SD")]:
    g = df[df.level.isin(LV)].groupby("level")["AMI"].agg(["mean", "std", "size"]).reset_index()
    ax.fill_between(g.level, g["mean"] - g["std"], g["mean"] + g["std"], color=col, alpha=.25, lw=0, zorder=2)
    ax.plot(g.level, g["mean"], "-o", color=col, ms=4, lw=1.8, label=lab, zorder=3)
    src += [dict(series=lab, level=int(r.level), target=TARGETS[int(r.level) - 1], mean=r["mean"], sd=r["std"], n_runs=int(r["size"]))
            for _, r in g.iterrows()]
ax.set_xticks(LV); ax.set_xticklabels([f"{l}\n({TARGETS[l-1]})" for l in LV])
ax.set_xlim(1.8, 7.4); ax.set_ylim(0, 1.06); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_xlabel("hierarchy level (target clusters)"); ax.set_ylabel("AMI"); ax.grid(axis="y", alpha=.15, lw=.6)
ax.legend(frameon=False, loc="lower left", fontsize=7.5, labelspacing=.4, handlelength=1.9, borderaxespad=.3)
fig.tight_layout()
pl.save_figure(fig, FIG / "F_ladder", dpi=300)
pd.DataFrame(src).to_csv(FIG / "F_ladder.source.csv", index=False)
