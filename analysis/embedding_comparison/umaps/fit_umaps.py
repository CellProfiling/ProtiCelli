"""Panel A, compute side: one joint UMAP per cell line over the stacked real/generated arms, plus
the high-dimensional 15-NN statistics (arm separation, location purity). Writes one statistics
CSV and one coordinate NPZ per line into umaps/data/panelA_parts/, which plot_panel_A.py draws from.

    python fit_umaps.py "<atlas_name>"          # one line (ARMS_DIR from the environment)

Recipe: n_neighbors=125, min_dist=0.1, euclidean, seed 42, no PCA, each arm centred on its own
column mean BEFORE stacking (a translation per arm: it removes the encoder offset between arms and
cannot move points within an arm). Statistics are computed in the full 1,536-D space with two
label sets: the first-listed HPA location as is ("dominant", 35 locations) and collapsed onto the
13 display groups ("grouped", the panel colouring).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "utils")]
from paperlib import SHORT                                                         # noqa: E402
from build_arms import dominant_labels, load_arms, load_location_map  # noqa: E402

PARTS = Path(__file__).resolve().parent / "data" / "panelA_parts"
SAMP = 0                       # panel A shows one sampling seed
K, MIN_DIST, METRIC, SEED = 125, 0.1, "euclidean", 42
KNN_K = 15
NONE = "Unknown"               # label of a protein without a location; excluded from purity
# The display grouping (35 HPA locations -> 13 groups) the panel colours by.
LOCATION_MAPPING = {
    "Actin filaments": "Cytoskeleton", "Aggresome": "Cytosol",
    "Centriolar satellite": "Cytoskeleton", "Centrosome": "Cytoskeleton",
    "Cytokinetic bridge": "Cytokinetic bridge", "Cytoplasmic bodies": "Cytosol",
    "Cytosol": "Cytosol", "Focal adhesion sites": "Cytoskeleton",
    "Intermediate filaments": "Cytoskeleton", "Microtubules": "Cytoskeleton",
    "Microtubule ends": "Cytoskeleton", "Midbody": "Cytokinetic bridge",
    "Midbody ring": "Cytokinetic bridge", "Mitochondria": "Mitochondria",
    "Mitotic spindle": "Mitotic structures", "Rods & Rings": "Mitotic structures",
    "Nuclear bodies": "Nucleus", "Nuclear membrane": "Nuclear membrane",
    "Nuclear speckles": "Nuclear speckles", "Nucleoli": "Nucleoli",
    "Nucleoli fibrillar center": "Nucleoli", "Nucleoli rim": "Nucleoli",
    "Nucleoplasm": "Nucleoplasm", "Cell Junctions": "Plasma membrane",
    "Endoplasmic reticulum": "Endomembrane system", "Endosomes": "Vesicles",
    "Golgi apparatus": "Endomembrane system", "Lipid droplets": "Vesicles",
    "Lysosomes": "Vesicles", "Peroxisomes": "Vesicles",
    "Plasma membrane": "Plasma membrane", "Vesicles": "Endomembrane system",
}


# ------------------------------------------------------------------ statistics kernel
def knn_same_group_fraction(P, group, k=KNN_K) -> float:
    """Mean fraction of each point's k nearest neighbours (self excluded) sharing its group.
    Stacked arms + arm indicator = arm separation; one arm + location labels = purity."""
    from sklearn.neighbors import NearestNeighbors
    P = np.ascontiguousarray(np.asarray(P, np.float32))
    idx = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(P).kneighbors(P, return_distance=False)
    keep = np.empty((len(P), k), dtype=int)
    for i, row in enumerate(idx):                  # drop self by identity, not by position
        r = row[row != i][:k]
        if len(r) < k:
            r = np.concatenate([r, row[: k - len(r)]])
        keep[i] = r
    return float((group[keep] == group[:, None]).mean())


def chance(nR, nG) -> float:
    n = nR + nG
    return float((nR / n) * ((nR - 1) / (n - 1)) + (nG / n) * ((nG - 1) / (n - 1)))


def arm_sep(R, G) -> float:
    arm = np.r_[np.zeros(len(R), int), np.ones(len(G), int)]
    return knn_same_group_fraction(np.vstack([R, G]), arm)


def purity(P, labels) -> float:
    ok = labels != NONE
    return float("nan") if ok.sum() < KNN_K + 5 else knn_same_group_fraction(P[ok], labels[ok])


def joint_umap(real, gen):
    """One embedding over the stacked, per-arm centred arms; returns the two row blocks."""
    import umap
    X = np.ascontiguousarray(np.vstack([real - real.mean(0), gen - gen.mean(0)]).astype(np.float32))
    xy = umap.UMAP(n_neighbors=K, min_dist=MIN_DIST, metric=METRIC, random_state=SEED).fit_transform(X)
    return xy[:len(real)], xy[len(real):]


# ------------------------------------------------------------------ one line
def fit(atlas: str) -> None:
    t0 = time.time()
    PARTS.mkdir(parents=True, exist_ok=True)
    line = SHORT[atlas]
    genes, R, G, _ = load_arms(atlas, SAMP)
    R, G = np.asarray(R, np.float32), np.asarray(G, np.float32)
    lab_dom = dominant_labels(genes, line, load_location_map(), none=NONE)
    lab_grp = np.array([LOCATION_MAPPING.get(x, NONE) for x in lab_dom])
    Rc, Gc = R - R.mean(0), G - G.mean(0)
    n = len(genes)
    row = dict(line=line, n_genes=n, chance=round(chance(n, n), 4), arm_separation=round(arm_sep(Rc, Gc), 4),
               purity_real_grouped=round(purity(Rc, lab_grp), 4), purity_gen_grouped=round(purity(Gc, lab_grp), 4),
               purity_real_dominant=round(purity(Rc, lab_dom), 4), purity_gen_dominant=round(purity(Gc, lab_dom), 4),
               n_labelled_grouped=int((lab_grp != NONE).sum()), n_labelled_dominant=int((lab_dom != NONE).sum()),
               knn_k=KNN_K, umap_n_neighbors=K, umap_min_dist=MIN_DIST, umap_seed=SEED)
    rxy, gxy = joint_umap(R, G)
    pd.DataFrame([row]).to_csv(PARTS / f"panelA_{line}.csv", index=False)
    np.savez_compressed(PARTS / f"coords_{line}.npz", real=rxy.astype(np.float32), gen=gxy.astype(np.float32),
                        genes=genes, location_grouped=lab_grp)
    print(f"{line}: n={n} arm_sep={row['arm_separation']} purity(grouped) real={row['purity_real_grouped']} "
          f"gen={row['purity_gen_grouped']} in {round(time.time() - t0, 1)}s", flush=True)


if __name__ == "__main__":
    fit(sys.argv[1])
