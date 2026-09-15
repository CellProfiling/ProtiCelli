"""The arms of the embedding comparison, as the tree analysis reads them, and the published-recipe
cohort behind the split-half ceiling. Verbatim copies of the same functions in
embedding_comparison/utils/build_arms.py (and `stem` from its paperlib): every analysis folder is
self-contained, so this folder carries what it needs of the arm builder rather than importing it.

An ARM is one (cell line, sampling seed): the real per-protein mean over the published-recipe
cohort and a depth-matched generated mean over the same proteins, stored as
arms_published_<stem>_samp<seed>.npz (genes, real, gen, n_cells) in ARMS_DIR by the embedding
comparison's `arms` step. INPUT_H5AD is the published SubCell HPA embedding h5ad.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

INPUT_H5AD = os.environ.get("INPUT_H5AD", "/scratch/groups/emmalu/konstantin/subcell/published_input/"
                                          "HPA_embedding_harmonized_microscope_plateID.h5ad")
ARMS_DIR = os.environ.get("ARMS_DIR", "/scratch/groups/emmalu/konstantin/proticelli/paper_v2/arms")
BLOCK = 65536                                                  # X read block, rows


def stem(atlas: str) -> str:
    """Filesystem-safe atlas_name: 'BJ [Human fibroblast]' -> 'BJ__Human_fibroblast_'."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in atlas)


def arms_paths(line: str, samp_seed: int) -> tuple[str, str]:
    base = f"{ARMS_DIR}/arms_published_{stem(line)}_samp{samp_seed}"
    return f"{base}.npz", f"{base}.info.json"


def load_arms(line: str, samp_seed: int):
    """(genes, real, gen, n_cells) for one (line, sampling seed)."""
    npz, _ = arms_paths(line, samp_seed)
    with np.load(npz, allow_pickle=False) as z:
        return z["genes"].astype(str), z["real"], z["gen"], z["n_cells"]


def read_cat(f, col):
    """Materialise an .obs column, categorical or plain, as an object/np array."""
    import h5py
    o = f[f"obs/{col}"]
    if isinstance(o, h5py.Group):
        cats = o["categories"][:]
        if cats.dtype.kind in "SO":
            cats = cats.astype(str)
        codes = o["codes"][:]
        out = np.empty(len(codes), dtype=object)
        nn = codes >= 0
        out[nn] = cats[codes[nn]]
        out[~nn] = None                     # NaN category -> None (locations NA lives here)
        return out
    v = o[:]
    return v.astype(str) if v.dtype.kind in "SO" else v


# ------------------------------------------------------------------ the published-recipe cohort
def cohort(f, line: str):
    """The SubCell paper's own preprocessing (its notebook, cells 2-8), for one line: the rows of
    the open input h5ad `f` that survive, their obs, and a count per step.

    Drop cells without a location, multi-gene antibodies and Uncertain reliability; keep ONE best
    antibody per (atlas_name, ensembl_ids) by reliability rank -- with a STABLE sort on
    [reliability_rank, antibody], the one deliberate deviation from the notebook's unstable
    quicksort."""
    log = {"n_cells_all_lines": int(f["X"].shape[0])}
    atlas = read_cat(f, "atlas_name")
    m = (atlas == line)
    log["initial_cells"] = int(m.sum())
    assert log["initial_cells"] > 0, f"{line!r}: no cells with atlas_name == {line!r}"
    rows = np.flatnonzero(m)
    obs = pd.DataFrame({c: read_cat(f, c)[rows] for c in (
        "atlas_name", "antibody", "ensembl_ids", "gene_names", "locations",
        "gene_reliability", "reliability_rank")})
    for step, keep in (("after_drop_no_location", lambda o: o["locations"].notna()),
                       ("after_drop_multigene", lambda o: ~o["gene_names"].astype(str).str.contains(",")),
                       ("after_drop_uncertain", lambda o: o["gene_reliability"] != "Uncertain")):
        k = keep(obs).to_numpy()
        obs, rows = obs[k].reset_index(drop=True), rows[k]
        log[step] = int(len(obs))
    best = obs.sort_values(by=["reliability_rank", "antibody"], kind="stable")
    best = best.drop_duplicates(subset=["atlas_name", "ensembl_ids"], keep="first")
    k = obs["antibody"].isin(best["antibody"]).to_numpy()
    obs, rows = obs[k].reset_index(drop=True), rows[k]
    log["after_best_antibody"] = int(len(obs))
    return obs, rows, log


def stream_x(f, rows):
    """X[rows] of the open h5ad, read in contiguous blocks (rows ascending)."""
    n_all, n_feat = f["X"].shape
    X = np.empty((len(rows), n_feat), np.float32)
    p = 0
    for lo in range(0, n_all, BLOCK):
        hi = min(lo + BLOCK, n_all)
        sel = rows[(rows >= lo) & (rows < hi)]
        if len(sel):
            X[p:p + len(sel)] = f["X"][lo:hi][sel - lo]
            p += len(sel)
    assert p == len(rows), (p, len(rows))
    return X
