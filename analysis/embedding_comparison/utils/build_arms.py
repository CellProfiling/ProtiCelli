"""The arms Fig 3A, 3B and the U2OS ladder (Fig 3C, 3D) read, and their shared loaders.

Heavy: reads the 1.14M-cell SubCell input h5ad (7 GB) and the per-line generated shards. Nothing
in the plot scripts depends on this file.

    python build_arms.py build "<atlas_name>" <samp_seed>     # one (line, seed) arm -> ARMS_DIR

Inputs are taken from the environment (defaults are the lab's paths): NORM_V2 (generated shards,
location map, floor caches), INPUT_H5AD (SubCell embeddings of the HPA single-cell images, one h5ad),
ARMS_DIR (output). Both embedding sets come from running the images through SubCell
(https://github.com/CellProfiling/SubCellPortable).

An ARM is one (cell line, sampling seed): the real per-protein mean over the HPA cells that pass
the SubCell paper's filters, and a simulated mean over the same number of cells per protein, stored as
arms_published_<stem>_samp<seed>.npz (genes, real, gen, n_cells) + .info.json.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperlib import ATLAS, LINES, stem  # noqa: E402

# ------------------------------------------------------------------ external inputs
NORM_V2 = Path(os.environ.get("NORM_V2", "/scratch/groups/emmalu/konstantin/subcell/norm_v2"))
INPUT_H5AD = os.environ.get("INPUT_H5AD", "/scratch/groups/emmalu/konstantin/subcell/published_input/"
                                          "HPA_embedding_harmonized_microscope_plateID.h5ad")
ARMS_DIR = os.environ.get("ARMS_DIR", "/scratch/groups/emmalu/konstantin/proticelli/paper_v2/arms")
# gene -> {atlas_name: locations}: the HPA subcellular location annotation per gene and cell line
# (downloadable from proteinatlas.org, "Subcellular location data"), pickled as a nested dict.
LOCATION_MAP = NORM_V2 / "annotations" / "location_map.pkl"
BLOCK = 65536                                                  # X read block, rows


def gen_dir(line: str) -> str:
    return str(NORM_V2 / LINES[line])


def arms_paths(line: str, samp_seed: int) -> tuple[str, str]:
    base = f"{ARMS_DIR}/arms_published_{stem(line)}_samp{samp_seed}"
    return f"{base}.npz", f"{base}.info.json"


def load_arms(line: str, samp_seed: int):
    """(genes, real, gen, n_cells) for one (line, sampling seed)."""
    npz, _ = arms_paths(line, samp_seed)
    with np.load(npz, allow_pickle=False) as z:
        return z["genes"].astype(str), z["real"], z["gen"], z["n_cells"]


def load_location_map() -> dict:
    with open(LOCATION_MAP, "rb") as fh:
        return pickle.load(fh)


def dominant_labels(genes, line: str, loc_map: dict, none: str = "unlabelled") -> np.ndarray:
    """First-listed HPA location per (gene, line) -- HPA's own main location; `none` if absent."""
    key = ATLAS[line]
    out = np.full(len(genes), none, dtype=object)
    for i, g in enumerate(genes):
        entry = loc_map.get(g) or loc_map.get(str(g).replace("+", ","))
        if not isinstance(entry, dict):
            continue
        locs = entry.get(key)
        if isinstance(locs, str):
            locs = [x.strip() for x in locs.split(",") if x.strip()]
        if locs:
            out[i] = str(list(locs)[0]).strip()
    return out.astype(str)


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


# ------------------------------------------------------------------ the HPA cell selection (SubCell paper's filters)
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


# ------------------------------------------------------------------ stage A: the real arm
def stage_a(line: str):
    """The HPA cells of one line that pass the SubCell paper's filters, then the mean over them per
    protein. Arms are keyed on gene symbol; symbols shared by two ensembl ids are dropped so
    joins are 1:1."""
    import h5py
    log = {"line": line, "gen_dir": gen_dir(line)}
    with h5py.File(INPUT_H5AD, "r") as f:
        obs, rows, counts = cohort(f, line)
        log.update(counts)
        X = stream_x(f, rows)

    key = (obs["atlas_name"].astype(str) + "_" + obs["ensembl_ids"].astype(str)).to_numpy()
    obs["atlas_name_ensembl_ids"] = key
    df = pd.DataFrame(X)
    df["atlas_name_ensembl_ids"] = key
    grouped = df.groupby("atlas_name_ensembl_ids").mean()
    del df, X
    keys_sorted = grouped.index.to_numpy()
    Xagg = grouped.values.astype(np.float32)
    del grouped
    gene_name = (obs.drop_duplicates(subset="atlas_name_ensembl_ids", keep="first")
                 .set_index("atlas_name_ensembl_ids").reindex(keys_sorted)["gene_names"].astype(str).to_numpy())
    n_cells = obs.groupby("atlas_name_ensembl_ids").size().reindex(keys_sorted).to_numpy()
    log["n_proteins_after_aggregation"] = int(len(keys_sorted))
    vc = pd.Series(gene_name).value_counts()
    dup = set(vc[vc > 1].index)
    log["n_symbol_collisions"] = int(len(dup))
    if dup:
        k = ~np.isin(gene_name, list(dup))
        Xagg, gene_name, n_cells = Xagg[k], gene_name[k], n_cells[k]
    log["n_proteins_symbol_unique"] = int(len(gene_name))
    o = np.argsort(gene_name, kind="stable")                # symbol order = the canonical arm order
    return dict(genes=gene_name[o], X_agg=Xagg[o], n_cells=n_cells[o].astype(np.int64)), log


def index_generated(line: str, real_genes, log: dict):
    """Which generated shard rows belong to each protein of this cohort."""
    import h5py
    real_set = set(real_genes.tolist())
    per, off, blocks = {}, 0, []
    for path in sorted(glob.glob(f"{gen_dir(line)}/*.h5")):
        with h5py.File(path, "r") as f:
            g = f["obs/gene_name"][:].astype(str)
        blocks.append([path, int(off), int(len(g))])
        for i, name in enumerate(g):
            if name in real_set:
                per.setdefault(name, []).append(off + i)
        off += len(g)
    assert blocks, f"{line}: no generated shards under {gen_dir(line)}"
    log.update(gen_cells_scanned=int(off), gen_shards=len(blocks), gen_proteins_present=int(len(per)),
               common_proteins=int(len(set(per) & real_set)))
    return {g: np.asarray(per[g], np.int64) for g in sorted(per)}, blocks


def build(line: str, samp: int) -> None:
    """One arm. The depth-matched draw takes, per protein, exactly as many generated cells as the
    real mean used, without replacement from default_rng(samp); rng construction and call order
    are load-bearing for reproducing the shipped arms."""
    import h5py
    t0 = time.time()
    real, log = stage_a(line)
    real_genes, Xr = real["genes"], real["X_agg"]
    real_n = dict(zip(real_genes.tolist(), real["n_cells"].tolist()))
    print(f"{line}: stage A done in {time.time()-t0:.0f}s, {len(real_genes)} proteins", flush=True)
    rows_of, blocks = index_generated(line, real_genes, log)
    genes = np.array(sorted(set(real_genes.tolist()) & set(rows_of)))
    real_pos = {g: i for i, g in enumerate(real_genes.tolist())}
    R = np.stack([Xr[real_pos[g]] for g in genes]).astype(np.float32)

    rng = np.random.default_rng(samp)
    take, short = {}, []
    for g in genes:
        avail, want = np.asarray(rows_of[g]), int(real_n[g])
        if want >= len(avail):
            take[g] = avail
            if want > len(avail):
                short.append(dict(gene=str(g), wanted=want, available=int(len(avail))))
        else:
            take[g] = rng.choice(avail, size=want, replace=False)
    union = np.unique(np.concatenate([take[g] for g in genes]))
    buf = np.zeros((len(union), 1536), np.float32)
    for path, o, n in blocks:                                 # one read per shard
        lo, hi = np.searchsorted(union, o), np.searchsorted(union, o + n)
        if hi <= lo:
            continue
        loc = union[lo:hi] - o
        with h5py.File(path, "r") as f:                       # h5py wants increasing indices
            buf[lo:hi] = f["X"][np.sort(loc)][np.argsort(np.argsort(loc))]
    pos = {int(r): i for i, r in enumerate(union.tolist())}
    got = np.array([len(take[g]) for g in genes], np.int64)
    wanted = np.array([int(real_n[g]) for g in genes], np.int64)
    G = np.stack([buf[[pos[int(r)] for r in take[g]]].mean(0) for g in genes]).astype(np.float32)

    out, info_p = arms_paths(line, samp)
    os.makedirs(ARMS_DIR, exist_ok=True)
    tmp = out + f".tmp{os.getpid()}"
    np.savez_compressed(tmp, genes=genes, real=R, gen=G, n_cells=got)
    os.replace(tmp + ".npz", out)
    info = dict(line=line, samp_seed=samp, n_proteins=int(len(genes)), n_cells=int(got.sum()),
                median_cells=float(np.median(got)), exact_match=int((wanted == got).sum()),
                n_shortfall=len(short), shortfalls=short[:50], n_real_proteins=int(len(real_genes)),
                n_real_only=int(len(set(real_genes.tolist()) - set(rows_of))),
                real_cells_aggregated=int(sum(real_n.values())),
                published_recipe_cohort={k: log[k] for k in [
                    "n_cells_all_lines", "initial_cells", "after_drop_no_location",
                    "after_drop_multigene", "after_drop_uncertain", "after_best_antibody",
                    "n_proteins_after_aggregation"]},
                n_symbol_collisions=log["n_symbol_collisions"],
                n_proteins_symbol_unique=log["n_proteins_symbol_unique"],
                gen_dir=log["gen_dir"], gen_cells_scanned=log["gen_cells_scanned"],
                gen_proteins_present=log["gen_proteins_present"], input_h5ad=INPUT_H5AD)
    tmp = info_p + f".tmp{os.getpid()}"
    json.dump(info, open(tmp, "w"), indent=1)
    os.replace(tmp, info_p)
    print(f"wrote {out} n_proteins={len(genes)} n_cells={int(got.sum())} "
          f"exact={int((wanted == got).sum())} short={len(short)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] != ["build"] or len(sys.argv) != 4:
        sys.exit(__doc__)
    build(sys.argv[2], int(sys.argv[3]))
