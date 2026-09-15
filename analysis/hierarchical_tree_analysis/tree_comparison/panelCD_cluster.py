"""Fig 3C and 3D, compute side: the clustering runs behind the AMI ladder and its ceiling, the
representative run renumbered for the tree, and the Fig 3D UMAP fits. Heavy (the arms and the
7 GB input h5ad; the ladder and ceiling were run on Zen 3 with 4 threads). plot_panels_C_D.py
draws from what this writes.

    python panelCD_cluster.py halves --splits 0 1 2          # split-half real cohort -> HALVES_DIR/halves_split*.npz
    python panelCD_cluster.py ladder --samp S --clust C      # one observed run -> data/ladders/pub_sampS_seedC.json + labels
    python panelCD_cluster.py ceiling --split S --clust C    # one ceiling run  -> data/ceiling/cei_splitS_seedC.json
    python panelCD_cluster.py pick                           # ladder tables, the representative run, renumbered labels
    python panelCD_cluster.py enrich                         # g:Profiler GO terms per node of the generated tree (network) -> data/enrichment/
    python panelCD_cluster.py panel-d-umap                   # the two Fig 3D UMAPs + gene_names.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent / "utils")]
import hier as H                                                                  # noqa: E402
import treelib as tl                                                              # noqa: E402
from arms import INPUT_H5AD, arms_paths, cohort, load_arms, stream_x             # noqa: E402

RES = _HERE / "data"
LADDERS, CEILING, LABELS = RES / "ladders", RES / "ceiling", RES / "labels"
# The split halves are 3 x 109 MB of real embeddings: a compute-side input, so they live next to the arms.
HALVES = Path(os.environ.get("HALVES_DIR", "/scratch/groups/emmalu/konstantin/proticelli/paper_v2/halves"))
LINE = "U2OS"          # Fig 3C and 3D are U2OS only
N_GENES = 9533         # arm size, sampling-invariant
BLOCK = 65536
MAX_LEVELS = 7
UMAP_METRIC, UMAP_K, UMAP_MIN_DIST, UMAP_SEED = "euclidean", 125, 0.1, 42     # Fig 3D, separate fit per arm


def _score(labs, metas, keys):
    """Per-level AMI between the two label sets, plus each side's tuner record."""
    from sklearn.metrics import adjusted_mutual_info_score as AMI
    ka, kb = keys
    out = []
    for i, target in enumerate(H.TARGETS):
        col = "level%d" % (i + 1)
        a, b = labs[ka][col].to_numpy().astype(str), labs[kb][col].to_numpy().astype(str)
        out.append(dict(level=i + 1, target=target, AMI=float(AMI(a, b)),
                        **{ka: metas[ka]["records"][i], kb: metas[kb]["records"][i]}))
    return out


def _write_json(rec, out: Path):
    tmp = Path(f"{out}.tmp{os.getpid()}")
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, out)


def _done(out: Path) -> bool:
    """Idempotence: a cell whose JSON exists and parses is skipped."""
    try:
        json.load(open(out))
        return True
    except (OSError, ValueError):
        return False


# ------------------------------------------------------------------ split halves (the ceiling's input)
def build_halves(splits):
    """Two independent half-means per protein from the real U2OS cohort (the published recipe,
    `arms.cohort`), on the SAME 9,533 proteins as the observed ladder (the real-and-generated
    intersection the arms carry), so that fraction-of-ceiling divides like by like."""
    import h5py
    HALVES.mkdir(parents=True, exist_ok=True)
    todo = [s for s in splits if not (HALVES / f"halves_split{s}.npz").exists()]
    print("splits to build:", todo, flush=True)
    if not todo:
        return
    t0 = time.time()
    with h5py.File(INPUT_H5AD, "r") as h:
        obs, rows, trace = cohort(h, LINE)
        print("cohort trace:", json.dumps(trace), flush=True)
        X = stream_x(h, rows)
    print(f"X streamed: {X.shape} ({time.time()-t0:.0f}s)", flush=True)
    grp = obs.groupby("gene_names").indices
    real_genes = sorted(grp)
    arm_genes, _, _, _ = load_arms(LINE, 0)
    genes = sorted(set(real_genes) & set(arm_genes.astype(str)))
    assert len(genes) == N_GENES == len(arm_genes), (len(genes), N_GENES, len(arm_genes))
    nab = np.array([obs.antibody.iloc[grp[g]].nunique() for g in genes])
    info = dict(cohort="published_recipe", line=LINE, n_cells=int(len(obs)), n_proteins=len(genes),
                n_real_proteins=len(real_genes), n_real_only_dropped=len(real_genes) - len(genes),
                proteins_multi_antibody=int((nab > 1).sum()), max_antibodies_per_protein=int(nab.max()),
                cohort_trace=trace, gene_set="real AND generated (arms)")
    for S in todo:
        rng = np.random.default_rng(S)
        A = np.zeros((len(genes), X.shape[1]), np.float32); B = np.zeros_like(A)
        na = np.zeros(len(genes), np.int32); nb = np.zeros(len(genes), np.int32)
        for i, g in enumerate(genes):
            c = grp[g]; p = rng.permutation(len(c)); h_ = len(c) // 2
            A[i] = X[c[p[:h_]]].mean(0); B[i] = X[c[p[h_:]]].mean(0)
            na[i], nb[i] = h_, len(c) - h_
        tmp = HALVES / f"halves_split{S}.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, genes=np.array(genes), A=A, B=B, n_cells_A=na, n_cells_B=nb, n_antibodies=nab)
        os.replace(tmp, HALVES / f"halves_split{S}.npz")
        assert (na > 0).all(), f"split{S}: {int((na == 0).sum())} proteins have a single cell, so half A is empty"
        print(f"  split{S}: median cells/half {np.median(na):.0f}/{np.median(nb):.0f}", flush=True)
    json.dump(info, open(HALVES / "halves_info.json", "w"), indent=1)


# ------------------------------------------------------------------ one observed cell
def run_ladder(samp, clust):
    """The 7-level ladder on BOTH U2OS arms at one (sampling, clustering) seed, scored real vs gen."""
    out = LADDERS / f"pub_samp{samp}_seed{clust}.json"
    if _done(out):
        return out, "skip"
    t0 = time.time()
    src, _ = arms_paths(LINE, samp)
    genes, real, gen, _n = load_arms(LINE, samp)
    assert len(genes) == N_GENES, (len(genes), N_GENES)
    labs, metas, shift = {}, {}, {}
    for arm, A in (("real", real), ("gen", gen)):
        X = np.asarray(A, np.float32)
        mu = X.mean(0)                                        # per-arm centring
        shift[arm] = float(np.linalg.norm(mu))
        labs[arm], metas[arm] = H.run_arm((X - mu).astype(np.float32), clust)
    levels = _score(labs, metas, ("real", "gen"))
    LABELS.mkdir(parents=True, exist_ok=True)
    for arm in ("real", "gen"):
        d = labs[arm].rename(columns={"level%d" % (i + 1): "leiden_lvl%d_res%s" % (i, repr(float(metas[arm]["records"][i]["res"])))
                                      for i in range(len(H.TARGETS))})
        d.insert(0, "gene_name", genes)
        d.to_csv(LABELS / f"labels_samp{samp}_seed{clust}_{arm}.tsv.gz", sep="\t", index=False)
    _write_json(dict(kind="observed", cohort="published_recipe", line=LINE, samp_seed=samp, clust_seed=clust,
                     rule=H.RULE, n_genes=int(real.shape[0]), levels=levels, targets=H.TARGETS,
                     neighbors=H.NEIGHBORS, grid=metas["real"]["grid"],
                     leiden=metas["real"]["leiden"], centring_shift_norm=shift,
                     total_seconds=round(time.time() - t0, 1), arms_file=src), out)
    return out, "done"


# ------------------------------------------------------------------ one ceiling cell
def run_ceiling(split, clust):
    """Cluster half A and half B of one split with the identical pipeline; score A vs B."""
    out = CEILING / f"cei_split{split}_seed{clust}.json"
    if _done(out):
        return out, "skip"
    t0 = time.time()
    src = HALVES / f"halves_split{split}.npz"
    z = np.load(src, allow_pickle=True)
    labs, metas, shift = {}, {}, {}
    for half in ("A", "B"):
        X = np.asarray(z[half], np.float32)
        mu = X.mean(0)
        shift[half] = float(np.linalg.norm(mu))
        labs[half], metas[half] = H.run_arm((X - mu).astype(np.float32), clust)
    levels = _score(labs, metas, ("A", "B"))
    nab = int(np.asarray(z["n_antibodies"]).max())
    _write_json(dict(kind="ceiling", cohort="published_recipe", line=LINE, split_seed=split, clust_seed=clust,
                     rule=H.RULE, n_genes=int(z["A"].shape[0]), levels=levels, targets=H.TARGETS,
                     neighbors=H.NEIGHBORS, grid=metas["A"]["grid"], leiden=metas["A"]["leiden"],
                     centring_shift_norm=shift, same_antibody=bool(nab == 1), max_antibodies_per_protein=nab,
                     total_seconds=round(time.time() - t0, 1), halves_file=str(src)), out)
    return out, "done"


# ------------------------------------------------------------------ the tables and the representative run
def pick():
    """Assemble the 30 + 30 runs into the four ladder tables, choose the run Fig 3C and 3D show,
    and renumber its labels left to right.

    The representative run is chosen by a rule fixed in advance: the real arm's level-2 minority
    fraction closest to the median over all 30 runs, ties to the lower (sampling, clustering) seed
    pair. Its raw labels (both arms, aligned to the common genes) are renumbered so each level of
    the tree reads 0, 1, 2, ... left to right in the `dot` layout -- a pure relabelling (partition
    unchanged, asserted), iterated to a fixed point because node identity is (resolution, id) and
    renaming can itself nudge the layout. The raw label files are never modified; Fig 3C and 3D
    read the `.renumbered` copies, the ladder keeps reading the raw ones."""
    expected = [(s, c) for s in (0, 1, 2) for c in range(10)]
    arm_rows, run_rows, seen_lad = [], [], []
    for f in sorted(glob.glob(f"{LADDERS}/pub_*.json")):
        d = json.load(open(f))
        seen_lad.append((d["samp_seed"], d["clust_seed"]))
        for L in d["levels"]:
            run_rows.append(dict(samp_seed=d["samp_seed"], clust_seed=d["clust_seed"], level=L["level"], target=L["target"],
                                 AMI=L["AMI"], n_genes=d["n_genes"]))
            for arm in ("real", "gen"):
                r = L[arm]
                arm_rows.append(dict(samp_seed=d["samp_seed"], clust_seed=d["clust_seed"], arm=arm, level=L["level"],
                                     target=L["target"], res=r.get("res"), k=r.get("k"), status=r.get("status"),
                                     sizes=r.get("sizes"), minority=r.get("minority"), minority_frac=r.get("minority_frac"),
                                     modularity=r.get("modularity"), selector=r.get("selector"), n_attaining=r.get("n_attaining"),
                                     n_evals=r.get("n_evals"), seconds=r.get("seconds"),
                                     n_iterations=d["leiden"]["n_iterations"], rule=d["rule"], n_genes=d["n_genes"]))
    cei_rows, seen_cei = [], []
    for f in sorted(glob.glob(f"{CEILING}/cei_*.json")):
        d = json.load(open(f))
        seen_cei.append((d["split_seed"], d["clust_seed"]))
        for L in d["levels"]:
            cei_rows.append(dict(split_seed=d["split_seed"], clust_seed=d["clust_seed"], level=L["level"], target=L["target"],
                                 AMI=L["AMI"], same_antibody=d["same_antibody"],
                                 max_antibodies_per_protein=d["max_antibodies_per_protein"], n_genes=d["n_genes"]))
    for kind, seen in (("observed ladder", seen_lad), ("ceiling", seen_cei)):
        missing, extra = sorted(set(expected) - set(seen)), sorted(set(seen) - set(expected))
        assert not missing and not extra, f"{kind}: read {len(seen)}/30; missing={missing}; unexpected={extra}"
    arms, runs, cei = pd.DataFrame(arm_rows), pd.DataFrame(run_rows), pd.DataFrame(cei_rows)
    arms.to_csv(RES / "published_ladder_u2os.csv", index=False)
    runs.to_csv(RES / "published_ladder_u2os_runs.csv", index=False)
    cei.to_csv(RES / "published_ceiling_u2os.csv", index=False)

    rows = []       # per level: observed and ceiling AMI over the 30 runs each, and their ratio
    for lvl in sorted(runs.level.unique()):
        ro, co = runs[runs.level == lvl], cei[cei.level == lvl]
        d = dict(level=lvl, target=int(ro.target.iloc[0]))
        for tag, dd in (("obs", ro), ("cei", co)):
            v = dd["AMI"].to_numpy(float)
            d[f"{tag}_AMI_mean"], d[f"{tag}_AMI_std"], d[f"{tag}_AMI_count"] = v.mean(), v.std(ddof=1), len(v)
        d["frac_AMI"] = d["obs_AMI_mean"] / d["cei_AMI_mean"]
        aa = arms[arms.level == lvl]
        for arm in ("real", "gen"):
            s = aa[aa.arm == arm]
            d[f"n_runs_{arm}"] = int(len(s)); d[f"n_attained_{arm}"] = int((s.k == s.target).sum())
            d[f"n_criterion_{arm}"] = int((s.selector == "criterion").sum())
            d[f"n_single_attaining_{arm}"] = int((s.selector == "single_attaining").sum())
            d[f"n_shipped_fallback_{arm}"] = int((s.selector == "shipped_fallback").sum())
        rows.append(d)
    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "published_ladder_with_ceiling.csv", index=False)
    print(summ.round(4).to_string(index=False))

    arms = pd.read_csv(RES / "published_ladder_u2os.csv")     # re-read: the pick is made from the table as written
    l2 = arms[(arms.level == 2) & (arms.arm == "real")].copy()
    med = float(l2.minority_frac.median())
    l2["d"] = (l2.minority_frac - med).abs()
    chosen = l2.sort_values(["d", "samp_seed", "clust_seed"]).iloc[0]
    SS, CS = int(chosen.samp_seed), int(chosen.clust_seed)
    print(f"representative run: samp{SS} clust{CS} minority_frac={chosen.minority_frac:.4f} (median {med:.4f})")

    raw_real, rrc = tl.load_cluster_labels(LABELS / f"labels_samp{SS}_seed{CS}_real.tsv.gz")
    raw_gen, rgc = tl.load_cluster_labels(LABELS / f"labels_samp{SS}_seed{CS}_gen.tsv.gz")
    raw_real, raw_gen = tl.align_datasets(raw_real, raw_gen, rrc, rgc)
    map_rows, summary = [], {}
    for arm, df, cols in (("real", raw_real, rrc), ("generated", raw_gen, rgc)):
        work, summary[arm], rows = tl.renumber_left_to_right(df, cols, MAX_LEVELS, iter_max=8)
        map_rows += [dict(arm=arm, **r) for r in rows]
        out = LABELS / f"labels_samp{SS}_seed{CS}_{'real' if arm == 'real' else 'gen'}.renumbered.tsv.gz"
        tmp = out.with_name(out.name + ".tmp")
        work.to_csv(tmp, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})   # mtime 0: byte-stable reruns
        os.replace(tmp, out)
        print(f"{arm}: iterations={summary[arm]['iterations']} converged={summary[arm]['converged']} "
              f"all_levels_ascending={all(summary[arm]['levels_ascending'].values())}")
    pd.DataFrame(map_rows).to_csv(RES / "cluster_renumbering_map.csv", index=False)
    json.dump(dict(samp_seed=SS, clust_seed=CS, max_levels=MAX_LEVELS, iter_max=8, per_arm=summary,
                   rule="0..k-1 by increasing graphviz `dot` x position at each level",
                   selection_rule="level-2 real-arm minority_frac closest to the median over all runs; ties -> lower (samp, clust)",
                   level2_minority_frac=float(chosen.minority_frac), level2_minority_frac_median=med,
                   n_runs_considered=int(len(l2)), raw_labels_untouched=True),
              open(RES / "cluster_renumbering.json", "w"), indent=1)


# ------------------------------------------------------------------ GO labels for the panel C trees
def enrich():
    """The top GO term per node of the generated tree (Fig 3C), keyed by (level, RAW cluster id):
    data/enrichment/enrichment_cache_generated.pkl.rawids. Needs the g:Profiler web service."""
    chosen = json.load(open(RES / "cluster_renumbering.json"))
    SS, CS = chosen["samp_seed"], chosen["clust_seed"]
    raw_real, rrc = tl.load_cluster_labels(LABELS / f"labels_samp{SS}_seed{CS}_real.tsv.gz")
    raw_gen, rgc = tl.load_cluster_labels(LABELS / f"labels_samp{SS}_seed{CS}_gen.tsv.gz")
    _, raw_gen = tl.align_datasets(raw_real, raw_gen, rrc, rgc)
    (RES / "enrichment").mkdir(exist_ok=True)
    G, levels = tl.hierarchy_from_labels(raw_gen, rgc, MAX_LEVELS)
    out = tl.go_enrichment(G, levels[-1])
    pickle.dump(out, open(RES / "enrichment" / "enrichment_cache_generated.pkl.rawids", "wb"))
    print(f"generated: {sum(v is not None for v in out.values())}/{len(out)} nodes labelled", flush=True)


# ------------------------------------------------------------------ panel D UMAP caches
def panel_d_umap():
    """Separate per-arm UMAPs of the representative sampling seed (per-arm centred) on the genes
    common to the arms and both label files, in sorted gene order, into data/umap_cache/
    (umap_coords_{real,generated}.npy + gene_names.txt), so the plot never touches the arms."""
    import anndata as ad
    import scanpy as sc
    chosen = json.load(open(RES / "cluster_renumbering.json"))
    SS, CS = chosen["samp_seed"], chosen["clust_seed"]
    genes, real, gen, _n = load_arms(LINE, SS)
    lr, _ = tl.load_cluster_labels(LABELS / f"labels_samp{SS}_seed{CS}_real.renumbered.tsv.gz")
    lg, _ = tl.load_cluster_labels(LABELS / f"labels_samp{SS}_seed{CS}_gen.renumbered.tsv.gz")
    common = sorted(set(genes) & set(lr["gene_name"]) & set(lg["gene_name"]))
    pos = {g: i for i, g in enumerate(genes)}
    rows = [pos[g] for g in common]
    cache = RES / "umap_cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "gene_names.txt").write_text("\n".join(common) + "\n")
    for name, A in (("real", real), ("generated", gen)):
        X = np.asarray(A, np.float32)
        a = ad.AnnData(X=np.ascontiguousarray((X - X.mean(0))[rows].astype(np.float32)))
        sc.pp.neighbors(a, n_neighbors=UMAP_K, n_pcs=None, use_rep="X", metric=UMAP_METRIC, random_state=UMAP_SEED)
        sc.tl.umap(a, random_state=UMAP_SEED, min_dist=UMAP_MIN_DIST)
        np.save(cache / f"umap_coords_{name}.npy", a.obsm["X_umap"])
        print(f"  {name} UMAP done ({len(common)} genes)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("halves", "ladder", "ceiling", "pick", "enrich", "panel-d-umap"))
    ap.add_argument("--samp", type=int)
    ap.add_argument("--clust", type=int)
    ap.add_argument("--split", type=int)
    ap.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    for d in (LADDERS, CEILING, LABELS):
        d.mkdir(parents=True, exist_ok=True)
    if a.mode == "halves":
        build_halves(a.splits)
    elif a.mode == "ladder":
        print(run_ladder(a.samp, a.clust), flush=True)
    elif a.mode == "ceiling":
        print(run_ceiling(a.split, a.clust), flush=True)
    elif a.mode == "pick":
        pick()
    elif a.mode == "enrich":
        enrich()
    else:
        panel_d_umap()
