"""POI feature extraction + integrity check for MDA (CM4AI) real vs generated images.

    python mda_pipeline.py channels          # inspect channel order
    python mda_pipeline.py dry               # parse + column set, writes nothing
    python mda_pipeline.py                   # extract (honours SLURM array vars)
    python mda_pipeline.py merge             # concatenate shards, header-safe
    python mda_pipeline.py check             # integrity report

    sbatch --array=0-31 --wrap "python mda_pipeline.py"

Ported from mac_features.py + check_features.py. Differences that matter:

  * Anchor-based filename parsing. Head layout varies across B2AI_Jun26 (extra
    tokens before the round index, gene at position 5), so a positional parser
    drops files, and the drops cluster by gene.
  * The gene field can hold several comma-separated symbols (HPA antibodies that
    recognise more than one product, e.g. EIF5A,EIF5AL1). `gene` is the first
    symbol, `gene_full` the literal string.
  * cell_id is the filename stem minus the arm suffix. Positional keys collide
    across layout variants and produce false pairs in [5] and [8].
  * crc32 sharding, stable when the directory grows.
  * FIELDS is a union over N_PROBE files, not the first. nc_ratio/nuc_frac and
    obj_area/obj_area_cv are conditionally emitted; a probe file lacking one
    kills the shard mid-run via DictWriter's default extrasaction="raise".
  * Empty-mask skips are logged, so [4] can distinguish them from a partial run.

Design constraints carried over
-------------------------------
1. SCALE INVARIANCE. Decoder clamps predicted POI to [0, 1.5], real is [0, 1].
   Features marked [ABS] are not scale-invariant.
2. REFERENCE CHANNELS IDENTICAL WITHIN A PAIR. Cell mask is identical for a
   pair. [COVAR] features are covariates, never evaluation targets.
"""
import os, re, sys, csv, glob, zlib
import numpy as np
import pandas as pd
from tifffile import imread
from tqdm.auto import tqdm
from skimage.filters import threshold_otsu
from skimage.measure import label
from scipy.ndimage import (binary_fill_holes, binary_closing, distance_transform_edt,
                           center_of_mass, gaussian_filter, convolve)

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_DIR   = "/scratch/groups/emmalu/multimodal_phenotyping/generated_images/mda_Aug_11"
CACHE_CSV   = "mda_features"      # shard suffix appended
MERGED_CSV  = "mda_features.csv"

CH_DAPI, CH_POI, CH_ATUB, CH_ER = 2, 1, 0, 3   # confirmed same as macrophage

SUB         = 4      # stride for percentile estimates, speed only
CLOSE_SZ    = 5      # mask closing element
N_SHELLS    = 3      # radial bins
SMOOTH_OBJ  = 1.0    # gaussian sigma before object counting
GRAN_SIGMAS = (1, 2, 4, 8)
MORAN_LAGS  = (1, 2, 4, 8)
FLUSH_EVERY = 1000
N_PROBE     = 200    # files unioned to fix the column set
MIN_N       = 20     # must match analyze_features.py

SPLIT_GENE  = True   # gene = first symbol before the comma; gene_full keeps all
COPIED      = ["area_cell", "nuc_area_frac"]
META_COLS   = ("path", "kind", "gene", "gene_full", "treat", "cell_id")

# ── Feature groups. Set False to drop. Deltas are ms/image. ─────────────────
ENABLE = {
    "core":          True,   #  +3  mean_cell[ABS] total_cell[ABS] p99_cell[ABS]
                             #      area_cell[COVAR] mean_all[ABS]
    "dist_shape":    True,   #  +6  cv_cell top1_frac top5_frac gini entropy
                             #      p90_p50 p90_p10 mad_ratio skew kurt
    "translocation": True,   #  +5  nc_ratio nuc_frac nuc_area_frac[COVAR]
    "radial":        True,   #  +3  rad0..radN rad_com
    "autocorr":      True,   #  +6  moran_lag1/2/4/8
    "coloc":         True,   #  +8  corr_dapi/atub/er manders_dapi/atub/er
    "massdisp":      True,   #  +2  mass_disp
    "objects":       True,   #  +8  n_obj obj_area obj_area_cv
    "legacy_otsu":   True,   #  +4  mean_otsu[ABS] total[ABS] area_frac otsu_thr[ABS]
    "granularity":   False,  # +57
    "tas":           False,  # +15
    "haralick":      False,  # +20  duplicates autocorr + cv_cell
    "zernike":       False,  # +70  needs mahotas
}

_HAS_MAHOTAS = False
if ENABLE.get("zernike"):
    try:
        import mahotas
        _HAS_MAHOTAS = True
    except ImportError:
        print("WARNING zernike enabled but mahotas not installed; skipping")
        ENABLE["zernike"] = False

# ── Parsing ─────────────────────────────────────────────────────────────────
# Head layout varies; the positional block and the tail do not. Parse by anchor.
# Gene class includes ',' for multi-target antibodies (EIF5A,EIF5AL1).
TAIL = re.compile(r"__(?P<line>[A-Za-z0-9\-]+)_(?P<gene>[A-Za-z0-9,\-]+)_(?P<kind>real|pred)\.tif$")
POS  = re.compile(r"_(?P<well>[A-Za-z0-9]+)_R(?P<row>\d+)_z(?P<z>\d+)_crop_(?P<crop>\d+)__")
TREATS = {"UNTREATED", "PACLITAXEL", "VORINOSTAT"}


def parse_name(path):
    fn = os.path.basename(path)
    t, p = TAIL.search(fn), POS.search(fn)
    if not (t and p):
        return None
    hits = {tok.upper() for tok in fn[:p.start()].split("_")} & TREATS
    if len(hits) != 1:
        return None
    kind = t.group("kind")
    g = t.group("gene")
    return dict(kind=kind,
                gene=g.split(",")[0] if SPLIT_GENE else g,
                gene_full=g,
                treat=hits.pop(),
                cell_id=fn[: -len(f"_{kind}.tif")])


def load_img(path):
    img = imread(path).astype(np.float32)
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[-1] > 4:
        img = np.moveaxis(img, 0, -1)
    return img


# ── Feature extraction ──────────────────────────────────────────────────────
def features(path):
    img = load_img(path)
    poi = img[..., CH_POI]

    # Cell mask from REFERENCE channels only, so it cannot co-move with POI
    # induction. An Otsu-on-POI mask grows as a protein redistributes.
    ref = np.zeros(poi.shape, np.float32)
    for c in (CH_DAPI, CH_ATUB, CH_ER):
        ch = img[..., c]
        lo, hi = np.percentile(ch[::SUB, ::SUB], [1, 99.5])
        ref += np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)
    r8 = (ref * (255 / 3)).astype(np.uint8)
    raw = r8 > threshold_otsu(r8)
    if not raw.any():
        return None
    ys, xs = np.where(raw)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    cell = binary_fill_holes(binary_closing(raw[sl], np.ones((CLOSE_SZ, CLOSE_SZ))))
    if not cell.any():
        return None

    p = poi[sl]
    v = p[cell]
    tot = float(v.sum()) + 1e-8
    m = float(v.mean())
    s = float(v.std()) + 1e-8
    o = {}

    if ENABLE.get("core"):
        o["mean_cell"] = m
        o["total_cell"] = tot
        o["p99_cell"] = float(np.percentile(v, 99))
        o["area_cell"] = float(cell.sum())
        o["mean_all"] = float(poi.mean())

    if ENABLE.get("dist_shape"):
        sv = np.sort(v)
        o["cv_cell"] = s / (m + 1e-8)
        for frac, nm in ((0.01, "top1_frac"), (0.05, "top5_frac")):
            k = max(1, int(frac * sv.size))
            o[nm] = float(sv[-k:].sum() / tot)
        o["gini"] = float(1 - 2 * (np.cumsum(sv) / tot).mean())
        h, _ = np.histogram(v, bins=64)
        h = h[h > 0] / v.size
        o["entropy"] = float(-(h * np.log2(h)).sum())
        q10, q50, q90 = np.percentile(v, [10, 50, 90])
        o["p90_p50"] = float(q90 / (q50 + 1e-8))
        o["p90_p10"] = float(q90 / (q10 + 1e-8))
        o["mad_ratio"] = float(np.median(np.abs(v - q50)) / (q50 + 1e-8))
        o["skew"] = float(((v - m) ** 3).mean() / s ** 3)
        o["kurt"] = float(((v - m) ** 4).mean() / s ** 4)

    nuc = None
    if ENABLE.get("translocation"):
        dap = img[..., CH_DAPI][sl]
        d8 = (dap * 255 / max(float(dap.max()), 1e-6)).astype(np.uint8)
        nuc = binary_fill_holes(d8 > threshold_otsu(d8)) & cell
        cyto = cell & ~nuc
        if nuc.any() and cyto.any():
            o["nc_ratio"] = float(p[nuc].mean() / (p[cyto].mean() + 1e-8))
            o["nuc_frac"] = float(p[nuc].sum() / tot)
        o["nuc_area_frac"] = float(nuc.sum() / cell.sum())

    if ENABLE.get("radial"):
        dt = distance_transform_edt(cell)
        dt /= (dt.max() + 1e-8)
        edges = np.linspace(0, 1, N_SHELLS + 1)
        edges[-1] = 1.01
        for i in range(N_SHELLS):
            sh = cell & (dt >= edges[i]) & (dt < edges[i + 1])
            o[f"rad{i}"] = float(p[sh].sum() / tot) if sh.any() else 0.0
        o["rad_com"] = float((v * dt[cell]).sum() / tot)   # 0 = edge, 1 = centre

    if ENABLE.get("autocorr"):
        x = np.where(cell, p - m, 0.0)
        den = float((x * x).sum()) + 1e-8
        for lag in MORAN_LAGS:
            nb = (np.roll(x, lag, 0) + np.roll(x, -lag, 0) +
                  np.roll(x, lag, 1) + np.roll(x, -lag, 1))
            o[f"moran_lag{lag}"] = float((x * nb).sum() / (4 * den))

    if ENABLE.get("coloc"):
        for nm, c in (("dapi", CH_DAPI), ("atub", CH_ATUB), ("er", CH_ER)):
            b = img[..., c][sl][cell]
            o[f"corr_{nm}"] = float(np.corrcoef(v, b)[0, 1])
            o[f"manders_{nm}"] = float(v[b > b.mean()].sum() / tot)

    if ENABLE.get("massdisp"):
        cy, cx = center_of_mass(cell)
        py, px = center_of_mass(np.where(cell, p, 0.0))
        o["mass_disp"] = float(np.hypot(py - cy, px - cx) /
                               (np.sqrt(cell.sum() / np.pi) + 1e-8))

    if ENABLE.get("objects"):
        sm = gaussian_filter(np.where(cell, p, 0.0), SMOOTH_OBJ)
        hi = sm > np.percentile(sm[cell], 90)
        lab, nobj = label(hi, return_num=True)
        o["n_obj"] = int(nobj)
        if nobj:
            areas = np.bincount(lab.ravel())[1:]
            o["obj_area"] = float(areas.mean())
            o["obj_area_cv"] = float(areas.std() / (areas.mean() + 1e-8))

    if ENABLE.get("legacy_otsu"):
        # Original metric, kept for comparison. Inverts genuine induction when a
        # protein redistributes (verified: +1.5x total reads as log2FC -1.87).
        thr = float(threshold_otsu(p))
        mk = p > thr
        if mk.any():
            vo = p[mk]
            o["mean_otsu"] = float(vo.mean())
            o["total"] = float(vo.sum())
            o["area_frac"] = float(mk.sum() / cell.sum())
            o["otsu_thr"] = thr

    if ENABLE.get("granularity"):
        vt = np.where(cell, p, 0.0)
        base = float(vt.var()) + 1e-8
        for sg in GRAN_SIGMAS:
            o[f"gran{sg}"] = float(gaussian_filter(vt, sg).var() / base)

    if ENABLE.get("tas"):
        k3 = np.ones((3, 3)); k3[1, 1] = 0
        for bi, (lo, hi) in enumerate([(m - s, m + s), (m, np.inf), (m + s, np.inf)]):
            b = (p > lo) & (p < hi) & cell
            cnt = convolve(b.astype(np.uint8), k3, mode="constant")
            hist = np.bincount(cnt[b], minlength=9)[:9].astype(float)
            hist /= (hist.sum() + 1e-8)
            for j in range(9):
                o[f"tas{bi}{j}"] = float(hist[j])

    if ENABLE.get("haralick"):
        from skimage.feature import graycomatrix, graycoprops
        lo, hi = np.percentile(v, [1, 99])
        q = np.zeros(p.shape, np.uint8)
        q[cell] = (np.clip((p[cell] - lo) / max(hi - lo, 1e-6), 0, 1) * 31).astype(np.uint8)
        g = graycomatrix(q, [1, 4], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
        for prop in ("contrast", "correlation", "homogeneity", "ASM"):
            o[f"h_{prop}"] = float(graycoprops(g, prop).mean())
        gg = g[g > 0]
        o["h_entropy"] = float(-(gg * np.log2(gg)).sum() / (g.shape[-1] * g.shape[-2]))

    if ENABLE.get("zernike") and _HAS_MAHOTAS:
        r = np.sqrt(cell.sum() / np.pi)
        z = mahotas.features.zernike_moments(np.where(cell, p, 0.0), r, degree=8)
        for i, val in enumerate(z):
            o[f"zern{i:02d}"] = float(val)

    return o


# ── Mode: channels ──────────────────────────────────────────────────────────
def mode_channels(n=6):
    fs = sorted(glob.glob(os.path.join(IMAGE_DIR, "*_real.tif")))
    if not fs:
        raise SystemExit(f"no *_real.tif in {IMAGE_DIR}")
    step = max(1, len(fs) // n)
    print(f"{len(fs)} real tifs; sampling {n}\n")
    for f in fs[::step][:n]:
        a = load_img(f)
        print(os.path.basename(f))
        print(f"  shape {a.shape} dtype {a.dtype}")
        for i in range(a.shape[-1]):
            c = a[..., i]
            print(f"  ch{i}  mean {c.mean():.4f}  p99 {np.percentile(c, 99):.4f}  "
                  f"max {c.max():.4f}  frac>0.1 {(c > 0.1).mean():.3f}")
        print()
    print("DAPI = smallest frac>0.1, compact, sharp mean-to-p99 gap.")
    print("ER   = most diffuse, largest frac>0.1.")
    print(f"Current constants: DAPI={CH_DAPI} POI={CH_POI} ATUB={CH_ATUB} ER={CH_ER}")


# ── Mode: merge ─────────────────────────────────────────────────────────────
def mode_merge():
    shards = sorted(glob.glob(f"{CACHE_CSV}_[0-9][0-9][0-9].csv"))
    if not shards:
        raise SystemExit(f"no shards matching {CACHE_CSV}_NNN.csv "
                         f"(serial runs write {MERGED_CSV} directly, no merge needed)")
    hdr, n = None, 0
    with open(MERGED_CSV, "w", newline="") as out:
        for sh in shards:
            with open(sh) as f:
                h = f.readline()
                if hdr is None:
                    hdr = h
                    out.write(h)
                elif h != hdr:
                    raise SystemExit(f"{sh} header differs; shards used different "
                                     f"ENABLE or column sets. Delete and rerun.")
                for line in f:
                    out.write(line); n += 1
    print(f"{len(shards)} shards -> {MERGED_CSV}, {n} data rows")


# ── Mode: extract ───────────────────────────────────────────────────────────
def mode_extract(dry=False):
    TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    N_TASKS = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*_real.tif")) +
                   glob.glob(os.path.join(IMAGE_DIR, "*_pred.tif")))
    meta = {p: parse_name(p) for p in paths}
    unparsed = [p for p, v in meta.items() if not v]
    meta = {p: v for p, v in meta.items() if v}
    print(f"{len(paths)} tifs, {len(meta)} parsed, {len(unparsed)} unparsed")
    for p in unparsed[:5]:
        print(f"  unparsed: {os.path.basename(p)}")
    if unparsed:
        print("  ^ fix TAIL/POS/TREATS before the array run; drops cluster by gene")
    ngene = len({v["gene"] for v in meta.values()})
    print(f"{ngene} genes, treatments: "
          + ", ".join(f"{t} {sum(1 for v in meta.values() if v['treat'] == t)}"
                      for t in sorted(TREATS)))
    print("enabled: " + ", ".join(k for k, v in ENABLE.items() if v))

    if N_TASKS > 1:
        # crc32 is stable across processes and under directory growth;
        # sorted()[i::n] reassigns files when new ones land. hash() is salted.
        keys = [k for k in sorted(meta) if zlib.crc32(k.encode()) % N_TASKS == TASK_ID]
        meta = {k: meta[k] for k in keys}
        print(f"shard {TASK_ID}/{N_TASKS}: {len(meta)} files")
    CACHE = f"{CACHE_CSV}_{TASK_ID:03d}.csv" if N_TASKS > 1 else f"{CACHE_CSV}.csv"
    SKIPLOG = f"{CACHE_CSV}_skip_{TASK_ID:03d}.txt"

    # Column set as a UNION over N_PROBE files. nc_ratio/nuc_frac and
    # obj_area/obj_area_cv are conditional; taking the first file alone kills the
    # shard later via DictWriter extrasaction="raise". restval="" fills gaps.
    probe = {}
    for p in sorted(meta)[:N_PROBE]:
        try:
            fe = features(p)
        except Exception:
            continue
        if fe:
            probe.update({k: None for k in fe})
    if not probe:
        raise SystemExit("could not extract features from any file")
    FIELDS = list(META_COLS) + list(probe)
    print(f"{len(probe)} features: {', '.join(probe)}")

    if dry:
        print("dry run, nothing written")
        return

    done = set()
    if os.path.exists(CACHE):
        done = set(pd.read_csv(CACHE, usecols=["path"])["path"])
        print(f"cache: {len(done)} already measured")
    todo = [p for p in sorted(meta) if p not in done]
    print(f"{len(todo)} to measure")
    if not todo:
        print(f"saved {CACHE}")
        return

    new = not os.path.exists(CACHE)
    f = open(CACHE, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
    if new:
        w.writeheader()
    sk = open(SKIPLOG, "a")
    buf, bad, empty = [], 0, 0
    for p in tqdm(todo, desc="features"):
        try:
            fe = features(p)
        except Exception as e:
            bad += 1
            sk.write(f"{p}\terror\t{e}\n")
            if bad <= 5:
                print(f"  skip {os.path.basename(p)}: {e}")
            continue
        if fe is None:
            empty += 1
            sk.write(f"{p}\tempty_mask\n")   # logged so [4] can tell this apart
            continue                          # from an incomplete run
        buf.append(dict(path=p, **meta[p], **fe))
        if len(buf) >= FLUSH_EVERY:
            w.writerows(buf); f.flush(); buf = []
    if buf:
        w.writerows(buf)
    f.close(); sk.close()
    print(f"{bad} errored, {empty} empty masks (see {SKIPLOG})")
    print(f"saved {CACHE}")


# ── Mode: check ─────────────────────────────────────────────────────────────
def mode_check():
    print(f"reading {MERGED_CSV} ...")
    d = pd.read_csv(MERGED_CSV, low_memory=False)
    print(f"{len(d)} rows, {len(d.columns)} columns")

    # 1. interleaved headers from `cat`
    bad = (d["path"] == "path").sum()
    print(f"\n[1] header rows embedded as data: {bad}"
          + ("   <-- use `python mda_pipeline.py merge`" if bad else "   OK"))
    if bad:
        d = d[d["path"] != "path"]
        for c in d.columns:
            if c not in META_COLS:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        print(f"    dropped, {len(d)} rows remain")

    # 2. columns that should be numeric but are not
    obj = [c for c in d.columns if c not in META_COLS and d[c].dtype == object]
    print(f"\n[2] non-numeric feature columns: {obj if obj else 'none'}   "
          + ("<-- investigate" if obj else "OK"))

    # 3. duplicate paths from shard reassignment
    dup = d["path"].duplicated().sum()
    print(f"\n[3] duplicate paths: {dup}"
          + ("   <-- shard overlap; de-duplicating" if dup else "   OK"))
    if dup:
        d = d.drop_duplicates("path")
        print(f"    {len(d)} rows remain")

    # 4. coverage against disk, net of logged skips
    tifs = glob.glob(os.path.join(IMAGE_DIR, "*_real.tif")) + \
           glob.glob(os.path.join(IMAGE_DIR, "*_pred.tif"))
    nskip = 0
    for s in glob.glob(f"{CACHE_CSV}_skip_*.txt"):
        with open(s) as f:
            nskip += sum(1 for _ in f)
    print(f"\n[4] tifs on disk : {len(tifs)}")
    print(f"    rows in csv  : {len(d)}")
    print(f"    logged skips : {nskip} (empty mask or error)")
    missing = len(tifs) - len(d) - nskip
    print(f"    unaccounted  : {missing} ({100*missing/max(len(tifs),1):.1f}%)"
          + ("   <-- run incomplete, rerun to resume" if missing > 0 else "   OK"))

    # 5. pairing
    k = d.groupby(["gene", "treat", "cell_id"])["kind"].nunique()
    paired = int((k == 2).sum())
    print(f"\n[5] cells with both real and pred: {paired}")
    print(f"    unpaired (dropped downstream)  : {int((k == 1).sum())}")
    print(f"    rows accounted for             : {2*paired + int((k==1).sum())} / {len(d)}")

    # 6. design
    print(f"\n[6] genes {d['gene'].nunique()}, treatments {d['treat'].nunique()}")
    print(d.groupby("treat")["cell_id"].size().div(2).astype(int)
           .sort_values(ascending=False).to_string())
    per = d[d["kind"] == "real"].groupby(["gene", "treat"]).size()
    print(f"\n    cells per gene x treatment: median {per.median():.0f}, "
          f"10th pct {per.quantile(.1):.0f}, 90th pct {per.quantile(.9):.0f}")
    print(f"    gene x treatment groups with >= {MIN_N} cells: "
          f"{int((per >= MIN_N).sum())} / {len(per)}")

    # 7. NaN rates
    feats = [c for c in d.columns if c not in META_COLS]
    nan = d[feats].isna().mean().sort_values(ascending=False)
    print(f"\n[7] features with >1% NaN:")
    hi = nan[nan > 0.01]
    print("    none" if hi.empty else hi.to_string(float_format=lambda v: f"{v:.3f}"))

    # 8. the covariate identity test. Nothing downstream is trustworthy if this fails.
    for c in COPIED:
        if c not in d.columns:
            print(f"\n[8] {c}: COLUMN ABSENT   <-- test did not run, "
                  f"check the feature list from extraction")
            continue
        p = d.pivot_table(index=["gene", "treat", "cell_id"], columns="kind",
                          values=c, aggfunc="first").dropna()
        if len(p):
            md = float((p["real"] - p["pred"]).abs().max())
            print(f"\n[8] {c}: max |real - pred| = {md:.2e}   "
                  + ("OK, reference channels identical" if md < 1e-6
                     else "<-- UNEXPECTED, reference channels differ within a pair"))

    # 9. cell_id uniqueness. A collision makes [5] report a false pair and [8]
    #    compare two unrelated cells.
    c9 = d.groupby(["gene", "treat", "kind"])["cell_id"].agg(["size", "nunique"])
    coll = int((c9["size"] - c9["nunique"]).sum())
    print(f"\n[9] colliding cell_ids: {coll}"
          + ("   <-- pairing key not unique, fix parse_name" if coll else "   OK"))

    # 9b. multi-symbol antibodies collapsing onto one gene after the comma split
    if "gene_full" in d.columns:
        g = d.groupby("gene")["gene_full"].nunique()
        amb = g[g > 1]
        print(f"\n[9b] genes with >1 gene_full string: {len(amb)}"
              + ("   <-- SPLIT_GENE merged distinct antibodies" if len(amb) else "   OK"))
        if len(amb):
            print(amb.to_string())

    # 10. mask size by treatment. The mask takes the bbox of every above-threshold
    #     pixel with no connected-component selection. Paclitaxel rounds cells and
    #     increases crowding, so multi-cell masks are treatment-dependent, which
    #     contaminates every radial and colocalisation feature differentially.
    r = d[d["kind"] == "real"]
    if "area_cell" in d.columns:
        q = r.groupby("treat")["area_cell"].quantile([.5, .9, .99]).unstack()
        q["p99_p50"] = q[.99] / q[.5]
        print(f"\n[10] area_cell by treatment:")
        print(q.to_string(float_format=lambda v: f"{v:.1f}"))
        print("     p99/p50 above ~4, or much higher under one drug, means multi-cell masks")

    # 11. tautological targets: POI that is also a reference channel stain
    src = r["gene_full"] if "gene_full" in r.columns else r["gene"]
    ref_genes = sorted({s for s in src.astype(str).unique()
                        if any(x.upper().startswith(("TUBA", "TUBB", "TUBG"))
                               for x in s.split(","))})
    print(f"\n[11] tubulin POIs (reference channel is anti-tubulin): "
          f"{ref_genes if ref_genes else 'none'}")
    if ref_genes:
        print("     exclude from headline numbers; near-copy task, and the direct")
        print("     target of Paclitaxel on top of that")

    if bad or dup:
        out = MERGED_CSV.replace(".csv", "_clean.csv")
        d.to_csv(out, index=False)
        print(f"\nwrote {out} ({len(d)} rows). Use this one.")


# ── Dispatch ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "extract"
    {"channels": mode_channels,
     "extract":  mode_extract,
     "dry":      lambda: mode_extract(dry=True),
     "merge":    mode_merge,
     "check":    mode_check}[mode]()