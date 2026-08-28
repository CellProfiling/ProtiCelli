# ProtiCelli drug-perturbation profiling

Quantify drug-induced changes in subcellular protein distribution from
ProtiCelli-generated immunofluorescence images.

## Modes

- **validation** (real + generated images): reproduces the manuscript results,
  per-drug concordance vs a split-half repeatability ceiling (Fig. D),
  strong-responder recovery AP (Fig. E), accuracy by true-effect decile (Fig. F),
  and the feature ranking (Table S4).
- **profiling** (generated images only): ranks proteins by simulated perturbation
  effect within each drug. The intended antibody-free use.

Both modes share `profiling_core.py` and differ only in whether a real arm exists.

## Files

| file | purpose |
|---|---|
| `profiling_core.py` | 40-feature extraction, Cliff's delta, effects table |
| `run_profiling.py` | command-line runner, both modes, cached + resumable |
| `proticelli_profiling_demo.ipynb` | demonstration and small runs |

## Quick start

```bash
python run_profiling.py --mode validation --image_dir IMAGES \
    --control UNTREATED --drugs PACLITAXEL VORINOSTAT --out results_val
python run_profiling.py --mode profiling  --image_dir IMAGES \
    --control DMSO --drugs DRUGA DRUGB --out results_prof
```

## Inputs

TIFF stacks, one per cell x protein x arm. Channel order index 0 = microtubule,
1 = protein of interest, 2 = DAPI, 3 = ER (set as
`CH_DAPI, CH_POI, CH_MT, CH_ER = 2, 1, 0, 3` in `profiling_core.py`; verify
against your stacks, a wrong mapping fails silently). Filenames must encode kind
(real|pred), gene, treatment and a unique cell id; the default parser matches
`{prefix}_{TREAT}_..._crop_{n}__{CELLLINE}_{GENE}_{real|pred}.tif`.

## Caveat

The best-recovered feature is radial distribution, computed on a mask from the
reference channels the model conditions on, so recovery there can partly reflect
cell geometry. Recovery is most reliable for large spatial effects; weak effects
are near chance. Treat profiling rankings as hypotheses and validate hits
experimentally.
