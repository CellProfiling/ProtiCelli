# Reference-channel determinability

How much of a protein's within-cell localisation the three landmark stains (microtubules,
nucleus, ER) determine, per single cell, then per protein and compartment. Produces Fig 2I
(the determinability boxes), Fig S4B and Fig S4C.

## Run

```bash
PY=/path/to/python bash run_reference_determinability.sh          # every step, in order
STEPS=plot bash run_reference_determinability.sh                  # only re-plot from data/
STEPS=worker SHARDS="$(seq 0 99)" bash run_reference_determinability.sh   # a subset of shards
```

Steps, in order: `index` -> `codebooks` -> `worker` (1000 shards, resumable, about 1000 CPU-hours
in total) -> `ceiling` -> `aggregate` -> `ksweep` -> `plot`. The runner is plain bash; hand the
worker loop to any scheduler. `determinability.py` holds the whole compute chain, one subcommand
per step (see its docstring for the method); `plot_determinability.py` draws the three figures
from `data/`.

## Inputs

| input | env | used by |
|---|---|---|
| HPA single-cell crops: five 16-bit PNGs per cell (red = MT, green = protein, blue = nucleus, yellow = ER, cell mask) + `crops_index.parquet` (atlas_name, ensembl_ids, gene_names, locations, if_plate_id, position, sample, cell) | `HPA_CROPS` | `index`, `codebooks`, `worker`, `ceiling`, `ksweep` |
| HPA location -> compartment group | `location_groups.csv` (included) | `aggregate`, `plot` |
| the per-compartment reliability ranking of the ProtiCelli reliability analysis (Fig 2I, blue): one compartment per line, most reliable first | `reliability_ranking.txt` (included; override with `RELIABILITY_RANKING`) | Fig S4C |

The HPA single-cell crops are segmented from the HPA subcellular images (this analysis used HPA v25.1; ProtiCelli itself
used v24, https://v24.proteinatlas.org/) as described in the Methods. A processed version of the v23 data,
segmented crops with their SubCell embeddings, can be downloaded for testing from
https://virtualcellmodels.cziscience.com/dataset/hpa-subcellular-section-subcell and should yield
similar results.

## Outputs

| figure | script section | files |
|---|---|---|
| Fig 2I, determinability by compartment | `plot_determinability.py` §1 | `figures/determinability_by_compartment.*`, `data/determinability_by_compartment_{pairs,summary}.csv` |
| Fig S4B, single landmarks vs joint | §2 | `figures/joint_vs_single_channels.*` |
| Fig S4C, reliability rank vs determinability rank | §3 | `figures/reliability_vs_determinability.*`, `data/reliability_vs_determinability_ranking.csv` |

The K-sweep numbers quoted in the Methods come from `ksweep` (`data/convention_test_summary.csv`).
Compute steps write to `data/` (`index_sc_full.parquet`, `codebooks.joblib`, `parts_matched_k/`,
`ceiling_joint_k96.csv`, `matched_k96_per_protein_sc.csv`).

## Environment

Python >= 3.10 with `requirements.txt` (the plot script needs only its first block).
