# claid-infrastructure

**A biologically calibrated temporal reference map of disease progression**

This repository contains the analytical pipeline for the Temporal
Knowledge Base (TKB), as described in:

> Tian J. et al. *A biologically calibrated temporal reference map of
> disease progression.* (2026)

The pipeline starts from `corrs.csv`, the aggregate temporal Spearman
correlation matrix derived from de-identified EHRs. All steps from
`corrs.csv` onward are fully reproducible from the deposited data.
Patient-level EHR data and derived feature files are not included and
are available only under MGB data use agreement.

---

## Repository structure

```
TKB-infrastructure/
│
├── 1_features/
│   ├── feature_extraction_rf.py      
│   └── feature_extraction_gnn.py
│
├── 2_classifiers/
│   ├── rf/
│   │   └── rf_mlho.R               
│   └── gnn/
│       └── gnn_train.py    
│
├── 3_genetic_validation/
│   ├── map_corrs_phenx_icd_ukbb.py
│   └──  genetic_enrichment.py           

├── 4_discovery/
│   └── extract_mechanistic_discoveries_fullcorpus.py
│
├── data/
│   ├── corrs.csv                          # Full tSPM corpus (435,240 rows)
│   ├── validation_dataset.csv             # 200-row human-annotated development set
│   ├── labels_10k.csv                     # MedGemma silver-standard labels for 10,000 training sequences
│   ├── ukbb_rg_cleaned.csv                # Neale Lab LDSC pairwise rg table
│   ├── ukbb_trait_icd10_mapping.csv       # Curated ICD-10-to-UKBB trait
│   ├── CCSR_PASC_ICD.csv                  # CCSR-to-ICD-10-CM crosswalk
│   └── genetic_validation_check.csv       # UKBB-matched validation set
│
└── README.md
```


## Replication walkthrough

### Level 1 — Reproduce all paper results from deposited data

No EHR access required. All inputs are in `data/`.

```bash
# 1. Clone and set up
git clone https://github.com/CLAI-group/TKB-infrastructure
cd TKB-infrastructure
conda env create -f environment.yml
conda activate tkb

# 2. Reproduce Figure 1 and enrichment statistics (Table in §2.3)
python 4_genetic_validation/genetic_enrichment.py

# 3. Reproduce discovery pools (§2.4)
python 5_discovery/extract_mechanistic_discoveries_fullcorpus.py

# 4. Reproduce Table 2 and all corpus-level statistics
jupyter notebook notebooks/reproducibility_walkthrough.ipynb
```

Expected outputs match the manuscript exactly when seeds are unchanged
(see Reproducibility seeds below).

### Level 2 — Retrain classifiers from features

Requires `features_rf_10k.csv`, `features_gnn_10k.csv`, and
`features_rf_validation.csv`, available on request subject to MGB
data use agreement.

```bash
# RF (R)
Rscript 3_classifiers/rf/rf_mlho.R

# GNN (Python)
python 3_classifiers/gnn/gnn_train_final.py   # train
python 3_classifiers/gnn/gnn_val.py           # validate on dev set
python 3_classifiers/gnn/gnn_predict.py       # score full corpus
```

### Level 3 — Re-extract features from corrs.csv

`corrs.csv` is the public entry point. Features can be re-extracted
from it without any EHR access.

```bash
python 1_features/feature_extraction_rf.py
python 1_features/feature_extraction_gnn.py
```

Note: the MedGemma labeling step (Step 2) requires a locally deployed
MedGemma-7B instance. The 200-row human-annotated development set and
the 10,000 silver-standard labels are deposited directly and do not need
to be regenerated for replication.

---

## Reproducibility seeds

| Script | Seed | Purpose |
|--------|------|---------|
| `genetic_enrichment.py` | `np.random.seed(42)` | Permutation test — anchors manuscript $p < 0.001$ values. **Do not change.** |
| `gnn_train_final.py` | `set_seed(42)` | GNN training reproducibility |
| `rf_mlho.R` | `set.seed(40)` | RF training reproducibility |

---

## Dependencies

**Python** (see `environment.yml`)
- Python 3.10+
- PyTorch 2.x + PyTorch Geometric
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn

**R** (see `requirements_r.txt`)
- R 4.3+
- mlho, caret, pROC, tidyverse, data.table, praznik

---

## Citation

```bibtex
@article{tian2026tkb,
  title   = {A biologically calibrated temporal reference map of
             disease progression},
  author  = {Tian, Jiazi and others},
  journal = {},
  year    = {2026}
}
```

---

## Contact

CLAI Research Group, Mass General Brigham
GitHub: [https://github.com/CLAI-group](https://github.com/CLAI-group)
