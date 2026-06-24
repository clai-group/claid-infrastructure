# claid-infrastructure

**A calibrated temporal reference map of disease progression**

This repository contains the analytical pipeline for the Temporal
Knowledge Base (TKB) and ClaiD, its interactive explorer, as
described in:

> Tian J. et al. *A biologically calibrated temporal reference map of
> disease progression.* (2026)

The pipeline starts from `corrs.csv`, the aggregate temporal Spearman
correlation matrix derived from de-identified EHRs. All steps from
`corrs.csv` onward are fully reproducible from the deposited data.
Patient-level EHR data and derived feature files are not included.

---

## ClaiD — interactive explorer

**[Launch ClaiD →](https://clai-group.github.io/claid-infrastructure/)**

ClaiD browses the TKB directly: pick a phenotype, and every downstream
condition reached from it is plotted by time lag (x-axis) against model
confidence (y-axis), with classifier scores, tSPM correlation, and UK
Biobank genetic correlation all inspectable per edge.

### Guided examples

The explorer ships with a built-in tour reproducing the manuscript's key
findings. Each step can be triggered from the **Tour** button in the
top bar.

**1 · A temporal coordinate system**
Select *Osteoarthritis* as the source phenotype. Every dot is a
downstream condition; horizontal position is time after the source
onset, vertical position is model confidence that the progression is
real signal rather than administrative co-occurrence.

**2 · Confidence calibrates against genetics**
With *Osteoarthritis* still selected, switch the genetic layer to
**Supported**. The calibration trace asks whether higher model
confidence tracks higher genetic support ($r_g$); across the corpus it
is concentrated in the highest confidence strata and depleted in the lowest, showing the model recovers biology it was never
shown — the interactive counterpart to Figure 1.

**3 · Topological bridges**
Select *Acute myocardial infarction*, relax the temporal $p$ filter, and
set the genetic layer to **Supported**. The adjusted temporal $p$ is
weak, so the Random Forest is unsure ($P_\text{RF} = 0.332$) — yet the
Graph Neural Network is confident ($P_\text{GNN} = 0.930$), inferring
the acute MI → epilepsy link from network context rather than direct
co-occurrence. The interactive counterpart to Figure 3.

**4 · Mechanical cascades**
Set RF ≥ 0.9, GNN ≥ 0.9, and the genetic layer to **Null**. This
isolates real, high-confidence sequences that share no genetic
architecture. *Musculoskeletal pain, not low back pain* → cardiac
dysrhythmias is one such cascade ($P_\text{avg} = 0.984$,
$r_g = 0.046$, $p_{r_g} = 0.539$) — absence of shared genetics is the
signal here, not missing data. The interactive counterpart to Figure 2.

After the tour, click any edge to seed a sequence in the builder panel
and grow it in both directions — backward into what precedes a
condition, forward into what follows — chaining multi-step progressions
A → B → C with every alternative branch visible. Any view or trajectory
can be exported as CSV. Note: the 0–14 day lag window is treated as
co-occurrence rather than progression and is drawn dashed throughout.

---

## Repository structure

```
claid-infrastructure/
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
│   └── genetic_enrichment.py           
│
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
├── docs/
│   └──index.html                         # Interactive explorer (standalone)
│
└── README.md
```


## Replication walkthrough

### Level 1 — Reproduce all paper results from deposited data

No EHR access required. All inputs are in `data/`.

```bash
# 1. Clone and set up
git clone https://github.com/CLAI-group/claid-infrastructure
cd claid-infrastructure

# 2. Map UKBB genetic correlations onto the corpus
python 3_genetic_validation/map_corrs_phenx_icd_ukbb.py

# 3. Reproduce Figure 1 and enrichment statistics
python 3_genetic_validation/genetic_enrichment.py

# 4. Reproduce discovery pools
python 4_discovery/extract_mechanistic_discoveries_fullcorpus.py
```

Expected outputs match the manuscript exactly when seeds are unchanged

### Level 2 — Re-extract features from corrs.csv

`corrs.csv` is the public entry point. Features can be re-extracted
from it without any EHR access.

```bash
python 1_features/feature_extraction_rf.py
python 1_features/feature_extraction_gnn.py
```

Note: the MedGemma labeling step requires a locally deployed MedGemma-7B
instance and is not included in this pipeline. The 200-row
human-annotated development set (`validation_dataset.csv`) and the
10,000 silver-standard labels (`labels_10k.csv`) are deposited directly
and do not need to be regenerated for replication.

### Level 2 — Retrain classifiers from features

Requires `features_rf_10k.csv`, `features_gnn_10k.csv`, and
`features_rf_validation.csv`, available on request subject to MGB
data use agreement.

```bash
# RF (R)
Rscript 2_classifiers/rf/rf_mlho.R

# GNN (Python)
python 2_classifiers/gnn/gnn_train.py   # train
```

### Level 3 — Re-extract features from corrs.csv

`corrs.csv` is the public entry point. Features can be re-extracted
from it without any EHR access.

```bash
python 1_features/feature_extraction_rf.py
python 1_features/feature_extraction_gnn.py
```

Note: the MedGemma labeling step requires a locally deployed
MedGemma-7B instance. The 200-row human-annotated development set and
the 10,000 silver-standard labels are deposited directly and do not need
to be regenerated for replication.

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
  title   = {A calibrated temporal reference map of disease progression},
  author  = {Tian, Jiazi and others},
  journal = {},
  year    = {2026}
}
```

---

## Contact

CLAI Research Group, Mass General Brigham
GitHub: [https://github.com/CLAI-group](https://github.com/CLAI-group)
