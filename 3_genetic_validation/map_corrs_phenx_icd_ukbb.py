"""
map_corrs_phenx_icd_ukbb.py
"""

import os
import pandas as pd
import numpy as np
from collections import Counter


# ============================================================
# Paths — edit these to match your directory structure
# ============================================================

CORRS_PATH      = "data/corrs.csv"
CCSR_ICD_PATH   = "data/CCSR_PASC_ICD.csv"
UKBB_MAP_PATH   = "data/ukbb_trait_icd10_mapping.csv"
UKBB_RG_PATH    = "data/ukbb_rg_cleaned.csv"

OUT_MATCHED = "data/path_not_set"
OUT_DETAIL  = "data/path_not_set"
OUT_SUMMARY = "data/path_not_set"

# ============================================================
# Stage 1: CCSR phenotype → ICD-10-CM mapping
# ============================================================
print("=" * 60)
print("Stage 1: CCSR phenotype → ICD-10-CM mapping")
print("=" * 60)

corrs = pd.read_csv(CORRS_PATH)
print(f"corrs shape: {corrs.shape}")

# Extract all unique phenotype labels from the corpus
phenx = pd.unique(corrs[["endPhenx_def", "startPhen_def"]].values.ravel())
print(f"Unique phenotypes: {len(phenx)}")
corrs_phenx = pd.DataFrame({"phenx": phenx})

# Map to ICD-10-CM codes via CCSR crosswalk
ccsr_icd = pd.read_csv(CCSR_ICD_PATH).dropna()
corrs_phenx_icd = corrs_phenx.merge(
    ccsr_icd[["ICD10", "phenx", "ICD10_desc"]],
    on="phenx",
    how="left",
)
print(f"Phenotypes without ICD10 mapping: {corrs_phenx_icd['ICD10'].isna().sum()}")


# ============================================================
# Stage 2: ICD-10-CM → UKBB trait mapping (3-char prefix join)
# ============================================================
print("\n" + "=" * 60)
print("Stage 2: ICD-10-CM → UKBB trait mapping")
print("=" * 60)

df_ukbb = pd.read_csv(UKBB_MAP_PATH)
print(f"UKBB mapping table: {len(df_ukbb):,} rows, "
      f"{df_ukbb['icd10_code'].notna().sum()} with ICD10 code")

# Separate phenotypes with and without ICD10 codes
df_corrs_coded  = corrs_phenx_icd[corrs_phenx_icd["ICD10"].notna()].copy()
df_corrs_no_icd = corrs_phenx_icd[corrs_phenx_icd["ICD10"].isna()].copy()
print(f"Phenotype rows with ICD10:    {len(df_corrs_coded):,}")
print(f"Phenotype rows without ICD10: {len(df_corrs_no_icd):,} "
      f"({df_corrs_no_icd['phenx'].nunique()} unique phenotypes)")

# Truncate ICD-10-CM to 3-character WHO ICD-10 root for matching
# ICD-10-CM codes extend to 7 chars; UKBB mapping uses 3-char root
df_corrs_coded["icd10_3char"] = df_corrs_coded["ICD10"].str[:3].str.upper()

df_ukbb_mapped = (
    df_ukbb[df_ukbb["icd10_code"].notna()]
    .copy()
    .rename(columns={
        "icd10_code"        : "ukbb_icd10_code",
        "icd10_description" : "ukbb_icd10_description",
        "mapping_tier"      : "ukbb_mapping_tier",
        "mapping_source"    : "ukbb_mapping_source",
        "source_note"       : "ukbb_source_note",
    })
)

print("\nJoining on 3-character ICD10 prefix ...")
df_joined = df_corrs_coded.merge(
    df_ukbb_mapped,
    left_on  ="icd10_3char",
    right_on ="ukbb_icd10_code",
    how      ="left",
)

# Classify match quality
df_joined["match_type"] = np.where(
    df_joined["ukbb_trait"].notna(),
    np.where(
        df_joined["ICD10"].str.len() == 3,
        "exact_3char",       # code already 3-char: exact match
        "prefix_truncated",  # longer code matched via prefix
    ),
    "no_ukbb_match",
)

print(f"Total joined rows:        {len(df_joined):,}")
print(f"Rows with UKBB match:     {df_joined['ukbb_trait'].notna().sum():,}")
print(f"Rows without UKBB match:  {df_joined['ukbb_trait'].isna().sum():,}")
print("\nMatch type breakdown:")
print(df_joined["match_type"].value_counts().to_string())

# Save detail table
detail_cols = [
    "phenx", "ICD10", "ICD10_desc", "icd10_3char", "match_type",
    "ukbb_trait", "ukbb_icd10_code", "ukbb_icd10_description",
    "ukbb_mapping_tier", "ukbb_mapping_source", "ukbb_source_note",
]
detail_cols = [c for c in detail_cols if c in df_joined.columns]
df_detail = df_joined[detail_cols].copy()
df_detail.to_csv(OUT_DETAIL, index=False)
print(f"\nSaved detailed mapping -> {OUT_DETAIL}")

# Build per-phenotype summary
def summarise_phenx(group):
    matched   = group[group["ukbb_trait"].notna()]
    unmatched = group[group["match_type"] == "no_ukbb_match"]["ICD10"].unique().tolist()
    return pd.Series({
        "n_icd10_codes"         : group["ICD10"].nunique(),
        "n_ukbb_matches"        : matched["ukbb_trait"].nunique(),
        "ukbb_traits_matched"   : "; ".join(sorted(matched["ukbb_trait"].dropna().unique())),
        "ukbb_icd10_codes"      : "; ".join(sorted(matched["ukbb_icd10_code"].dropna().unique())),
        "match_type"            : (
            "full_match"    if len(matched) > 0 and len(unmatched) == 0 else
            "partial_match" if len(matched) > 0 else
            "no_match"
        ),
        "unmatched_icd_prefixes": "; ".join(sorted({c[:3] for c in unmatched})),
        "mapping_sources"       : "; ".join(sorted(matched["ukbb_mapping_source"].dropna().unique())),
    })

summary_coded = (
    df_detail
    .groupby("phenx", sort=False)
    .apply(summarise_phenx, include_groups=False)
    .reset_index()
)

no_icd_phenx = (
    df_corrs_no_icd[["phenx"]]
    .drop_duplicates()
    .pipe(lambda d: d[~d["phenx"].isin(summary_coded["phenx"])])
    .assign(
        n_icd10_codes=0, n_ukbb_matches=0,
        ukbb_traits_matched="", ukbb_icd10_codes="",
        match_type="no_icd_in_corrs",
        unmatched_icd_prefixes="", mapping_sources="",
    )
)

summary = (
    pd.concat([summary_coded, no_icd_phenx], ignore_index=True)
    .sort_values(["match_type", "n_ukbb_matches"], ascending=[True, False])
)
summary.to_csv(OUT_SUMMARY, index=False)
print(f"Saved per-phenotype summary -> {OUT_SUMMARY}")

print("\n" + "=" * 60)
print("STAGE 2 SUMMARY")
print("=" * 60)
print(f"  Unique phenotypes in corrs:          {corrs_phenx['phenx'].nunique()}")
print(f"  Phenotypes with ≥1 ICD10 code:       {df_corrs_coded['phenx'].nunique()}")
print(f"  Phenotypes matched to ≥1 UKBB trait: {(summary['n_ukbb_matches'] > 0).sum()}")
print(f"  Phenotypes with no UKBB match:        {(summary['n_ukbb_matches'] == 0).sum()}")
print("\n  Match type distribution (per phenotype):")
print(summary["match_type"].value_counts().to_string())
print("\n  Top matched UKBB traits:")
all_traits = "; ".join(summary["ukbb_traits_matched"].dropna())
trait_counts = Counter(t.strip() for t in all_traits.split(";") if t.strip())
for trait, count in trait_counts.most_common(15):
    print(f"    {count:3d}x  {trait}")
print("=" * 60)


# ============================================================
# Stage 3: Attach LDSC rg values to corrs rows
# ============================================================
print("\n" + "=" * 60)
print("Stage 3: Attach LDSC rg values to corrs rows")
print("=" * 60)

df_rg = pd.read_csv(UKBB_RG_PATH)
print(f"UKBB rg table: {len(df_rg):,} rows")

# Build per-phenotype UKBB trait lookup from the detail table
lookup = (
    df_detail[df_detail["ukbb_trait"].notna()]
    .groupby("phenx")["ukbb_trait"]
    .apply(lambda x: "; ".join(sorted(x.unique())))
    .reset_index()
    .rename(columns={"ukbb_trait": "ukbb_traits"})
)

# Attach UKBB traits to corrs for both start and end phenotypes
df_out = corrs.merge(
    lookup.rename(columns={"phenx": "startPhen_def", "ukbb_traits": "start_ukbb_traits"}),
    on="startPhen_def", how="left",
).merge(
    lookup.rename(columns={"phenx": "endPhenx_def", "ukbb_traits": "end_ukbb_traits"}),
    on="endPhenx_def", how="left",
)

matched_start = df_out["start_ukbb_traits"].notna().sum()
matched_end   = df_out["end_ukbb_traits"].notna().sum()
print(f"Rows: {len(df_out):,}")
print(f"start_ukbb_traits filled: {matched_start:,} ({100 * matched_start / len(df_out):.1f}%)")
print(f"end_ukbb_traits filled:   {matched_end:,}   ({100 * matched_end   / len(df_out):.1f}%)")

# Use the original row index for join integrity
row_id_col = "Unnamed: 0" if "Unnamed: 0" in df_out.columns else None
if row_id_col is None:
    df_out = df_out.reset_index().rename(columns={"index": "_row_id"})
    row_id_col = "_row_id"

def explode_traits(df, col):
    """Split '; '-joined trait strings into one row per trait."""
    return (
        df.assign(**{col: df[col].str.split("; ")})
          .explode(col)
          .assign(**{col: lambda d: d[col].str.strip()})
    )

df_start = explode_traits(
    df_out[[row_id_col, "start_ukbb_traits"]].dropna(subset=["start_ukbb_traits"]),
    "start_ukbb_traits",
)
df_end = explode_traits(
    df_out[[row_id_col, "end_ukbb_traits"]].dropna(subset=["end_ukbb_traits"]),
    "end_ukbb_traits",
)

# Cross-reference start × end traits against the rg table
print("\nCross-referencing start × end UKBB trait pairs against rg table ...")
rg_start = df_rg.merge(
    df_start.rename(columns={"start_ukbb_traits": "trait1"}),
    on="trait1", how="inner",
)
rg_matched = rg_start.merge(
    df_end.rename(columns={"end_ukbb_traits": "trait2"}),
    on=[row_id_col, "trait2"],
    how="inner",
)
print(f"Candidate rg matches: {len(rg_matched):,}")

# Retain the estimate with the lowest p-value per corrs row
rg_best = (
    rg_matched
    .sort_values("p")
    .groupby(row_id_col, as_index=False)
    .agg(
        rg_trait1=("trait1", "first"),
        rg_trait2=("trait2", "first"),
        rg       =("rg",     "first"),
        rg_se    =("se",     "first"),
        rg_p     =("p",      "first"),
    )
)

# Merge back onto original corrs rows
df_final = df_out.merge(rg_best, on=row_id_col, how="left")

matched_rows = df_final["rg"].notna().sum()
print(f"\nCorrs rows with rg attached: {matched_rows:,} "
      f"({100 * matched_rows / len(df_final):.1f}%)")
print(f"Corrs rows without rg:       {df_final['rg'].isna().sum():,}")

# Save final output — rows with rg only (consistent with manuscript)
df_matched = df_final[df_final["rg"].notna()].copy()
df_matched.to_csv(OUT_MATCHED, index=False)
print(f"\nSaved matched output -> {OUT_MATCHED}")
print(f"  Rows: {len(df_matched):,}")
print(f"  Unique sequence-duration pairs: "
      f"{df_matched[['sequence', 'duration']].drop_duplicates().shape[0]:,}")

print("\nDone.")
