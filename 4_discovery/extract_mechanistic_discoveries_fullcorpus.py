"""
extract_mechanistic_discoveries_fullcorpus.py
"""

import pandas as pd
import numpy as np
import os


# ============================================================
# Paths — edit here
# ============================================================
RF_PATH      = "data/rf_predictions_allcorrs.csv"
GNN_PATH     = "data/gnn_predictions_allcorrs.csv"
UKBB_RG_PATH = "data/corrs_phenx_icd_ukbb_matched_rg.csv"
OUTPUT_DIR   = "data/"


# ============================================================
# Helpers
# ============================================================
def s(v, d=3):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, d)
    except (TypeError, ValueError):
        return None


def print_discovery1_row(row):
    print(f"  Pathway:        {row['startPhen_def']} -> {row['endPhenx_def']}")
    print(f"  Lag:            {row['duration']}")
    print(f"  RF:             {s(row['rf_prob'])}")
    print(f"  GNN:            {s(row['gnn_prob'])}")
    print(f"  P_avg:          {s(row['prob_rf_gnn_avg'])}")
    if pd.notna(row.get("rho")):
        print(f"  tSPM rho:       {s(row['rho'], 4)}  (p = {row['p.value']:.2e})")
    print(f"  r_g:            {s(row['rg'])}  (p = {s(row['rg_p'], 4)})")


def print_discovery2_row(row):
    print(f"  Pathway:        {row['startPhen_def']} -> {row['endPhenx_def']}")
    print(f"  Lag:            {row['duration']}")
    print(f"  RF (low):       {s(row['rf_prob'])}")
    print(f"  GNN (high):     {s(row['gnn_prob'])}")
    if pd.notna(row.get("rho")):
        print(f"  tSPM rho:       {s(row['rho'], 4)}  (p = {row['p.value']:.2e})")
    if pd.notna(row.get("rg")):
        print(f"  r_g:            {s(row['rg'])}  (p = {s(row['rg_p'], 4)})")


# ============================================================
# Main
# ============================================================
def extract_mechanistic_discoveries(rf_path, gnn_path, ukbb_rg_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load predictions
    print("Loading RF predictions ...")
    rf = pd.read_csv(rf_path)
    print(f"  {len(rf):,} rows")

    print("Loading GNN predictions ...")
    gnn = pd.read_csv(gnn_path)
    if "pred_prob" in gnn.columns and "gnn_prob" not in gnn.columns:
        gnn = gnn.rename(columns={"pred_prob": "gnn_prob"})
    print(f"  {len(gnn):,} rows")

    print("Loading UKBB rg file ...")
    rg_raw = pd.read_csv(ukbb_rg_path)
    rg_cols = ["sequence", "startPhen_def", "endPhenx_def", "duration",
               "rho", "p.value", "rg", "rg_p", "rg_se"]
    rg_cols = [c for c in rg_cols if c in rg_raw.columns]
    rg_raw  = rg_raw[rg_cols].drop_duplicates(subset=["sequence", "duration"])
    print(f"  {len(rg_raw):,} rows")

    # Merge predictions onto the rg spine
    df = rg_raw.copy()
    df = df.merge(rf[["sequence", "duration", "rf_prob"]],
                  on=["sequence", "duration"], how="left")
    df = df.merge(gnn[["sequence", "duration", "gnn_prob"]],
                  on=["sequence", "duration"], how="left")
    df = df.dropna(subset=["rf_prob", "gnn_prob"]).copy()

    df["prob_rf_gnn_avg"] = (df["rf_prob"] + df["gnn_prob"]) / 2
    df["rg_abs"]          = df["rg"].abs()

    print(f"\nMerged corpus: {len(df):,} rows with predictions")

    # ── Discovery 1: Mechanical cascade ──────────────────────────────────
    print("\n" + "=" * 60)
    print(" Discovery 1: Mechanical Cascade")
    print("=" * 60)
    print("Criterion: P_avg > 0.95  AND  |rg| < 0.05  AND  p_rg > 0.50")
    print("(Confirmed null rg required — missing values excluded)")

    mechanical = df[
        (df["prob_rf_gnn_avg"] > 0.95) &
        (df["rg_abs"]          < 0.05) &
        (df["rg_p"]            > 0.50)
    ].copy()
    print(f"Pool size: {len(mechanical):,} rows")

    print("\nTop 10 by P_avg:")
    cols_show = ["startPhen_def", "endPhenx_def", "duration",
                 "prob_rf_gnn_avg", "rf_prob", "gnn_prob", "rho", "p.value", "rg", "rg_p"]
    cols_show = [c for c in cols_show if c in mechanical.columns]
    print(mechanical.sort_values("prob_rf_gnn_avg", ascending=False)
                    [cols_show].head(10).to_string(index=False))

    # Manuscript exemplar: Musculoskeletal pain -> Cardiac dysrhythmias
    target_1 = mechanical[
        mechanical["startPhen_def"].str.contains("Musculoskeletal pain", case=False, na=False) &
        mechanical["endPhenx_def"].str.contains("Cardiac dysrhythmias", case=False, na=False)
    ]
    print(f"\nManuscript exemplar (Musculoskeletal pain -> Cardiac dysrhythmias): "
          f"{len(target_1)} row(s)")
    for _, row in target_1.iterrows():
        print_discovery1_row(row)

    # ── Discovery 2: Topological bridge ──────────────────────────────────
    print("\n" + "=" * 60)
    print(" Discovery 2: Topological Bridge")
    print("=" * 60)
    print("Criterion: P_RF < 0.40  AND  P_GNN > 0.90")

    bridge = df[
        (df["rf_prob"]  < 0.40) &
        (df["gnn_prob"] > 0.90)
    ].copy()
    bridge["divergence"] = bridge["gnn_prob"] - bridge["rf_prob"]
    print(f"Pool size: {len(bridge):,} rows")

    print("\nTop 10 by divergence (GNN - RF):")
    cols_show2 = ["startPhen_def", "endPhenx_def", "duration",
                  "rf_prob", "gnn_prob", "divergence", "rho", "p.value", "rg", "rg_p"]
    cols_show2 = [c for c in cols_show2 if c in bridge.columns]
    print(bridge.sort_values("divergence", ascending=False)
                [cols_show2].head(10).to_string(index=False))

    # Manuscript exemplar: Acute MI -> Epilepsy
    target_2 = bridge[
        bridge["startPhen_def"].str.contains("Acute myocardial infarction", case=False, na=False) &
        bridge["endPhenx_def"].str.contains("Epilepsy", case=False, na=False)
    ]
    print(f"\nManuscript exemplar (Acute MI -> Epilepsy): {len(target_2)} row(s)")
    for _, row in target_2.iterrows():
        print_discovery2_row(row)

    # ── Save outputs ──────────────────────────────────────────────────────
    mechanical.to_csv(output_dir + "discovery1_mechanical_pool.csv", index=False)
    bridge.to_csv(output_dir + "discovery2_bridge_pool.csv", index=False)

    target_rows = pd.concat([
        target_1.assign(discovery="mechanical"),
        target_2.assign(discovery="topological_bridge"),
    ])
    target_rows.to_csv(output_dir + "discovery_target_pathways.csv", index=False)

    print(f"\nSaved discovery1_mechanical_pool.csv      ({len(mechanical):,} rows)")
    print(f"Saved discovery2_bridge_pool.csv           ({len(bridge):,} rows)")
    print(f"Saved discovery_target_pathways.csv        ({len(target_rows)} rows)")


if __name__ == "__main__":
    extract_mechanistic_discoveries(RF_PATH, GNN_PATH, UKBB_RG_PATH, OUTPUT_DIR)
