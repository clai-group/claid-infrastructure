"""
genetic_enrichment.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths — edit here
# ============================================================
RG_PATH      = "data/corrs_phenx_icd_ukbb_matched_rg.csv"
RF_PATH      = "data/rf_predictions_allcorrs.csv"
GNN_PATH     = "data/gnn_predictions_allcorrs.csv"
OUTPUT_CSV   = "data/genetic_validation_check.csv"
OUTPUT_FIG   = "data/decile_enrichment.png"

# ============================================================
# Load
# ============================================================
rg  = pd.read_csv(RG_PATH)
rf  = pd.read_csv(RF_PATH)
gnn = pd.read_csv(GNN_PATH)

# ============================================================
# Per-value deduplication
# Retains one row per unique rg value to prevent inflation from
# multiple CCSR pairs that resolve to the same UKBB trait pair.
# ============================================================
val = (rg.dropna(subset=["rg"])
         .drop_duplicates(subset="rg", keep="first")
         .copy())

val["sig_pass"]  = val["rg_p"] < 0.05
val["mag_pass"]  = val["rg"].abs() > 0.15
val["bio_label"] = (val["sig_pass"] & val["mag_pass"]).astype(int)

N        = len(val)
baseline = val["bio_label"].mean()
print(f"Unique-rg validation set: N = {N}")
print(f"Validated positives: {val['bio_label'].sum()}  baseline: {baseline:.4f}")

# ============================================================
# Attach classifier probabilities
# ============================================================
val = val.merge(rf[["sequence", "duration", "rf_prob"]],
                on=["sequence", "duration"], how="left")
val = val.merge(gnn[["sequence", "duration", "pred_prob"]],
                on=["sequence", "duration"], how="left")
val = val.rename(columns={"pred_prob": "gnn_prob"})

print(f"rf_prob  non-null: {val['rf_prob'].notna().sum()}")
print(f"gnn_prob non-null: {val['gnn_prob'].notna().sum()}")

# ============================================================
# Decile assignment
# ============================================================
def assign_deciles(series, n_dec=10):
    sub   = series.dropna()
    order = np.argsort(sub.to_numpy(), kind="stable")
    dec   = np.empty(len(sub), int)
    for d, idx in enumerate(np.array_split(order, n_dec)):
        dec[idx] = d
    out = pd.Series(np.nan, index=series.index)
    out.loc[sub.index] = dec
    return out

val["rf_decile"]  = assign_deciles(val["rf_prob"])
val["gnn_decile"] = assign_deciles(val["gnn_prob"])

# ============================================================
# Enrichment and permutation test
# NOTE: seed anchors manuscript p-values — do not change
# ============================================================
np.random.seed(42)

def decile_prevalence(prob, label, n_dec=10):
    order = np.argsort(prob, kind="stable")
    prev  = np.empty(n_dec)
    for d, idx in enumerate(np.array_split(order, n_dec)):
        prev[d] = label[idx].mean()
    return prev

def permutation_test(prob, label, n_perm=1000):
    rng  = np.random.default_rng(42)
    base = label.mean()
    obs  = decile_prevalence(prob, label)[-1] / base
    count = sum(
        decile_prevalence(rng.permutation(prob), label)[-1] / base >= obs - 1e-9
        for _ in range(n_perm))
    return obs, (count + 1) / (n_perm + 1)

# ============================================================
# Save validation table
# ============================================================
cols = ["sequence", "startPhen_def", "endPhenx_def", "duration",
        "rg", "rg_p", "rg_se", "sig_pass", "mag_pass", "bio_label",
        "rf_prob", "rf_decile", "gnn_prob", "gnn_decile"]
cols = [c for c in cols if c in val.columns]
val[cols].to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved validation table -> {OUTPUT_CSV}")

# ============================================================
# Figure 1
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
deciles   = np.arange(10)
ymax      = baseline * 100 + 5

print("\nPer-decile prevalence (%):")
for ax, (name, col) in zip(axes, [("Random Forest",       "rf_prob"),
                                   ("Graph Neural Network", "gnn_prob")]):
    sub   = val.dropna(subset=[col])
    prob  = sub[col].to_numpy()
    label = sub["bio_label"].to_numpy()

    prev             = decile_prevalence(prob, label) * 100
    enrich, pval     = permutation_test(prob, label)
    p_str = "p < 0.001" if pval <= 0.001 else f"p = {pval:.3f}"
    ymax  = max(ymax, prev.max() + 5)

    print(f"  {name}: {np.round(prev, 1).tolist()}  "
          f"enrichment={enrich:.2f}x  {p_str}")

    colors    = ["#bdbdbd"] * 10
    colors[0] = "#3b78c2"   # lowest decile: blue
    colors[9] = "#d62728"   # top decile: red

    ax.bar(deciles, prev, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(baseline * 100, ls="--", color="black", lw=1.4,
               label=f"Baseline ({baseline * 100:.1f}%)")
    ax.set_title(f"{name}\nTop-decile enrichment: {enrich:.2f}\u00d7 ($p$ = {p_str})",
                 fontsize=13)
    ax.set_xlabel("Confidence decile (0 = lowest)")
    ax.set_xticks(deciles)
    ax.legend(loc="upper left", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

for ax in axes:
    ax.set_ylim(0, ymax)
axes[0].set_ylabel("Genetic signal prevalence (%)")
fig.suptitle(
    f"Biological enrichment across confidence deciles  "
    f"(per-value deduplicated $N$ = {N:,})",
    fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_FIG, dpi=350, bbox_inches="tight")
print(f"Saved figure -> {OUTPUT_FIG}")
