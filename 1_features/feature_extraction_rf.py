"""
feature_extraction_rf.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import warnings
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

# ============================================================
# Duration constants
# ============================================================
DURATION_ORDER = {
    "0-14 days":  1,
    "15-30 days": 2,
    "30-90 days": 3,
    "90+ days":   4,
}
DURATIONS = ["0-14 days", "15-30 days", "30-90 days", "90+ days"]

# ============================================================
# CCSR phenotype → body system mapping
# ============================================================
PHENX_TO_SYSTEMS = {
    "Administration of albumin and globulin": ["Procedure"],
    "Administration of antibiotics": ["Procedure"],
    "Administration of anti-inflammatory agents": ["Procedure"],
    "Administration of nutritional and electrolytic substances": ["Procedure"],
    "Administration of thrombolytics and platelet inhibitors": ["Procedure"],
    "Administration of diagnostic substances, NEC": ["Procedure"],
    "Administration of therapeutic substances, NEC": ["Procedure"],
    "Administration and transfusion of bone marrow, stem cells, pancreatic islet cells, and t-cells": ["Procedure"],
    "Chemotherapy": ["Procedure", "Oncologic"],
    "Radiation Therapy": ["Procedure", "Oncologic"],
    "Encounter for antineoplastic therapies": ["Procedure", "Oncologic"],
    "Encounter for prophylactic or other procedures": ["Procedure"],
    "Implant, device or graft related encounter": ["Procedure"],
    "Vaccinations": ["Procedure"],
    "COVID-19 vaccinations": ["Procedure"],
    "Allergic reactions": ["Immune"],
    "Coronary atherosclerosis and other heart disease": ["Cardiovascular"],
    "Heart failure": ["Cardiovascular"],
    "Cardiac dysrhythmias": ["Cardiovascular"],
    "Cardiac arrest and ventricular fibrillation": ["Cardiovascular"],
    "Acute myocardial infarction": ["Cardiovascular"],
    "Conduction disorders": ["Cardiovascular"],
    "HTN": ["Cardiovascular"],
    "Essential hypertension": ["Cardiovascular"],
    "Cerebral infarction": ["Neurology", "Cardiovascular"],
    "Acute phlebitis; thrombophlebitis and thromboembolism": ["Cardiovascular"],
    "Arterial dissections": ["Cardiovascular"],
    "Pneumonia (except that caused by tuberculosis)": ["Respiratory", "Infectious"],
    "Asthma": ["Respiratory"],
    "Chronic obstructive pulmonary disease and bronchiectasis": ["Respiratory"],
    "Respiratory failure; insufficiency; arrest": ["Respiratory"],
    "Epilepsy; convulsions": ["Neurology"],
    "Headache; including migraine": ["Neurology"],
    "Paralysis (other than cerebral palsy)": ["Neurology"],
    "Neurodevelopmental disorders": ["Neurology", "Psych", "Pediatric/Neonatal"],
    "Neurocognitive disorders": ["Neurology", "Psych"],
    "Multiple sclerosis": ["Neurology", "Immune"],
    "Diabetes mellitus, Type 2": ["Endocrine", "Cardiovascular"],
    "Thyroid disorders": ["Endocrine"],
    "Disorders of lipid metabolism": ["Endocrine", "Cardiovascular"],
    "Obesity": ["Endocrine", "Cardiovascular", "MSK"],
    "Fluid and electrolyte disorders": ["Endocrine", "Systemic"],
    "Osteoarthritis": ["MSK"],
    "Rheumatoid arthritis and related disease": ["MSK", "Immune"],
    "Systemic lupus erythematosus and connective tissue disorders": ["MSK", "Immune", "Cardiovascular", "Renal/GU"],
    "Other specified connective tissue disease": ["MSK", "Immune"],
    "Low back pain": ["MSK"],
    "Musculoskeletal pain, not low back pain": ["MSK"],
    "Other specified joint disorders": ["MSK"],
    "Osteoporosis": ["MSK"],
    "Aseptic necrosis and osteonecrosis": ["MSK"],
    "Acute and unspecified renal failure": ["Renal/GU"],
    "Chronic kidney disease": ["Renal/GU"],
    "Urinary tract infections": ["Renal/GU", "Infectious"],
    "Depressive disorders": ["Psych"],
    "Bipolar and related disorders": ["Psych"],
    "Anxiety and fear-related disorders": ["Psych"],
    "Trauma- and stressor-related disorders": ["Psych"],
    "Schizophrenia spectrum and other psychotic disorders": ["Psych"],
    "Opioid-related disorders": ["Psych"],
    "Alcohol-related disorders": ["Psych"],
    "Malaise and fatigue": ["Systemic"],
    "Fever": ["Systemic", "Infectious"],
    "Coagulation and hemorrhagic disorders": ["Hematologic"],
    "Leukemia": ["Hematologic", "Oncologic"],
    "Non-Hodgkin lymphoma": ["Hematologic", "Oncologic"],
    "Septicemia": ["Infectious", "Cardiovascular"],
    "Bacterial infections": ["Infectious"],
    "HIV infection": ["Infectious", "Immune"],
    "COVID1": ["Infectious", "Respiratory", "Systemic"],
    "COVID2": ["Infectious", "Respiratory", "Systemic"],
    "COVID3": ["Infectious", "Respiratory", "Systemic"],
    "COVID4": ["Infectious", "Respiratory", "Systemic"],
    "COVID5": ["Infectious", "Respiratory", "Systemic"],
    "COVID6": ["Infectious", "Respiratory", "Systemic"],
}

PHENX_PRIMARY_SYSTEM  = {k: v[0] for k, v in PHENX_TO_SYSTEMS.items()}
MULTISYSTEM_PHENOTYPES = {k for k, v in PHENX_TO_SYSTEMS.items() if len(v) > 1}
VAGUE_PHENOTYPES       = {k for k, v in PHENX_TO_SYSTEMS.items() if "Systemic" in v and len(v) == 1}
PHENOTYPE_SPECIFICITY  = {k: 1.0 / len(v) for k, v in PHENX_TO_SYSTEMS.items()}
ALL_SYSTEMS            = sorted({s for systems in PHENX_TO_SYSTEMS.values() for s in systems})

ACUTE_KEYWORDS   = ["acute", "sudden", "initial encounter", "injury", "arrest", "infarction",
                    "hemorrhage", "rupture", "fracture", "crisis", "poisoning", "embolism",
                    "failure", "trauma"]
CHRONIC_KEYWORDS = ["chronic", "long-term", "late effect", "sequela", "history of",
                    "persistent", "long-standing", "disease", "disorder"]
VAGUE_KEYWORDS   = ["unspecified", "other specified", "ill-defined", "abnormal findings",
                    "signs and symptoms", "other general", "nec", "not elsewhere"]


# ============================================================
# System helpers
# ============================================================

def get_primary_system(condition):
    if pd.isna(condition):
        return "Unknown"
    c = str(condition)
    if c in PHENX_PRIMARY_SYSTEM:
        return PHENX_PRIMARY_SYSTEM[c]
    cl = c.lower()
    if any(k in cl for k in ["procedure", "transfusion", "administration", "vaccination",
                               "encounter for", "implant"]):
        return "Procedure"
    if any(k in cl for k in ["pregnancy", "maternal", "obstetric", "puerperium"]):
        return "Pregnancy"
    if any(k in cl for k in ["cancer", "malignant", "neoplasm", "tumor", "lymphoma",
                               "leukemia", "myeloma", "sarcoma"]):
        return "Oncologic"
    if any(k in cl for k in ["neonatal", "perinatal", "newborn", "birth"]):
        return "Pediatric/Neonatal"
    return "Unknown"


def get_all_systems(condition):
    if pd.isna(condition):
        return ["Unknown"]
    return PHENX_TO_SYSTEMS.get(str(condition), [get_primary_system(condition)])


def get_phenotype_specificity(condition):
    if pd.isna(condition):
        return 0.0
    c = str(condition)
    if c in PHENOTYPE_SPECIFICITY:
        score = PHENOTYPE_SPECIFICITY[c]
        return score * 0.3 if any(p in c.lower() for p in VAGUE_KEYWORDS) else score
    cl = c.lower()
    if any(v in cl for v in ["unspecified", "other specified", "abnormal findings"]):
        return 0.1
    return 0.6 if len(c.split()) >= 4 else 0.4 if len(c.split()) >= 2 else 0.2


def compute_system_relatedness(df):
    sig = df[df["p.adjust"] < 0.05].copy()
    sig["src_sys"] = sig["startPhen_def"].map(PHENX_PRIMARY_SYSTEM).fillna("Unknown")
    sig["tgt_sys"] = sig["endPhenx_def"].map(PHENX_PRIMARY_SYSTEM).fillna("Unknown")
    cross = sig.groupby(["src_sys", "tgt_sys"]).size().reset_index(name="n_cross")
    sys_total = (pd.concat([sig["src_sys"].rename("system"), sig["tgt_sys"].rename("system")])
                 .value_counts().rename("n_total").reset_index())
    sys_total_dict = dict(zip(sys_total["system"], sys_total["n_total"]))
    raw = {}
    for _, row in cross.iterrows():
        sa, sb, n = row["src_sys"], row["tgt_sys"], row["n_cross"]
        raw[(sa, sb)] = n / np.sqrt(sys_total_dict.get(sa, 1) * sys_total_dict.get(sb, 1))
    cross_scores = [v for (a, b), v in raw.items() if a != b]
    max_cross    = max(cross_scores) if cross_scores else 1.0
    normalized   = {k: (0.95 * v / max_cross) if k[0] != k[1] else 1.0
                    for k, v in raw.items()}
    sym = {(s, s): 1.0 for s in ALL_SYSTEMS + ["Unknown"]}
    for (sa, sb), score in normalized.items():
        sym[(sa, sb)] = score
        if (sb, sa) not in normalized:
            sym[(sb, sa)] = score
    return sym


# ============================================================
# Feature classes
# ============================================================

def class1_statistical(df):
    f = pd.DataFrame(index=df.index)
    f["rho"]             = df["rho"]
    f["rho_abs"]         = df["rho"].abs()
    f["rho_squared"]     = df["rho"] ** 2
    f["neg_log10_padj"]  = -np.log10(df["p.adjust"].replace(0, 1e-300))
    f["signal_strength"] = f["rho_abs"] * f["neg_log10_padj"]
    f["bayes_factor"]    = np.log1p(f["rho_abs"] / (df["p.adjust"] + 1e-300))
    return f


def class2_directional(df):
    f = pd.DataFrame(index=df.index)
    rev_idx = df.set_index(["endPhenx_def", "startPhen_def", "duration"])[["rho", "p.adjust"]]
    keys    = list(zip(df["startPhen_def"], df["endPhenx_def"], df["duration"]))
    rev_rho = np.array([rev_idx["rho"].get(k, np.nan) for k in keys])
    fwd_abs = df["rho"].abs().values
    rev_abs = np.where(np.isnan(rev_rho), 0.0, np.abs(rev_rho))
    f["rho_reverse_abs"]       = rev_abs
    f["directional_dominance"] = (fwd_abs - rev_abs) / (fwd_abs + rev_abs + 1e-6)
    f["direction_confidence"]  = fwd_abs * (1 - rev_abs / (fwd_abs + 1e-6))
    return f


def class3_temporal(df):
    f = pd.DataFrame(index=df.index)
    pivot_rho = (df.pivot_table(index=["startPhen_def", "endPhenx_def"],
                                columns="duration", values="rho", aggfunc="first")
                   .reindex(columns=DURATIONS))
    rho_mat = (df[["startPhen_def", "endPhenx_def"]]
               .merge(pivot_rho.reset_index(), on=["startPhen_def", "endPhenx_def"], how="left")
               [DURATIONS].values)
    rho_abs_mat = np.abs(rho_mat)
    n_observed  = (~np.isnan(rho_mat)).sum(axis=1)

    dur_ord = df["duration"].map(DURATION_ORDER).fillna(0).astype(int)
    f["duration_ordinal"] = dur_ord

    safe_mat = np.nan_to_num(rho_abs_mat, nan=0.0)
    f["temporal_auc"]      = np.trapz(safe_mat, axis=1)

    pair_mean = np.nanmean(rho_abs_mat, axis=1)
    pair_sd   = np.nanstd(rho_abs_mat, axis=1)
    f["temporal_volatility"] = np.where(pair_mean > 0, pair_sd / (pair_mean + 1e-6), 0.0)
    f["rho_z_vs_pair_own"]   = np.where(
        (n_observed > 1) & (pair_sd > 0),
        (df["rho"].abs().values - pair_mean) / (pair_sd + 1e-6), 0.0)

    safe_max = np.nanmax(np.where(np.isnan(rho_abs_mat), -999, rho_abs_mat), axis=1)
    safe_min = np.nanmin(np.where(np.isnan(rho_abs_mat),  999, rho_abs_mat), axis=1)
    f["rho_temporal_range"] = np.where(n_observed >= 2, safe_max - safe_min, 0.0)

    x   = np.array([1, 2, 3, 4], dtype=float)
    x_c = x - x.mean()

    def slope(row):
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            return 0.0
        return np.dot(x_c[valid], row[valid]) / (np.dot(x_c[valid], x_c[valid]) + 1e-6)

    f["rho_temporal_slope"] = [slope(rho_abs_mat[i]) for i in range(len(df))]
    f["rho_x_short_lag"]    = df["rho"].abs() * (dur_ord == 1).astype(float)
    f["rho_x_long_lag"]     = df["rho"].abs() * (dur_ord == 4).astype(float)
    f["rho_x_duration"]     = df["rho"].abs() * dur_ord.astype(float)
    return f


def class4_empirical_priors(df):
    f = pd.DataFrame(index=df.index)
    sig = df[df["p.adjust"] < 0.05].copy()
    f["source_fan_out"] = df["startPhen_def"].map(
        sig.groupby("startPhen_def")["endPhenx_def"].nunique()).fillna(0)
    f["target_fan_in"]  = df["endPhenx_def"].map(
        sig.groupby("endPhenx_def")["startPhen_def"].nunique()).fillna(0)

    global_mean_rho = df["rho"].abs().mean()
    src_counts = df.groupby("startPhen_def").size()
    f["src_support"]       = df["startPhen_def"].map(src_counts).fillna(1)
    f["bayesian_shrunk_rho"] = (
        df["rho"].abs() * f["src_support"] + global_mean_rho * 10) / (f["src_support"] + 10)

    src_rho = df.groupby("startPhen_def")["rho"].agg(["mean", "std"])
    tgt_rho = df.groupby("endPhenx_def")["rho"].agg(["mean", "std"])
    tmp = (df[["startPhen_def", "endPhenx_def"]]
           .merge(src_rho.reset_index(), on="startPhen_def", how="left")
           .merge(tgt_rho.reset_index(), on="endPhenx_def", how="left",
                  suffixes=("_src", "_tgt")))
    tmp.index = df.index
    f["rho_z_vs_source"] = (df["rho"].abs() - tmp["mean_src"].abs()) / (tmp["std_src"] + 1e-6)
    f["rho_z_vs_target"] = (df["rho"].abs() - tmp["mean_tgt"].abs()) / (tmp["std_tgt"] + 1e-6)

    expected = (
        df["startPhen_def"].map(sig.groupby("startPhen_def").size() / src_counts).fillna(0) *
        df["endPhenx_def"].map(
            sig.groupby("endPhenx_def").size() / df.groupby("endPhenx_def").size()).fillna(0))
    f["association_surprise"] = np.log1p(
        (df["p.adjust"] < 0.05).astype(float) / (expected + 1e-6))
    return f


def class5_clinical_ontology(df, system_relatedness):
    f   = pd.DataFrame(index=df.index)
    s_sys = df["startPhen_def"].apply(get_primary_system)
    e_sys = df["endPhenx_def"].apply(get_primary_system)
    s_all = df["startPhen_def"].apply(get_all_systems)
    e_all = df["endPhenx_def"].apply(get_all_systems)
    f["same_primary_system"] = (s_sys == e_sys).astype(int)
    f["system_relatedness"]  = [
        system_relatedness.get((s, e), system_relatedness.get((e, s), 0.0))
        for s, e in zip(s_sys, e_sys)]
    f["system_jaccard"]  = [
        len(set(s) & set(e)) / len(set(s) | set(e)) if len(set(s) | set(e)) else 0.0
        for s, e in zip(s_all, e_all)]
    f["avg_specificity"] = (
        df["startPhen_def"].apply(get_phenotype_specificity) +
        df["endPhenx_def"].apply(get_phenotype_specificity)) / 2
    return f


def class7_markov_and_info_theory(df, G):
    f = pd.DataFrame(index=df.index)
    nodes    = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes  = len(nodes)

    in_deg   = dict(G.in_degree(weight="weight"))
    out_deg  = dict(G.out_degree(weight="weight"))
    total_wt = sum(out_deg.values()) + 1e-6
    p_src = {n: out_deg[n] / total_wt for n in nodes}
    p_tgt = {n: in_deg[n]  / total_wt for n in nodes}

    rows_list, cols_list, data_list = [], [], []
    for u, v, d in G.edges(data=True):
        rows_list.append(node_idx[u])
        cols_list.append(node_idx[v])
        data_list.append(d["weight"])
    adj = csr_matrix((data_list, (rows_list, cols_list)), shape=(n_nodes, n_nodes))
    rs  = adj.sum(axis=1).A1
    rs[rs == 0] = 1.0
    P  = adj.multiply(1 / rs[:, np.newaxis]).tocsr()
    P2 = P.dot(P)

    npmi_list, m1, m2 = [], [], []
    for _, row in df.iterrows():
        s, t = row["startPhen_def"], row["endPhenx_def"]
        if s in node_idx and t in node_idx:
            u, v = node_idx[s], node_idx[t]
            w = G[s][t]["weight"] if G.has_edge(s, t) else 0.0
            p_joint = w / total_wt
            if p_joint > 0:
                pmi  = np.log2(p_joint / (p_src[s] * p_tgt[t] + 1e-6))
                npmi = pmi / (-np.log2(p_joint + 1e-6))
            else:
                npmi = -1.0
            npmi_list.append(npmi)
            m1.append(P[u, v])
            m2.append(P2[u, v])
        else:
            npmi_list.append(0.0)
            m1.append(0.0)
            m2.append(0.0)

    f["info_npmi"]             = npmi_list
    f["markov_1step_prob"]     = m1
    f["markov_2step_prob"]     = m2
    f["markov_path_enhancement"] = np.array(m2) - np.array(m1)
    return f


def class8_advanced_topology(df, G):
    f = pd.DataFrame(index=df.index)
    try:
        hubs, authorities = nx.hits(G, max_iter=200, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hubs       = {n: 0 for n in G.nodes()}
        authorities = {n: 0 for n in G.nodes()}

    src_hub  = [hubs.get(s, 0.0)        for s in df["startPhen_def"]]
    tgt_auth = [authorities.get(t, 0.0) for t in df["endPhenx_def"]]
    f["source_hub_score"]       = src_hub
    f["target_authority_score"] = tgt_auth
    f["hub_to_authority_flow"]  = np.array(src_hub) * np.array(tgt_auth)

    jac = []
    for _, row in df.iterrows():
        s, t = row["startPhen_def"], row["endPhenx_def"]
        if G.has_node(s) and G.has_node(t):
            ss, st = set(G.successors(s)), set(G.successors(t))
            u = len(ss | st)
            jac.append(len(ss & st) / u if u > 0 else 0.0)
        else:
            jac.append(0.0)
    f["shared_downstream_jaccard"] = jac

    in_deg      = dict(G.in_degree())
    out_deg     = dict(G.out_degree())
    pagerank    = nx.pagerank(G, alpha=0.85, weight="weight")
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    f["target_sink_score"]  = [np.log1p(in_deg.get(t, 0)) for t in df["endPhenx_def"]]
    f["target_betweenness"] = [betweenness.get(t, 0.0) for t in df["endPhenx_def"]]
    f["target_pagerank"]    = [pagerank.get(t, 0.0)    for t in df["endPhenx_def"]]
    tgt_in  = np.array([in_deg.get(t, 0)  for t in df["endPhenx_def"]], dtype=float)
    tgt_out = np.array([out_deg.get(t, 0) for t in df["endPhenx_def"]], dtype=float)
    f["target_sink_score_net"] = tgt_in / (tgt_in + tgt_out + 1)
    return f


# ============================================================
# Reference graph
# ============================================================

def build_reference_graph(df, p_threshold=0.05):
    sig = df[df["p.adjust"] < p_threshold]
    G   = nx.DiGraph()
    for _, row in sig.iterrows():
        src, tgt = row["startPhen_def"], row["endPhenx_def"]
        w = -np.log10(row["p.adjust"] + 1e-300) * abs(row["rho"])
        if G.has_edge(src, tgt):
            if w > G[src][tgt]["weight"]:
                G[src][tgt]["weight"] = w
        else:
            G.add_edge(src, tgt, weight=w)
    return G


# ============================================================
# Main pipeline
# ============================================================

def extract_all_features(input_file, output_file, verbose=True):
    def log(msg):
        if verbose:
            print(msg)

    log("=" * 70)
    log("RF Feature Extraction Pipeline")
    log("=" * 70)

    log(f"\n[1/6] Loading {input_file} ...")
    df = pd.read_csv(input_file)
    log(f"      {len(df):,} rows loaded")

    log("\n[2/6] Computing system relatedness ...")
    sys_rel = compute_system_relatedness(df)

    log("\n[3/6] Extracting Classes 1–5 ...")
    c1 = class1_statistical(df)
    c2 = class2_directional(df)
    c3 = class3_temporal(df)
    c4 = class4_empirical_priors(df)
    c5 = class5_clinical_ontology(df, sys_rel)

    log("\n[4/6] Building reference graph ...")
    G = build_reference_graph(df)
    log(f"      {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    log("\n[5/6] Extracting Class 7 (Markov / Info Theory) ...")
    c7 = class7_markov_and_info_theory(df, G)

    log("\n[6/6] Extracting Class 8 (Advanced Topology / HITS) ...")
    c8 = class8_advanced_topology(df, G)

    id_cols    = [c for c in ["sequence", "startPhen_def", "endPhenx_def", "duration"] if c in df.columns]
    label_cols = [c for c in ["label"] if c in df.columns]

    out = pd.concat([df[id_cols + label_cols], c1, c2, c3, c4, c5, c7, c8], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.to_csv(output_file, index=False)

    n_features = out.shape[1] - len(id_cols) - len(label_cols)
    log(f"\nSaved: {output_file}  ({n_features} features, {len(out):,} rows)")
    return out


# ============================================================
# Entry point — edit paths here
# ============================================================

if __name__ == "__main__":

    CORRS_FILE  = "data/corrs.csv"
    LABELS_FILE = "../data/labels_10k.csv"
    VAL_FILE    = "../data/validation_dataset.csv"
    OUTPUT_FILE = "data/features_rf.csv"

    all_df = extract_all_features(CORRS_FILE, OUTPUT_FILE)

    # Training subset
    subset_labels = pd.read_csv(LABELS_FILE)[["sequence", "duration", "label"]]
    subset_df     = all_df.merge(subset_labels, on=["sequence", "duration"], how="inner")
    print(f"Training subset: {subset_df.shape}")
    subset_df.to_csv(OUTPUT_FILE.replace(".csv", "_10k.csv"), index=False)

    # Validation subset
    val_labels = pd.read_csv(VAL_FILE)[["sequence", "duration", "label_jt", "label_bp1"]]
    val_df     = all_df.merge(val_labels, on=["sequence", "duration"], how="inner")
    print(f"Validation subset: {val_df.shape}")
    val_df.to_csv(OUTPUT_FILE.replace(".csv", "_validation.csv"), index=False)
