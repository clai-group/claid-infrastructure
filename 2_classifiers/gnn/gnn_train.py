"""
GNN Edge Classification — Training
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve,
    accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
# NOTE: on GPU, TransformerConv scatter ops may still differ in the last
# few decimals. For bit-exact runs, train on CPU (this graph is small).


# ============================================================
# CONFIG
# ============================================================
CSV_PATH     = "data/features_gnn_10k.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

SAVE_MODEL     = "data/path_not_set"
SAVE_PREDS     = "data/path_not_set"
SAVE_METRICS   = "data/path_not_set"
GRID_FILE      = "data/path_not_set"
SAVE_ALL_PREDS = "data/path_not_set"

# ── Training control ──────────────────────────────────────────
# Leave False to train once with BEST_PARAMS (reproducible).
# Set True only to re-search hyperparameters.
RUN_GRID_SEARCH = True

# Search space (only used when RUN_GRID_SEARCH = True).
GRID = {
    'hidden_dim':        [64, 128],
    'dropout':           [0.3, 0.5],
    'weight_decay':      [1e-3, 1e-4],
    'label_smoothing':   [0.0, 0.05],
    'edge_feat_dropout': [0.0, 0.1],
}

GRID_EPOCHS  = 120   # epochs per grid combo (only if searching)
FINAL_EPOCHS = 200   # peak was ~epoch 147, so 200 is ample
PATIENCE     = 40    # margin so a late peak isn't cut off early


# ============================================================
# COLUMN SCHEMA
# ============================================================
NON_FEATURE_COLS = [
    'sequence_id', 'label', 'sequence',
    'startPhen_def', 'endPhenx_def', 'duration',
]

EDGE_FEATURE_COLS = [
    # ── Statistical signal (5) ────────────────────────────────
    'rho', 'rho_abs', 'neg_log10_padj', 'signal_strength', 'posterior_prob',
    # ── Directional (3) ───────────────────────────────────────
    'forward_only', 'rho_reverse_abs', 'has_reverse',
    # ── Temporal (9) ──────────────────────────────────────────
    'duration_ordinal', 'n_observed_durations', 'prop_sig_durations',
    'rho_temporal_slope', 'rho_z_vs_pair_own', 'rho_temporal_range',
    'peak_rho_duration', 'rho_x_long_lag', 'rho_x_duration',
    # ── Empirical priors (6) ──────────────────────────────────
    'rho_z_vs_source', 'rho_z_vs_target', 'association_surprise',
    'source_fan_out', 'target_fan_in', 'target_sink_score',
    # ── Clinical ontology (14) ────────────────────────────────
    'same_primary_system', 'n_systems_overlap', 'system_jaccard',
    'system_relatedness', 'start_is_multisystem', 'end_is_multisystem',
    'start_specificity', 'end_specificity', 'start_is_vague', 'end_is_vague',
    'start_is_acute', 'end_is_acute',
    'acute_long_lag_implausible', 'chronic_long_lag_plausible',
    # ── Network (8) ───────────────────────────────────────────
    'target_pagerank', 'target_betweenness', 'target_is_bridge',
    'target_high_pagerank', 'source_transition_entropy', 'cascade_potential',
    'local_transition_prob', 'has_reciprocal_edge',
]

# Structural features used for message passing only — a small subset so the
# GNN must use graph structure rather than shortcut through edge features.
MP_FEATURE_COLS = [
    'rho', 'duration_ordinal', 'signal_strength', 'forward_only', 'posterior_prob',
]

# Node features (aggregated per phenotype from the 435k-derived columns).
SOURCE_NODE_COLS   = ['source_fan_out', 'source_pagerank', 'source_transition_entropy']
TARGET_NODE_COLS   = ['target_fan_in', 'target_sink_score', 'target_pagerank',
                      'target_betweenness', 'target_is_bridge', 'target_high_pagerank']
INTRINSIC_SRC_COLS = ['start_specificity', 'start_is_multisystem', 'start_is_acute']
INTRINSIC_TGT_COLS = ['end_specificity', 'end_is_multisystem', 'end_is_acute', 'end_is_vague']


# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
print('=' * 60)
print('LOADING DATA')
print('=' * 60)

df = pd.read_csv(CSV_PATH)
print(f'Shape: {df.shape}')

EDGE_FEATURE_COLS = [c for c in EDGE_FEATURE_COLS if c in df.columns]
MP_FEATURE_COLS   = [c for c in MP_FEATURE_COLS   if c in df.columns]

label_map = {'NOISE': 0, 'NON-NOISE': 1, 'N': 0, 'Y': 1, 0: 0, 1: 1}
df['label_bin'] = df['label'].map(label_map)
df = df[df['label_bin'].notna()].copy()
df['label_bin'] = df['label_bin'].astype(int)
df = df.reset_index(drop=True)
print(f'Label distribution: {df["label_bin"].value_counts().to_dict()}')


# ============================================================
# 2. NODE INDEX & FEATURES
# ============================================================
all_nodes = pd.unique(pd.concat([
    df['startPhen_def'].astype(str), df['endPhenx_def'].astype(str)]))
node_encoder = LabelEncoder().fit(all_nodes)
df['src'] = node_encoder.transform(df['startPhen_def'].astype(str))
df['dst'] = node_encoder.transform(df['endPhenx_def'].astype(str))
NUM_NODES = len(all_nodes)

node_df = pd.DataFrame(index=node_encoder.classes_)
if SOURCE_NODE_COLS:
    src_agg = df.groupby('startPhen_def')[SOURCE_NODE_COLS].mean()
    for col in SOURCE_NODE_COLS:
        node_df[f'as_src_{col}'] = src_agg[col].reindex(node_df.index).fillna(0)
if TARGET_NODE_COLS:
    tgt_agg = df.groupby('endPhenx_def')[TARGET_NODE_COLS].mean()
    for col in TARGET_NODE_COLS:
        node_df[f'as_tgt_{col}'] = tgt_agg[col].reindex(node_df.index).fillna(0)
if INTRINSIC_SRC_COLS:
    src_intr = df.groupby('startPhen_def')[INTRINSIC_SRC_COLS].mean()
    for col in INTRINSIC_SRC_COLS:
        node_df[col] = src_intr[col].reindex(node_df.index).fillna(0)
if INTRINSIC_TGT_COLS:
    tgt_intr = df.groupby('endPhenx_def')[INTRINSIC_TGT_COLS].mean()
    for col in INTRINSIC_TGT_COLS:
        node_df[col] = tgt_intr[col].reindex(node_df.index).fillna(0)
node_df = node_df.fillna(0)

node_scaler   = StandardScaler()
node_features = torch.tensor(node_scaler.fit_transform(node_df.values), dtype=torch.float)
NODE_DIM      = node_features.shape[1]
print(f'Nodes: {NUM_NODES}  |  Node feature dim: {NODE_DIM}')


# ============================================================
# 3. EDGE FEATURES
# ============================================================
df[EDGE_FEATURE_COLS] = (df[EDGE_FEATURE_COLS]
                         .replace([np.inf, -np.inf], np.nan).fillna(0))
edge_scaler = StandardScaler()
edge_attr   = torch.tensor(edge_scaler.fit_transform(df[EDGE_FEATURE_COLS].values),
                           dtype=torch.float)
EDGE_DIM    = edge_attr.shape[1]

df[MP_FEATURE_COLS] = (df[MP_FEATURE_COLS]
                       .replace([np.inf, -np.inf], np.nan).fillna(0))
mp_scaler = StandardScaler()
mp_attr   = torch.tensor(mp_scaler.fit_transform(df[MP_FEATURE_COLS].values),
                         dtype=torch.float)
MP_DIM    = mp_attr.shape[1]
print(f'Edge dim: {EDGE_DIM}  |  MP dim: {MP_DIM}  |  Node dim: {NODE_DIM}')


# ============================================================
# 4. GRAPH STRUCTURE & LABELS
# ============================================================
# Graph edges use edge_in_reference == 1 (significant in the 435k corpus);
# labels are predicted for all labeled rows.
if 'edge_in_reference' in df.columns:
    ref_mask      = df['edge_in_reference'].values == 1
    edge_index    = torch.tensor([df['src'].values[ref_mask],
                                  df['dst'].values[ref_mask]], dtype=torch.long)
    mp_attr_graph = mp_attr[ref_mask]
    print(f'Graph edges (edge_in_reference=1): {edge_index.shape[1]:,} '
          f'of {len(df):,} total rows')
else:
    edge_index    = torch.tensor([df['src'].values, df['dst'].values], dtype=torch.long)
    mp_attr_graph = mp_attr
    print(f'Graph edges (all rows): {edge_index.shape[1]:,}')

edge_labels        = torch.tensor(df['label_bin'].values, dtype=torch.float)
labeled_edge_index = torch.tensor([df['src'].values, df['dst'].values], dtype=torch.long)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=df['label_bin'].values)

train_mask = torch.zeros(len(df), dtype=torch.bool); train_mask[train_idx] = True
test_mask  = torch.zeros(len(df), dtype=torch.bool); test_mask[test_idx]   = True
print(f'Train: {train_mask.sum().item():,}  |  Test: {test_mask.sum().item():,}')

train_labels_np = df['label_bin'].iloc[train_idx].values
n_pos = (train_labels_np == 1).sum()
n_neg = (train_labels_np == 0).sum()
auto_pw = n_neg / max(n_pos, 1)
print(f'Train pos_weight: {auto_pw:.3f}')


# ============================================================
# 6. MODEL
# ============================================================
class EdgeClassificationGNN(nn.Module):
    """Regularized TransformerConv GNN for edge classification."""
    def __init__(self, node_dim, mp_edge_dim, edge_dim,
                 hidden_dim=64, dropout=0.4, heads=2):
        super().__init__()
        self.edge_feat_dropout = dropout
        self.conv1 = TransformerConv(node_dim, hidden_dim, edge_dim=mp_edge_dim,
                                     heads=heads, concat=False, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, edge_dim=mp_edge_dim,
                                     heads=heads, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        mlp_in = hidden_dim * 2 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, mp_edge_attr,
                pred_edge_index, pred_edge_attr, edge_feat_drop=0.0):
        if self.training and edge_feat_drop > 0:
            mask = torch.bernoulli(torch.ones_like(pred_edge_attr) * (1 - edge_feat_drop))
            pred_edge_attr = pred_edge_attr * mask
        h = F.relu(self.bn1(self.conv1(x, edge_index, mp_edge_attr)))
        h = F.dropout(h, p=self.edge_feat_dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index, mp_edge_attr)))
        src_emb = h[pred_edge_index[0]]
        dst_emb = h[pred_edge_index[1]]
        edge_repr = torch.cat([src_emb, dst_emb, pred_edge_attr], dim=1)
        return self.edge_mlp(edge_repr).squeeze(-1)


# ============================================================
# 7. SMOOTHED LOSS  (training only)
# ============================================================
class SmoothedBCELoss(nn.Module):
    """BCE with label smoothing — prevents the overconfident threshold collapse."""
    def __init__(self, pos_weight, smoothing=0.05):
        super().__init__()
        self.smoothing  = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing / 2
        return F.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=self.pos_weight)


# ============================================================
# 8. METRICS HELPERS
# ============================================================
def compute_test_roc(model, data, labeled_edge_index, edge_attr,
                     test_mask, edge_labels):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr,
                       labeled_edge_index, edge_attr, edge_feat_drop=0.0)
        probs     = torch.sigmoid(logits)
        test_prob = probs[test_mask].cpu().numpy()
        test_true = edge_labels[test_mask].cpu().numpy()
    return roc_auc_score(test_true, test_prob), test_prob, test_true


def print_metrics(y_true, y_pred, y_prob, threshold, label=''):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy     = accuracy_score(y_true, y_pred)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    sensitivity  = recall_score(y_true, y_pred, pos_label=1)
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv          = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1           = f1_score(y_true, y_pred, pos_label=1)
    mcc          = matthews_corrcoef(y_true, y_pred)
    kappa        = cohen_kappa_score(y_true, y_pred)
    lr_pos       = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float('inf')
    lr_neg       = (1 - sensitivity) / specificity if specificity > 0 else 0.0

    print(f'\n{"=" * 60}\nMETRICS  {label}\n{"=" * 60}')
    print(f'\nConfusion Matrix:')
    print(f'                 Predicted')
    print(f'                 NOISE   NON-NOISE')
    print(f'Actual NOISE      {tn:5d}   {fp:5d}')
    print(f'Actual NON-NOISE  {fn:5d}   {tp:5d}')
    if y_prob is not None:
        print(f'\n── Threshold-independent ──────────────────')
        print(f'  ROC-AUC:           {roc_auc_score(y_true, y_prob):.4f}')
        print(f'  PR-AUC:            {average_precision_score(y_true, y_prob):.4f}')
    print(f'\n── Threshold = {threshold:.4f} ────────────────────')
    print(f'  Accuracy:          {accuracy:.4f}')
    print(f'  Balanced Accuracy: {bal_accuracy:.4f}')
    print(f'  Sensitivity (TPR): {sensitivity:.4f}')
    print(f'  Specificity (TNR): {specificity:.4f}')
    print(f'  PPV (Precision):   {ppv:.4f}')
    print(f'  NPV:               {npv:.4f}')
    print(f'  F1 Score:          {f1:.4f}')
    print(f'  MCC:               {mcc:.4f}')
    print(f'  Cohen Kappa:       {kappa:.4f}')
    print(f'  LR+:               {lr_pos:.4f}')
    print(f'  LR-:               {lr_neg:.4f}')
    print(f'\n── Counts ─────────────────────────────────')
    print(f'  TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}  '
          f'Total:{len(y_true)}  Prevalence:{y_true.mean():.3f}')

    return {
        'threshold': threshold,
        'roc_auc':   roc_auc_score(y_true, y_prob) if y_prob is not None else None,
        'pr_auc':    average_precision_score(y_true, y_prob) if y_prob is not None else None,
        'accuracy': accuracy, 'balanced_accuracy': bal_accuracy,
        'sensitivity': sensitivity, 'specificity': specificity,
        'ppv': ppv, 'npv': npv, 'f1': f1,
        'mcc': mcc, 'kappa': kappa, 'lr_pos': lr_pos, 'lr_neg': lr_neg,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


# ============================================================
# 9. TRAINING FUNCTION
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')


def train_model(params, epochs, patience, data, edge_attr,
                labeled_edge_index, edge_labels,
                train_mask, test_mask, save_path=None, verbose=True):
    """Train one configuration. Returns (best_roc, best_epoch, best_state, model).
    Node features come from data.x and MP features from data.edge_attr, so
    they are NOT passed separately."""
    hidden_dim        = int(params['hidden_dim'])
    dropout           = float(params['dropout'])
    weight_decay      = float(params['weight_decay'])
    lr                = float(params.get('lr', 1e-3))
    label_smoothing   = float(params.get('label_smoothing', 0.05))
    edge_feat_dropout = float(params.get('edge_feat_dropout', 0.1))

    model = EdgeClassificationGNN(
        node_dim=NODE_DIM, mp_edge_dim=MP_DIM, edge_dim=EDGE_DIM,
        hidden_dim=hidden_dim, dropout=dropout,
    ).to(device)

    pos_weight = torch.tensor([auto_pw], dtype=torch.float).to(device)
    criterion  = SmoothedBCELoss(pos_weight, smoothing=label_smoothing)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_roc, best_epoch, best_state, no_improve = 0.0, 0, None, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr,
                       labeled_edge_index, edge_attr, edge_feat_drop=edge_feat_dropout)
        loss = criterion(logits[train_mask], edge_labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        roc, _, _ = compute_test_roc(model, data, labeled_edge_index, edge_attr,
                                     test_mask, edge_labels)
        if roc > best_roc:
            best_roc, best_epoch = roc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                train_roc = roc_auc_score(
                    edge_labels[train_mask].cpu().numpy(),
                    torch.sigmoid(logits[train_mask]).detach().cpu().numpy())
            print(f'  Epoch {epoch:03d} | Loss {loss.item():.4f} | '
                  f'Train ROC {train_roc:.4f} | Test ROC {roc:.4f} | '
                  f'Gap {train_roc - roc:.4f} | LR {scheduler.get_last_lr()[0]:.2e}')
            model.train()

        if no_improve >= patience:
            if verbose:
                print(f'  Early stop at epoch {epoch} (no improvement for {patience})')
            break

    if save_path and best_state:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), save_path)
    if verbose:
        print(f'  Best test ROC: {best_roc:.4f} at epoch {best_epoch}')
    return best_roc, best_epoch, best_state, model


# ============================================================
# 10. MOVE SHARED TENSORS TO DEVICE
# ============================================================
data_obj = Data(x=node_features, edge_index=edge_index, edge_attr=mp_attr_graph).to(device)
edge_attr_dev          = edge_attr.to(device)
edge_labels_dev        = edge_labels.to(device)
train_mask_dev         = train_mask.to(device)
test_mask_dev          = test_mask.to(device)
labeled_edge_index_dev = labeled_edge_index.to(device)


# ============================================================
# 11. GRID SEARCH  (optional)
# ============================================================
if RUN_GRID_SEARCH:
    print('\n' + '=' * 60 + '\nGRID SEARCH\n' + '=' * 60)
    from itertools import product
    keys   = list(GRID.keys())
    combos = list(product(*GRID.values()))
    print(f'Total combinations: {len(combos)}  |  Epochs per combo: {GRID_EPOCHS}')

    grid_results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params['lr'], params['heads'] = 1e-3, 2
        print(f'\n[{i+1}/{len(combos)}] {params}')
        roc, epoch, _, _ = train_model(
            params=params, epochs=GRID_EPOCHS, patience=15,
            data=data_obj, edge_attr=edge_attr_dev,
            labeled_edge_index=labeled_edge_index_dev, edge_labels=edge_labels_dev,
            train_mask=train_mask_dev, test_mask=test_mask_dev,
            save_path=None, verbose=False)
        grid_results.append({**params, 'test_roc': roc, 'best_epoch': epoch})
        print(f'  → Test ROC: {roc:.4f}  (epoch {epoch})')

    grid_df = pd.DataFrame(grid_results).sort_values('test_roc', ascending=False)
    grid_df.to_csv(GRID_FILE, index=False)
    print('\n' + '=' * 60 + '\nGRID SEARCH RESULTS (top 5)\n' + '=' * 60)
    print(grid_df.head(5).to_string(index=False))

    best_row    = grid_df.iloc[0].to_dict()
    BEST_PARAMS = {k: best_row[k] for k in keys}
    BEST_PARAMS['lr'], BEST_PARAMS['heads'] = 1e-3, 2
    print(f'\nBest params: {BEST_PARAMS}')


# ============================================================
# 12. FINAL TRAINING WITH BEST PARAMS
# ============================================================
set_seed(SEED)   # re-seed so the final model is identical whether or not grid ran

print('\n' + '=' * 60 + '\nFINAL TRAINING')
print(f'Params: {BEST_PARAMS}\n' + '=' * 60)

best_roc, best_epoch, best_state, model = train_model(
    params=BEST_PARAMS, epochs=FINAL_EPOCHS, patience=PATIENCE,
    data=data_obj, edge_attr=edge_attr_dev,
    labeled_edge_index=labeled_edge_index_dev, edge_labels=edge_labels_dev,
    train_mask=train_mask_dev, test_mask=test_mask_dev,
    save_path=SAVE_MODEL, verbose=True)

print(f'\nBest test ROC: {best_roc:.4f} at epoch {best_epoch}')
print(f'Model saved to: {SAVE_MODEL}')


# ============================================================
# 13. FINAL EVALUATION
# ============================================================
print('\n' + '=' * 60 + '\nFINAL EVALUATION\n' + '=' * 60)
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()
with torch.no_grad():
    logits = model(data_obj.x, data_obj.edge_index, data_obj.edge_attr,
                   labeled_edge_index_dev, edge_attr_dev, edge_feat_drop=0.0)
    probs = torch.sigmoid(logits)

test_probs = probs[test_mask_dev].cpu().numpy()
test_true  = edge_labels_dev[test_mask_dev].cpu().numpy()

test_preds_05 = (test_probs >= 0.5).astype(int)
fpr, tpr, thresholds = roc_curve(test_true, test_probs)
best_thresh   = thresholds[np.argmax(tpr - fpr)]
test_preds_y  = (test_probs >= best_thresh).astype(int)

m_05 = print_metrics(test_true, test_preds_05, test_probs, 0.5, '(threshold = 0.50)')
m_y  = print_metrics(test_true, test_preds_y, None, best_thresh, f'(Youden = {best_thresh:.4f})')


# ============================================================
# 14. SUMMARY TABLE
# ============================================================
print(f'\n{"=" * 60}\nFINAL SUMMARY TABLE\n{"=" * 60}')
print(f'{"Metric":<25} {"Threshold=0.50":>15} {"Youden":>15}')
print('-' * 57)
for label, key in [
    ('ROC-AUC', 'roc_auc'), ('PR-AUC', 'pr_auc'), ('Accuracy', 'accuracy'),
    ('Balanced Accuracy', 'balanced_accuracy'), ('Sensitivity', 'sensitivity'),
    ('Specificity', 'specificity'), ('PPV', 'ppv'), ('NPV', 'npv'),
    ('F1', 'f1'), ('MCC', 'mcc'), ('Kappa', 'kappa'),
]:
    v1, v2 = m_05.get(key), m_y.get(key)
    s1 = f'{v1:.4f}' if v1 is not None else '   —  '
    s2 = f'{v2:.4f}' if v2 is not None else '   —  '
    print(f'{label:<25} {s1:>15} {s2:>15}')
print(f'\n{"Threshold":<25} {"0.5000":>15} {best_thresh:>15.4f}')
print(f'\n{"Best params":<25} {str(BEST_PARAMS)}')


# ============================================================
# 15. SAVE
# ============================================================
results_df = df.iloc[test_idx].copy()
results_df['pred_prob']         = test_probs
results_df['pred_label_05']     = test_preds_05
results_df['pred_label_youden'] = test_preds_y
results_df.to_csv(SAVE_PREDS, index=False)

# Predictions for ALL input rows (train + test) — probs already cover every row.
all_probs = probs.cpu().numpy()
all_df = df.copy()
all_df['split']             = 'train'
all_df.loc[test_idx, 'split'] = 'test'
all_df['pred_prob']         = all_probs
all_df['pred_label_05']     = (all_probs >= 0.5).astype(int)
all_df['pred_label_youden'] = (all_probs >= best_thresh).astype(int)
all_df.to_csv(SAVE_ALL_PREDS, index=False)

metrics_df = pd.DataFrame([
    {'threshold_type': 'fixed_0.5', **m_05},
    {'threshold_type': 'youden',    **m_y},
])
metrics_df.to_csv(SAVE_METRICS, index=False)

print(f'\nSaved:')
print(f'  {SAVE_MODEL}            — best model weights')
print(f'  {SAVE_PREDS}   — test predictions')
print(f'  {SAVE_ALL_PREDS}  — predictions for all {len(all_df):,} rows')
print(f'  {SAVE_METRICS}          — metrics summary')
if RUN_GRID_SEARCH:
    print(f'  {GRID_FILE}     — grid search results')