#!/usr/bin/env python3
"""
Neural network experiments for gas consumption prediction.

Approach:
    1. Load pre-engineered gas parquet (same as xgb_gas)
    2. Train a quick XGBoost to rank features by importance
    3. Select top-K features
    4. Train multiple NN architectures on those K features
    5. Compare R² across architectures and K values

Architectures:
    A. DeepMLP         — Wide hidden layers, BatchNorm, GELU, dropout
    B. ResNetMLP       — Residual blocks with skip connections
    C. TwoStageNet     — Shared backbone + classifier head + regressor head (multi-task)
    D. EntityEmbedMLP  — Entity embeddings for building ID + deep MLP

Usage:
    python nn_gas/experiment.py
    python nn_gas/experiment.py --top-k 15 --arch all
    python nn_gas/experiment.py --top-k 10 15 20 --arch deep_mlp resnet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import setup_output_dir, setup_console_logging, MLBaseConfig, OutputDir
from dataclasses import dataclass, field

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Constants (same as xgb_gas/train.py) ---
_META_COLS = {"simscode", "readingtime", "energy_per_sqft", "readingvalue",
              "grossarea", "buildingnumber"}
_SPARSE_PREFIXES = ("heat_", "steam_", "cooling_")
_ALWAYS_OFF_THRESHOLD = 0.999
ZERO_THRESHOLD = 1e-5


# ============================================================================
# Config
# ============================================================================

@dataclass
class NNGasConfig(MLBaseConfig):
    name: str = "gas_nn_experiment"
    model_type: str = "nn_experiment"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"plots": "plots"}
        )
    )


# ============================================================================
# Architectures
# ============================================================================

class DeepMLP(nn.Module):
    """Wide & deep MLP with BatchNorm, GELU, dropout."""

    def __init__(self, n_features, hidden_dims=(512, 512, 256, 256, 128, 64),
                 dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResBlock(nn.Module):
    """Residual block with projection shortcut when dims change."""

    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.block(x) + self.shortcut(x)))


class ResNetMLP(nn.Module):
    """MLP with residual skip connections."""

    def __init__(self, n_features, block_dims=(256, 256, 256, 128, 128, 64),
                 dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, block_dims[0]),
            nn.BatchNorm1d(block_dims[0]),
            nn.GELU(),
        )
        blocks = []
        for i in range(len(block_dims) - 1):
            blocks.append(ResBlock(block_dims[i], block_dims[i + 1], dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(block_dims[-1], 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


class TwoStageNet(nn.Module):
    """Multi-task: shared backbone with classifier head + regressor head.
    Loss = BCE(cls) + MSE(reg on non-zero) weighted.
    At inference: predicted = sigmoid(cls) * reg (soft blending)."""

    def __init__(self, n_features, backbone_dims=(512, 256, 128),
                 head_dims=(64, 32), dropout=0.3):
        super().__init__()
        # Shared backbone
        layers = []
        in_dim = n_features
        for h in backbone_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Classifier head
        cls_layers = []
        d = in_dim
        for h in head_dims:
            cls_layers.extend([nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)])
            d = h
        cls_layers.append(nn.Linear(d, 1))
        self.cls_head = nn.Sequential(*cls_layers)

        # Regressor head
        reg_layers = []
        d = in_dim
        for h in head_dims:
            reg_layers.extend([nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)])
            d = h
        reg_layers.append(nn.Linear(d, 1))
        self.reg_head = nn.Sequential(*reg_layers)

    def forward(self, x):
        feat = self.backbone(x)
        cls_logit = self.cls_head(feat).squeeze(-1)  # raw logit
        reg_out = self.reg_head(feat).squeeze(-1)     # regression
        return cls_logit, reg_out

    def predict(self, x):
        """Soft-blended prediction: sigmoid(cls) * reg."""
        cls_logit, reg_out = self.forward(x)
        prob = torch.sigmoid(cls_logit)
        return prob * reg_out


class EntityEmbedMLP(nn.Module):
    """Entity embedding for building ID + deep MLP for continuous features."""

    def __init__(self, n_continuous, n_buildings, embed_dim=32,
                 hidden_dims=(512, 256, 256, 128, 64), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(n_buildings, embed_dim)
        total_in = n_continuous + embed_dim
        layers = []
        in_dim = total_in
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cont, x_bldg_idx):
        emb = self.embedding(x_bldg_idx)
        x = torch.cat([x_cont, emb], dim=-1)
        return self.net(x).squeeze(-1)


# ============================================================================
# Training utilities
# ============================================================================

def train_regression_model(model, train_loader, val_loader, epochs=150,
                           lr=1e-3, weight_decay=1e-4, patience=20,
                           scheduler_type="cosine"):
    """Generic training loop for regression models."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_r2": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        train_loss = total_loss / n

        # Validate
        model.eval()
        val_preds, val_true = [], []
        total_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                pred = model(x)
                total_val_loss += F.mse_loss(pred, y).item() * len(y)
                n_val += len(y)
                val_preds.append(pred.cpu().numpy())
                val_true.append(y.cpu().numpy())
        val_loss = total_val_loss / n_val
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_r2 = float(r2_score(val_true, val_preds))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)

        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                  f"val_R²={val_r2:.4f} {'*' if wait == 0 else ''}")

        if wait >= patience:
            print(f"    Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_twostage_model(model, train_loader, val_loader, val_data,
                         epochs=150, lr=1e-3, weight_decay=1e-4, patience=20,
                         cls_weight=1.0):
    """Training loop for TwoStageNet with joint BCE + MSE loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_r2 = -float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_r2": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            x, y, y_cls = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            optimizer.zero_grad()
            cls_logit, reg_out = model(x)
            bce_loss = F.binary_cross_entropy_with_logits(cls_logit, y_cls)
            # Regression loss only on non-zero targets
            on_mask = y_cls > 0.5
            if on_mask.sum() > 0:
                mse_loss = F.mse_loss(reg_out[on_mask], y[on_mask])
            else:
                mse_loss = torch.tensor(0.0, device=DEVICE)
            loss = cls_weight * bce_loss + mse_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        train_loss = total_loss / n
        scheduler.step()

        # Validate: use soft-blended prediction for R²
        model.eval()
        with torch.no_grad():
            x_val = val_data["x"].to(DEVICE)
            pred = model.predict(x_val).cpu().numpy()
            y_true = val_data["y"].numpy()
        val_r2 = float(r2_score(y_true, pred))
        val_loss = float(mean_squared_error(y_true, pred))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.6f} val_R²={val_r2:.4f} "
                  f"{'*' if wait == 0 else ''}")

        if wait >= patience:
            print(f"    Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_entity_model(model, train_loader, val_loader, val_data,
                       epochs=150, lr=1e-3, weight_decay=1e-4, patience=20):
    """Training loop for EntityEmbedMLP."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_r2": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            x_cont, x_bldg, y = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_cont, x_bldg)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        train_loss = total_loss / n
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            x_c = val_data["x_cont"].to(DEVICE)
            x_b = val_data["x_bldg"].to(DEVICE)
            pred = model(x_c, x_b).cpu().numpy()
            y_true = val_data["y"].numpy()
        val_r2 = float(r2_score(y_true, pred))
        val_loss = float(mean_squared_error(y_true, pred))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.6f} val_R²={val_r2:.4f} "
                  f"{'*' if wait == 0 else ''}")

        if wait >= patience:
            print(f"    Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, X_test_tensor, y_test_np, scaler_y, arch_name,
                   bldg_idx_tensor=None):
    """Evaluate model and return metrics in original space."""
    model.eval()
    with torch.no_grad():
        x = X_test_tensor.to(DEVICE)
        if arch_name == "entity_embed" and bldg_idx_tensor is not None:
            b = bldg_idx_tensor.to(DEVICE)
            pred_scaled = model(x, b).cpu().numpy()
        elif arch_name == "twostage":
            pred_scaled = model.predict(x).cpu().numpy()
        else:
            pred_scaled = model(x).cpu().numpy()

    # Inverse-transform predictions and actuals
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual = scaler_y.inverse_transform(y_test_np.reshape(-1, 1)).ravel()

    # Clip negative predictions to 0
    pred = np.clip(pred, 0, None)

    r2 = float(r2_score(actual, pred))
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))

    # Non-zero subset
    nonzero = actual > ZERO_THRESHOLD
    r2_nz = float(r2_score(actual[nonzero], pred[nonzero])) if nonzero.sum() > 1 else float("nan")

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "r2_nonzero": r2_nz,
        "n_test": len(actual),
        "n_nonzero": int(nonzero.sum()),
    }


# ============================================================================
# Main experiment
# ============================================================================

def load_and_preprocess(parquet_path, split_date="2025-10-01"):
    """Load, clean, split — same logic as xgb_gas/train.py."""
    print(f"\n--- Loading Data ---")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols, "
          f"{df['simscode'].nunique()} buildings")

    # Drop sparse cross-utility cols
    sparse_cols = [c for c in df.columns
                   if any(c.startswith(p) for p in _SPARSE_PREFIXES)]
    if sparse_cols:
        print(f"  Dropping {len(sparse_cols)} sparse cross-utility columns")
        df = df.drop(columns=sparse_cols)

    # Separate always-off buildings
    target_col = "energy_per_sqft"
    bldg_zero_rate = df.groupby("simscode")[target_col].apply(
        lambda s: (np.abs(s) <= ZERO_THRESHOLD).mean()
    )
    always_off = bldg_zero_rate[bldg_zero_rate > _ALWAYS_OFF_THRESHOLD].index
    active_buildings = bldg_zero_rate[bldg_zero_rate <= _ALWAYS_OFF_THRESHOLD].index
    df_off = df[df["simscode"].isin(always_off)].copy()
    df = df[df["simscode"].isin(active_buildings)].reset_index(drop=True)
    print(f"  Active: {len(active_buildings)} buildings ({len(df):,} rows)")
    print(f"  Always-off: {len(always_off)} buildings ({len(df_off):,} rows)")

    # Derive features
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Temporal split
    split_dt = pd.Timestamp(split_date)
    train_mask = df["readingtime"] < split_dt
    test_mask = df["readingtime"] >= split_dt

    # Target-encode building identity (train-set only means)
    train_bldg_mean = df.loc[train_mask].groupby("simscode")[target_col].mean()
    global_mean = df.loc[train_mask, target_col].mean()
    df["building_target_enc"] = df["simscode"].map(train_bldg_mean).fillna(global_mean)
    feature_cols.append("building_target_enc")

    # Build building index mapping for entity embedding
    unique_buildings = sorted(df["simscode"].unique())
    bldg_to_idx = {b: i for i, b in enumerate(unique_buildings)}
    df["building_idx"] = df["simscode"].map(bldg_to_idx)

    X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
    X_test = df.loc[test_mask, feature_cols].values.astype(np.float32)
    y_train = df.loc[train_mask, target_col].values.astype(np.float32)
    y_test = df.loc[test_mask, target_col].values.astype(np.float32)
    bldg_train = df.loc[train_mask, "building_idx"].values.astype(np.int64)
    bldg_test = df.loc[test_mask, "building_idx"].values.astype(np.int64)

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Features: {len(feature_cols)}")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "bldg_train": bldg_train, "bldg_test": bldg_test,
        "feature_cols": feature_cols,
        "n_buildings": len(unique_buildings),
        "df_off": df_off,
        "split_date": split_date,
        "n_always_off": len(always_off),
        "n_active": len(active_buildings),
    }


def get_feature_importance(X_train, y_train, feature_cols, n_estimators=200):
    """Quick XGBoost to rank features."""
    print("\n--- Feature Importance (XGBoost) ---")
    xgb = XGBRegressor(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train, verbose=False)
    importance = xgb.feature_importances_
    ranked = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print("  Top 20 features:")
    for i, (fname, imp) in enumerate(ranked[:20]):
        print(f"    {i+1:2d}. {fname:30s} {imp:.4f}")
    return ranked


def select_top_k(ranked_features, k):
    """Return top-K feature names and their indices."""
    top = ranked_features[:k]
    return [f[0] for f in top]


def run_experiment(data, ranked_features, top_k, arch, epochs, lr, batch_size,
                   run_dir):
    """Run a single experiment: select features, train, evaluate."""
    selected = select_top_k(ranked_features, top_k)
    feature_cols = data["feature_cols"]
    feat_indices = [feature_cols.index(f) for f in selected]

    X_train = data["X_train"][:, feat_indices]
    X_test = data["X_test"][:, feat_indices]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Normalize features
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler_x.transform(X_test).astype(np.float32)

    # Normalize target
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)

    n_feat = len(selected)
    experiment_id = f"{arch}_k{top_k}"
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_id}")
    print(f"  Arch: {arch} | K={top_k} | Features: {selected[:5]}...")
    print(f"  Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    print(f"{'='*60}")

    if arch == "deep_mlp":
        model = DeepMLP(n_feat, hidden_dims=(512, 512, 256, 256, 128, 64),
                        dropout=0.3).to(DEVICE)
        train_ds = TensorDataset(
            torch.from_numpy(X_train_s), torch.from_numpy(y_train_s))
        val_ds = TensorDataset(
            torch.from_numpy(X_test_s), torch.from_numpy(y_test_s))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                                num_workers=2, pin_memory=True)
        model, history = train_regression_model(
            model, train_loader, val_loader, epochs=epochs, lr=lr, patience=20)
        metrics = evaluate_model(model, torch.from_numpy(X_test_s),
                                 y_test_s, scaler_y, arch)

    elif arch == "resnet":
        model = ResNetMLP(n_feat, block_dims=(256, 256, 256, 128, 128, 64),
                          dropout=0.3).to(DEVICE)
        train_ds = TensorDataset(
            torch.from_numpy(X_train_s), torch.from_numpy(y_train_s))
        val_ds = TensorDataset(
            torch.from_numpy(X_test_s), torch.from_numpy(y_test_s))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                                num_workers=2, pin_memory=True)
        model, history = train_regression_model(
            model, train_loader, val_loader, epochs=epochs, lr=lr, patience=20)
        metrics = evaluate_model(model, torch.from_numpy(X_test_s),
                                 y_test_s, scaler_y, arch)

    elif arch == "twostage":
        y_cls_train = (np.abs(y_train) > ZERO_THRESHOLD).astype(np.float32)
        y_cls_test = (np.abs(y_test) > ZERO_THRESHOLD).astype(np.float32)

        model = TwoStageNet(n_feat, backbone_dims=(512, 256, 128),
                            head_dims=(64, 32), dropout=0.3).to(DEVICE)
        train_ds = TensorDataset(
            torch.from_numpy(X_train_s),
            torch.from_numpy(y_train_s),
            torch.from_numpy(y_cls_train),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_test_s),
            torch.from_numpy(y_test_s),
            torch.from_numpy(y_cls_test),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                                num_workers=2, pin_memory=True)
        val_data = {
            "x": torch.from_numpy(X_test_s),
            "y": torch.from_numpy(y_test_s),
        }
        model, history = train_twostage_model(
            model, train_loader, val_loader, val_data,
            epochs=epochs, lr=lr, patience=20, cls_weight=0.5)
        metrics = evaluate_model(model, torch.from_numpy(X_test_s),
                                 y_test_s, scaler_y, "twostage")

    elif arch == "entity_embed":
        # Entity embedding uses continuous features + building index
        n_buildings = data["n_buildings"]
        bldg_train = data["bldg_train"]
        bldg_test = data["bldg_test"]

        model = EntityEmbedMLP(n_feat, n_buildings, embed_dim=32,
                               hidden_dims=(512, 256, 256, 128, 64),
                               dropout=0.3).to(DEVICE)
        train_ds = TensorDataset(
            torch.from_numpy(X_train_s),
            torch.from_numpy(bldg_train),
            torch.from_numpy(y_train_s),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_test_s),
            torch.from_numpy(bldg_test),
            torch.from_numpy(y_test_s),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                                num_workers=2, pin_memory=True)
        val_data = {
            "x_cont": torch.from_numpy(X_test_s),
            "x_bldg": torch.from_numpy(bldg_test),
            "y": torch.from_numpy(y_test_s),
        }
        model, history = train_entity_model(
            model, train_loader, val_loader, val_data,
            epochs=epochs, lr=lr, patience=20)
        metrics = evaluate_model(
            model, torch.from_numpy(X_test_s), y_test_s, scaler_y,
            "entity_embed", torch.from_numpy(bldg_test))

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Results:  R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.6f}  "
          f"MAE={metrics['mae']:.6f}  R²_nz={metrics['r2_nonzero']:.4f}")

    metrics["arch"] = arch
    metrics["top_k"] = top_k
    metrics["n_params"] = n_params
    metrics["features_used"] = selected
    metrics["epochs_trained"] = len(history["val_r2"])
    metrics["best_val_r2_scaled"] = float(max(history["val_r2"]))

    return metrics, history


def plot_results(all_results, run_dir):
    """Generate comparison plots."""
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)

    # 1. R² by architecture and K
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall R²
    ax = axes[0]
    for arch in df["arch"].unique():
        sub = df[df["arch"] == arch].sort_values("top_k")
        ax.plot(sub["top_k"], sub["r2"], "o-", label=arch, markersize=8)
    ax.set_xlabel("Top-K Features")
    ax.set_ylabel("R²")
    ax.set_title("Overall R² by Architecture & Feature Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.643, color="gray", linestyle="--", linewidth=1, label="XGBoost baseline")

    # Non-zero R²
    ax = axes[1]
    for arch in df["arch"].unique():
        sub = df[df["arch"] == arch].sort_values("top_k")
        ax.plot(sub["top_k"], sub["r2_nonzero"], "o-", label=arch, markersize=8)
    ax.set_xlabel("Top-K Features")
    ax.set_ylabel("R² (non-zero only)")
    ax.set_title("Non-zero Subset R² by Architecture & Feature Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Bar chart of best results per architecture
    best_per_arch = df.loc[df.groupby("arch")["r2"].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(best_per_arch["arch"], best_per_arch["r2"], color="steelblue",
                  edgecolor="black")
    ax.axhline(y=0.643, color="red", linestyle="--", linewidth=2,
               label="XGBoost 2-stage (R²=0.643)")
    for bar, r2 in zip(bars, best_per_arch["r2"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{r2:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Best R²")
    ax.set_title("Best R² per Architecture")
    ax.legend()
    ax.set_ylim(0, max(best_per_arch["r2"].max(), 0.65) * 1.1)
    fig.tight_layout()
    fig.savefig(plots_dir / "best_per_arch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {plots_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="NN experiments for gas prediction")
    parser.add_argument("--parquet", type=str,
                        default="data/tree_features_gas_cross.parquet")
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 15, 20, 30],
                        help="Feature counts to try")
    parser.add_argument("--arch", type=str, nargs="+",
                        default=["deep_mlp", "resnet", "twostage", "entity_embed"],
                        help="Architectures to try")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = NNGasConfig(name="gas_nn_experiment", seed=args.seed)
    run_dir = setup_output_dir(cfg)
    cleanup_logging = setup_console_logging(cfg, run_dir)

    print("=" * 60)
    print("Neural Network Gas Experiment")
    print(f"Device: {DEVICE}")
    print(f"Architectures: {args.arch}")
    print(f"Top-K values: {args.top_k}")
    print(f"Output: {run_dir}")
    print("=" * 60)

    t0 = time.time()

    try:
        # Load data
        data = load_and_preprocess(args.parquet)

        # Get feature importance from XGBoost
        ranked = get_feature_importance(
            data["X_train"], data["y_train"], data["feature_cols"])

        # Run all experiments
        all_results = []
        all_histories = {}

        for k in args.top_k:
            for arch in args.arch:
                try:
                    metrics, history = run_experiment(
                        data, ranked, k, arch, args.epochs, args.lr,
                        args.batch_size, run_dir)
                    all_results.append(metrics)
                    all_histories[f"{arch}_k{k}"] = history
                except Exception as e:
                    print(f"  FAILED: {arch} k={k}: {e}")
                    import traceback
                    traceback.print_exc()

        # Summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Architecture':<16} {'K':>3} {'R²':>8} {'R²_nz':>8} "
              f"{'RMSE':>10} {'Params':>10}")
        print("-" * 60)

        # Sort by R²
        for r in sorted(all_results, key=lambda x: -x["r2"]):
            print(f"{r['arch']:<16} {r['top_k']:>3} {r['r2']:>8.4f} "
                  f"{r['r2_nonzero']:>8.4f} {r['rmse']:>10.6f} "
                  f"{r['n_params']:>10,}")

        print("-" * 60)
        print(f"XGBoost 2-stage baseline:  R²=0.643")

        if all_results:
            best = max(all_results, key=lambda x: x["r2"])
            print(f"\nBest:  {best['arch']} k={best['top_k']} → "
                  f"R²={best['r2']:.4f}")

        # Save results
        results_path = run_dir / "results.json"
        results_path.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\nResults saved: {results_path}")

        # Plot
        if len(all_results) > 1:
            plot_results(all_results, run_dir)

        elapsed = time.time() - t0
        print(f"\nTotal time: {elapsed:.1f}s")
        print(f"Output: {run_dir}")

    finally:
        cleanup_logging()


if __name__ == "__main__":
    main()
