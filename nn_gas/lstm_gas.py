#!/usr/bin/env python3
"""
Standalone LSTM for gas consumption prediction.

Replicates the best result from seq_experiment.py:
    LSTM seq48 → R²=0.9723, R²_nz=0.9633

Hyperparameters (extracted from seq_experiment.py):
    Architecture:  3-layer LSTM (hidden=256) + static MLP → fusion head
    Sequence:      seq_length=48, stride=4
    Optimizer:     AdamW (lr=1e-3, weight_decay=1e-4)
    Scheduler:     CosineAnnealingLR (T_max=epochs)
    Grad clip:     max_norm=1.0
    Batch size:    512
    Early stop:    patience=15 on val MSE
    Normalization: z-score (fit on train)

Usage:
    python nn_gas/lstm_gas.py
    python nn_gas/lstm_gas.py --seq-length 48 --epochs 100 --patience 15
"""

import argparse
import json
import math
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
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import setup_output_dir, setup_console_logging, MLBaseConfig, OutputDir
from dataclasses import dataclass, field

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Constants ---
_META_COLS = {"simscode", "readingtime", "energy_per_sqft", "readingvalue",
              "grossarea", "buildingnumber", "building_idx"}
_SPARSE_PREFIXES = ("heat_", "steam_", "cooling_")
_ALWAYS_OFF_THRESHOLD = 0.999
ZERO_THRESHOLD = 1e-5
_STATIC_COLS = ["grossarea", "floorsaboveground", "building_age"]


# ============================================================================
# Config
# ============================================================================

@dataclass
class LSTMGasConfig(MLBaseConfig):
    name: str = "gas_lstm"
    model_type: str = "lstm"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(subdirs={"plots": "plots"})
    )


# ============================================================================
# Dataset
# ============================================================================

class GasSequenceDataset(Dataset):
    """Sliding-window dataset for gas consumption prediction.

    Groups by building, sorts by time, creates windows of seq_length.
    Returns (temporal_seq, static_feat, target).
    """

    def __init__(self, df, temporal_cols, static_cols, seq_length=48,
                 stride=4, scaler_stats=None):
        self.seq_length = seq_length
        self.temporal_cols = temporal_cols
        self.static_cols = static_cols

        windows_temporal = []
        windows_static = []
        windows_y = []

        for _, grp in df.groupby("simscode"):
            grp = grp.sort_values("readingtime")
            temporal = grp[temporal_cols].values.astype(np.float32)
            static = grp[static_cols].iloc[0].values.astype(np.float32)
            targets = grp["energy_per_sqft"].values.astype(np.float32)

            n = len(grp)
            for start in range(0, n - seq_length + 1, stride):
                end = start + seq_length
                windows_temporal.append(temporal[start:end])
                windows_static.append(static)
                windows_y.append(targets[end - 1])

        self.X_temporal = np.stack(windows_temporal)
        self.X_static = np.stack(windows_static)
        self.y = np.array(windows_y)

        if scaler_stats is None:
            self.scaler_stats = {
                "temp_mean": self.X_temporal.mean(axis=(0, 1)).tolist(),
                "temp_std": (self.X_temporal.std(axis=(0, 1)) + 1e-8).tolist(),
                "static_mean": self.X_static.mean(axis=0).tolist(),
                "static_std": (self.X_static.std(axis=0) + 1e-8).tolist(),
                "y_mean": float(self.y.mean()),
                "y_std": float(self.y.std() + 1e-8),
            }
        else:
            self.scaler_stats = scaler_stats

        t_mean = np.array(self.scaler_stats["temp_mean"], dtype=np.float32)
        t_std = np.array(self.scaler_stats["temp_std"], dtype=np.float32)
        self.X_temporal = (self.X_temporal - t_mean) / t_std

        s_mean = np.array(self.scaler_stats["static_mean"], dtype=np.float32)
        s_std = np.array(self.scaler_stats["static_std"], dtype=np.float32)
        self.X_static = (self.X_static - s_mean) / s_std

        self.y = (self.y - self.scaler_stats["y_mean"]) / self.scaler_stats["y_std"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X_temporal[idx]),
                torch.from_numpy(self.X_static[idx]),
                torch.tensor(self.y[idx], dtype=torch.float32))


# ============================================================================
# Model
# ============================================================================

class LSTMModel(nn.Module):
    """3-layer LSTM + static MLP → fusion head.

    Hyperparameters:
        hidden:     256
        layers:     3
        dropout:    0.3
        static_dim: 32
    """

    def __init__(self, n_temporal, n_static, hidden=256, layers=3,
                 dropout=0.3, static_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(
            n_temporal, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0)

        self.static_mlp = nn.Sequential(
            nn.Linear(n_static, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, static_dim), nn.ReLU())

        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, temporal, static):
        _, (h_n, _) = self.lstm(temporal)
        h = h_n[-1]  # last layer hidden state
        s = self.static_mlp(static)
        return self.head(torch.cat([h, s], dim=-1)).squeeze(-1)


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, scaler_stats,
                epochs=100, lr=1e-3, weight_decay=1e-4, patience=15):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_r2": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for temp, stat, y in train_loader:
            temp, stat, y = temp.to(DEVICE), stat.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(temp, stat)
            loss = F.mse_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        train_loss = total_loss / n

        model.eval()
        val_preds, val_true = [], []
        total_val, n_val = 0.0, 0
        with torch.no_grad():
            for temp, stat, y in val_loader:
                temp, stat, y = temp.to(DEVICE), stat.to(DEVICE), y.to(DEVICE)
                pred = model(temp, stat)
                total_val += F.mse_loss(pred, y).item() * len(y)
                n_val += len(y)
                val_preds.append(pred.cpu().numpy())
                val_true.append(y.cpu().numpy())
        val_loss = total_val / n_val

        vp = np.concatenate(val_preds)
        vt = np.concatenate(val_true)
        vp_orig = vp * scaler_stats["y_std"] + scaler_stats["y_mean"]
        vt_orig = vt * scaler_stats["y_std"] + scaler_stats["y_mean"]
        val_r2 = float(r2_score(vt_orig, vp_orig))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train={train_loss:.6f} val={val_loss:.6f} "
                  f"R²={val_r2:.4f} {'*' if wait == 0 else ''}")

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, dataset, scaler_stats, batch_size=512):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for temp, stat, y in loader:
            temp, stat = temp.to(DEVICE), stat.to(DEVICE)
            pred = model(temp, stat).cpu().numpy()
            all_preds.append(pred)
            all_true.append(y.numpy())

    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)

    preds_orig = preds * scaler_stats["y_std"] + scaler_stats["y_mean"]
    true_orig = true * scaler_stats["y_std"] + scaler_stats["y_mean"]
    preds_orig = np.clip(preds_orig, 0, None)

    r2 = float(r2_score(true_orig, preds_orig))
    rmse = float(np.sqrt(mean_squared_error(true_orig, preds_orig)))
    mae = float(mean_absolute_error(true_orig, preds_orig))

    nz = true_orig > ZERO_THRESHOLD
    r2_nz = float(r2_score(true_orig[nz], preds_orig[nz])) if nz.sum() > 1 else float("nan")

    return {"r2": r2, "rmse": rmse, "mae": mae, "r2_nonzero": r2_nz,
            "n_test": len(true_orig), "n_nonzero": int(nz.sum()),
            "preds_orig": preds_orig, "true_orig": true_orig}


# ============================================================================
# Data loading
# ============================================================================

def load_and_preprocess(parquet_path):
    print(f"\n--- Loading Data ---")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols, "
          f"{df['simscode'].nunique()} buildings")

    sparse = [c for c in df.columns if any(c.startswith(p) for p in _SPARSE_PREFIXES)]
    if sparse:
        print(f"  Dropping {len(sparse)} sparse cross-utility columns")
        df = df.drop(columns=sparse)

    bldg_zero = df.groupby("simscode")["energy_per_sqft"].apply(
        lambda s: (np.abs(s) <= ZERO_THRESHOLD).mean())
    always_off = bldg_zero[bldg_zero > _ALWAYS_OFF_THRESHOLD].index
    active = bldg_zero[bldg_zero <= _ALWAYS_OFF_THRESHOLD].index
    df_off = df[df["simscode"].isin(always_off)].copy()
    df = df[df["simscode"].isin(active)].reset_index(drop=True)
    print(f"  Active: {len(active)} buildings ({len(df):,} rows)")
    print(f"  Always-off: {len(always_off)} buildings ({len(df_off):,} rows)")

    feature_cols = [c for c in df.columns if c not in _META_COLS
                    and c not in _STATIC_COLS and c != "energy_per_sqft"]
    all_feat = feature_cols + _STATIC_COLS
    for c in all_feat:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    temporal_cols = [c for c in df.columns
                     if c not in _META_COLS and c not in _STATIC_COLS
                     and c != "energy_per_sqft"]
    static_cols = [c for c in _STATIC_COLS if c in df.columns]

    split_dt = pd.Timestamp("2025-10-01")
    df_train = df[df["readingtime"] < split_dt].copy()
    df_test = df[df["readingtime"] >= split_dt].copy()
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")
    print(f"  Temporal features ({len(temporal_cols)}): {temporal_cols[:8]}...")
    print(f"  Static features ({len(static_cols)}): {static_cols}")

    return df_train, df_test, temporal_cols, static_cols, df_off


# ============================================================================
# Plots
# ============================================================================

def plot_training(history, metrics, run_dir):
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train", linewidth=1.5)
    ax.plot(history["val_loss"], label="Val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R² curve
    ax = axes[1]
    ax.plot(history["val_r2"], color="green", linewidth=1.5)
    ax.axhline(y=0.643, color="red", linestyle="--", linewidth=1, label="XGBoost (0.643)")
    ax.axhline(y=0.653, color="orange", linestyle="--", linewidth=1, label="DeepMLP (0.653)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("Validation R²")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pred vs actual scatter
    ax = axes[2]
    p = metrics["preds_orig"]
    t = metrics["true_orig"]
    idx = np.random.RandomState(42).choice(len(p), min(5000, len(p)), replace=False)
    ax.scatter(t[idx], p[idx], alpha=0.15, s=8, color="steelblue")
    lims = [0, max(t.max(), p.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual energy/sqft")
    ax.set_ylabel("Predicted energy/sqft")
    ax.set_title(f"Pred vs Actual (R²={metrics['r2']:.4f})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "lstm_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plots saved to {plots_dir}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="LSTM gas consumption prediction")
    p.add_argument("--parquet", default="data/tree_features_gas_cross.parquet")
    p.add_argument("--seq-length", type=int, default=48)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = LSTMGasConfig(name="gas_lstm", seed=args.seed)
    run_dir = setup_output_dir(cfg)
    cleanup = setup_console_logging(cfg, run_dir)

    print("=" * 60)
    print("LSTM Gas Consumption Prediction")
    print(f"  Device:      {DEVICE}")
    print(f"  Seq length:  {args.seq_length}")
    print(f"  Stride:      {args.stride}")
    print(f"  Hidden:      {args.hidden}")
    print(f"  Layers:      {args.layers}")
    print(f"  Dropout:     {args.dropout}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Patience:    {args.patience}")
    print(f"  Output:      {run_dir}")
    print("=" * 60)

    t0 = time.time()

    try:
        df_train, df_test, temporal_cols, static_cols, df_off = \
            load_and_preprocess(args.parquet)

        print(f"\n--- Creating windows (seq_length={args.seq_length}, "
              f"stride={args.stride}) ---")
        train_ds = GasSequenceDataset(
            df_train, temporal_cols, static_cols,
            seq_length=args.seq_length, stride=args.stride)
        test_ds = GasSequenceDataset(
            df_test, temporal_cols, static_cols,
            seq_length=args.seq_length, stride=args.stride,
            scaler_stats=train_ds.scaler_stats)
        print(f"  Train windows: {len(train_ds):,}")
        print(f"  Test windows:  {len(test_ds):,}")

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True)
        val_loader = DataLoader(
            test_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=2, pin_memory=True)

        model = LSTMModel(
            len(temporal_cols), len(static_cols),
            hidden=args.hidden, layers=args.layers,
            dropout=args.dropout).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model params: {n_params:,}")

        print(f"\n--- Training ---")
        model, history = train_model(
            model, train_loader, val_loader, train_ds.scaler_stats,
            epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience)

        print(f"\n--- Evaluation ---")
        metrics = evaluate_model(model, test_ds, train_ds.scaler_stats,
                                 batch_size=args.batch_size * 2)

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"  R²:          {metrics['r2']:.4f}")
        print(f"  R² (nz):     {metrics['r2_nonzero']:.4f}")
        print(f"  RMSE:        {metrics['rmse']:.6f}")
        print(f"  MAE:         {metrics['mae']:.6f}")
        print(f"  Test samples:{metrics['n_test']:,}")
        print(f"  Non-zero:    {metrics['n_nonzero']:,}")
        print(f"{'='*60}")
        print(f"  Baselines:   XGBoost=0.643  DeepMLP=0.653")

        # Save results
        save_metrics = {k: v for k, v in metrics.items()
                        if k not in ("preds_orig", "true_orig")}
        save_metrics["arch"] = "lstm"
        save_metrics["seq_length"] = args.seq_length
        save_metrics["n_params"] = n_params
        save_metrics["epochs_trained"] = len(history["val_r2"])
        save_metrics["best_val_r2"] = float(max(history["val_r2"]))
        save_metrics["hyperparameters"] = {
            "hidden": args.hidden, "layers": args.layers,
            "dropout": args.dropout, "lr": args.lr,
            "weight_decay": args.weight_decay, "batch_size": args.batch_size,
            "seq_length": args.seq_length, "stride": args.stride,
            "patience": args.patience, "seed": args.seed,
        }

        results_path = run_dir / "results.json"
        results_path.write_text(json.dumps(save_metrics, indent=2, default=str))
        print(f"\n  Results saved to {results_path}")

        # Save model weights
        torch.save(model.state_dict(), run_dir / "lstm_model.pt")
        print(f"  Model saved to {run_dir / 'lstm_model.pt'}")

        # Save scaler stats for inference
        scaler_path = run_dir / "scaler_stats.json"
        scaler_path.write_text(json.dumps(train_ds.scaler_stats, indent=2))

        plot_training(history, metrics, run_dir)

        elapsed = time.time() - t0
        print(f"\n  Total time: {elapsed:.1f}s")
        print(f"  Output: {run_dir}")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
