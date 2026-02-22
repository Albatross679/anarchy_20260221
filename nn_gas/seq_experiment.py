#!/usr/bin/env python3
"""
Sequence-based neural network experiments for gas consumption prediction.

Creates sliding windows over per-building time series, then trains:
    A. LSTM              — standard LSTM + static building MLP
    B. BiLSTM            — bidirectional LSTM
    C. LSTM + Attention  — LSTM with temporal self-attention
    D. Transformer       — pure transformer encoder + static MLP
    E. GRU               — gated recurrent unit variant

Usage:
    python nn_gas/seq_experiment.py
    python nn_gas/seq_experiment.py --seq-length 48 96 --arch lstm bilstm transformer
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

# Static features (constant per building)
_STATIC_COLS = ["grossarea", "floorsaboveground", "building_age"]


# ============================================================================
# Config
# ============================================================================

@dataclass
class SeqGasConfig(MLBaseConfig):
    name: str = "gas_seq_experiment"
    model_type: str = "seq_experiment"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(subdirs={"plots": "plots"})
    )


# ============================================================================
# Dataset
# ============================================================================

class GasSequenceDataset(Dataset):
    """Sliding-window dataset for gas consumption prediction.

    Groups by building (simscode), sorts by time, creates windows.
    Returns (temporal_seq, static_feat, target).
    """

    def __init__(self, df, temporal_cols, static_cols, seq_length=96,
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

        self.X_temporal = np.stack(windows_temporal)  # (N, seq, n_temp)
        self.X_static = np.stack(windows_static)      # (N, n_static)
        self.y = np.array(windows_y)                   # (N,)

        # Compute or apply normalization
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

        # Normalize
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
# Architectures
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM + static MLP → fusion head."""

    def __init__(self, n_temporal, n_static, hidden=256, layers=3,
                 dropout=0.3, bidirectional=False, static_dim=32):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            n_temporal, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0, bidirectional=bidirectional)
        lstm_out = hidden * (2 if bidirectional else 1)

        self.static_mlp = nn.Sequential(
            nn.Linear(n_static, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, static_dim), nn.ReLU())

        self.head = nn.Sequential(
            nn.Linear(lstm_out + static_dim, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, temporal, static):
        _, (h_n, _) = self.lstm(temporal)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]
        s = self.static_mlp(static)
        return self.head(torch.cat([h, s], dim=-1)).squeeze(-1)


class LSTMAttention(nn.Module):
    """LSTM + temporal attention + static MLP."""

    def __init__(self, n_temporal, n_static, hidden=256, layers=2,
                 dropout=0.3, n_heads=4, static_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(
            n_temporal, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0)

        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout,
                                          batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)

        self.static_mlp = nn.Sequential(
            nn.Linear(n_static, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, static_dim), nn.ReLU())

        self.head = nn.Sequential(
            nn.Linear(hidden + static_dim, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, temporal, static):
        lstm_out, _ = self.lstm(temporal)  # (B, seq, hidden)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)
        # Take last timestep after attention
        h = attn_out[:, -1, :]
        s = self.static_mlp(static)
        return self.head(torch.cat([h, s], dim=-1)).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    """Transformer encoder + static MLP."""

    def __init__(self, n_temporal, n_static, d_model=128, n_heads=4,
                 n_layers=4, d_ff=256, dropout=0.2, static_dim=32):
        super().__init__()
        self.input_proj = nn.Linear(n_temporal, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.static_mlp = nn.Sequential(
            nn.Linear(n_static, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, static_dim), nn.ReLU())

        self.head = nn.Sequential(
            nn.Linear(d_model + static_dim, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, temporal, static):
        x = self.input_proj(temporal)
        x = self.pos_enc(x)
        x = self.encoder(x)
        h = x.mean(dim=1)  # mean pooling
        s = self.static_mlp(static)
        return self.head(torch.cat([h, s], dim=-1)).squeeze(-1)


class GRUModel(nn.Module):
    """GRU variant + static MLP."""

    def __init__(self, n_temporal, n_static, hidden=256, layers=3,
                 dropout=0.3, static_dim=32):
        super().__init__()
        self.gru = nn.GRU(
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
        _, h_n = self.gru(temporal)
        h = h_n[-1]
        s = self.static_mlp(static)
        return self.head(torch.cat([h, s], dim=-1)).squeeze(-1)


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, scaler_stats,
                epochs=100, lr=1e-3, weight_decay=1e-4, patience=15):
    """Train loop with early stopping. Returns model and history."""
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

        # R² in original space
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

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"train={train_loss:.6f} val={val_loss:.6f} "
                  f"R²={val_r2:.4f} {'*' if wait == 0 else ''}")

        if wait >= patience:
            print(f"    Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, dataset, scaler_stats, batch_size=512):
    """Evaluate model, return metrics in original space."""
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

    # Denormalize
    preds_orig = preds * scaler_stats["y_std"] + scaler_stats["y_mean"]
    true_orig = true * scaler_stats["y_std"] + scaler_stats["y_mean"]
    preds_orig = np.clip(preds_orig, 0, None)

    r2 = float(r2_score(true_orig, preds_orig))
    rmse = float(np.sqrt(mean_squared_error(true_orig, preds_orig)))
    mae = float(mean_absolute_error(true_orig, preds_orig))

    nz = true_orig > ZERO_THRESHOLD
    r2_nz = float(r2_score(true_orig[nz], preds_orig[nz])) if nz.sum() > 1 else float("nan")

    return {"r2": r2, "rmse": rmse, "mae": mae, "r2_nonzero": r2_nz,
            "n_test": len(true_orig), "n_nonzero": int(nz.sum())}


# ============================================================================
# Main
# ============================================================================

def load_and_preprocess(parquet_path):
    """Load gas parquet and preprocess (same as nn_gas/experiment.py)."""
    print(f"\n--- Loading Data ---")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols, "
          f"{df['simscode'].nunique()} buildings")

    # Drop sparse cross-utility cols
    sparse = [c for c in df.columns if any(c.startswith(p) for p in _SPARSE_PREFIXES)]
    if sparse:
        print(f"  Dropping {len(sparse)} sparse cross-utility columns")
        df = df.drop(columns=sparse)

    # Separate always-off buildings
    bldg_zero = df.groupby("simscode")["energy_per_sqft"].apply(
        lambda s: (np.abs(s) <= ZERO_THRESHOLD).mean())
    always_off = bldg_zero[bldg_zero > _ALWAYS_OFF_THRESHOLD].index
    active = bldg_zero[bldg_zero <= _ALWAYS_OFF_THRESHOLD].index
    df_off = df[df["simscode"].isin(always_off)].copy()
    df = df[df["simscode"].isin(active)].reset_index(drop=True)
    print(f"  Active: {len(active)} buildings ({len(df):,} rows)")
    print(f"  Always-off: {len(always_off)} buildings ({len(df_off):,} rows)")

    # Fill NaN
    feature_cols = [c for c in df.columns if c not in _META_COLS
                    and c not in _STATIC_COLS and c != "energy_per_sqft"]
    all_feat = feature_cols + _STATIC_COLS
    for c in all_feat:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Temporal cols = everything except meta, static, target
    temporal_cols = [c for c in df.columns
                     if c not in _META_COLS and c not in _STATIC_COLS
                     and c != "energy_per_sqft"]
    static_cols = [c for c in _STATIC_COLS if c in df.columns]

    # Temporal split
    split_dt = pd.Timestamp("2025-10-01")
    df_train = df[df["readingtime"] < split_dt].copy()
    df_test = df[df["readingtime"] >= split_dt].copy()
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")
    print(f"  Temporal features ({len(temporal_cols)}): {temporal_cols[:8]}...")
    print(f"  Static features ({len(static_cols)}): {static_cols}")

    return df_train, df_test, temporal_cols, static_cols, df_off


def build_model(arch, n_temporal, n_static):
    """Create model by architecture name."""
    if arch == "lstm":
        return LSTMModel(n_temporal, n_static, hidden=256, layers=3,
                         dropout=0.3, bidirectional=False)
    elif arch == "bilstm":
        return LSTMModel(n_temporal, n_static, hidden=192, layers=2,
                         dropout=0.3, bidirectional=True)
    elif arch == "lstm_attn":
        return LSTMAttention(n_temporal, n_static, hidden=256, layers=2,
                             dropout=0.3, n_heads=4)
    elif arch == "transformer":
        return TransformerModel(n_temporal, n_static, d_model=128, n_heads=4,
                                n_layers=4, d_ff=256, dropout=0.2)
    elif arch == "gru":
        return GRUModel(n_temporal, n_static, hidden=256, layers=3, dropout=0.3)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def plot_results(results, run_dir):
    """Generate comparison plots."""
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)

    # R² by arch and seq_length
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for arch in df["arch"].unique():
        sub = df[df["arch"] == arch].sort_values("seq_length")
        ax.plot(sub["seq_length"], sub["r2"], "o-", label=arch, markersize=8)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("R²")
    ax.set_title("Overall R² by Architecture & Sequence Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.643, color="gray", linestyle="--", linewidth=1,
               label="XGBoost baseline")
    ax.axhline(y=0.653, color="green", linestyle="--", linewidth=1,
               label="DeepMLP best")

    ax = axes[1]
    for arch in df["arch"].unique():
        sub = df[df["arch"] == arch].sort_values("seq_length")
        ax.plot(sub["seq_length"], sub["r2_nonzero"], "o-", label=arch, markersize=8)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("R² (non-zero only)")
    ax.set_title("Non-zero Subset R²")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "seq_comparison_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Bar chart best per arch
    best = df.loc[df.groupby("arch")["r2"].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(best["arch"], best["r2"], color="steelblue", edgecolor="black")
    ax.axhline(y=0.643, color="red", linestyle="--", linewidth=2,
               label="XGBoost 2-stage (0.643)")
    ax.axhline(y=0.653, color="green", linestyle="--", linewidth=2,
               label="DeepMLP best (0.653)")
    for bar, r2 in zip(bars, best["r2"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{r2:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Best R²")
    ax.set_title("Best R² per Sequence Architecture")
    ax.legend()
    ax.set_ylim(0, max(best["r2"].max(), 0.66) * 1.1)
    fig.tight_layout()
    fig.savefig(plots_dir / "seq_best_per_arch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {plots_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/tree_features_gas_cross.parquet")
    p.add_argument("--seq-length", type=int, nargs="+", default=[48, 96])
    p.add_argument("--arch", type=str, nargs="+",
                   default=["lstm", "bilstm", "lstm_attn", "transformer", "gru"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = SeqGasConfig(name="gas_seq_experiment", seed=args.seed)
    run_dir = setup_output_dir(cfg)
    cleanup = setup_console_logging(cfg, run_dir)

    print("=" * 60)
    print("Sequence-Based Gas NN Experiment")
    print(f"Device: {DEVICE}")
    print(f"Architectures: {args.arch}")
    print(f"Seq lengths: {args.seq_length}")
    print(f"Output: {run_dir}")
    print("=" * 60)

    t0 = time.time()

    try:
        df_train, df_test, temporal_cols, static_cols, df_off = \
            load_and_preprocess(args.parquet)

        all_results = []

        for seq_len in args.seq_length:
            # Create datasets for this seq_length
            print(f"\n--- Creating windows (seq_length={seq_len}, "
                  f"stride={args.stride}) ---")
            train_ds = GasSequenceDataset(
                df_train, temporal_cols, static_cols,
                seq_length=seq_len, stride=args.stride)
            test_ds = GasSequenceDataset(
                df_test, temporal_cols, static_cols,
                seq_length=seq_len, stride=args.stride,
                scaler_stats=train_ds.scaler_stats)
            print(f"  Train windows: {len(train_ds):,}")
            print(f"  Test windows:  {len(test_ds):,}")

            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=2, pin_memory=True)
            val_loader = DataLoader(
                test_ds, batch_size=args.batch_size * 2, shuffle=False,
                num_workers=2, pin_memory=True)

            for arch in args.arch:
                exp_id = f"{arch}_seq{seq_len}"
                print(f"\n{'='*60}")
                print(f"  Experiment: {exp_id}")
                print(f"{'='*60}")

                try:
                    torch.manual_seed(args.seed)
                    model = build_model(
                        arch, len(temporal_cols), len(static_cols)).to(DEVICE)
                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"  Params: {n_params:,}")

                    model, history = train_model(
                        model, train_loader, val_loader, train_ds.scaler_stats,
                        epochs=args.epochs, lr=args.lr,
                        weight_decay=1e-4, patience=args.patience)

                    metrics = evaluate_model(
                        model, test_ds, train_ds.scaler_stats,
                        batch_size=args.batch_size * 2)

                    print(f"  Results: R²={metrics['r2']:.4f} "
                          f"RMSE={metrics['rmse']:.6f} "
                          f"R²_nz={metrics['r2_nonzero']:.4f}")

                    metrics["arch"] = arch
                    metrics["seq_length"] = seq_len
                    metrics["n_params"] = n_params
                    metrics["epochs_trained"] = len(history["val_r2"])
                    metrics["best_val_r2"] = float(max(history["val_r2"]))
                    all_results.append(metrics)

                except Exception as e:
                    print(f"  FAILED: {exp_id}: {e}")
                    import traceback
                    traceback.print_exc()

        # Summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Architecture':<16} {'Seq':>4} {'R²':>8} {'R²_nz':>8} "
              f"{'RMSE':>10} {'Params':>10} {'Epochs':>6}")
        print("-" * 66)

        for r in sorted(all_results, key=lambda x: -x["r2"]):
            print(f"{r['arch']:<16} {r['seq_length']:>4} {r['r2']:>8.4f} "
                  f"{r['r2_nonzero']:>8.4f} {r['rmse']:>10.6f} "
                  f"{r['n_params']:>10,} {r['epochs_trained']:>6}")

        print("-" * 66)
        print(f"XGBoost 2-stage baseline:     R²=0.643")
        print(f"DeepMLP (tabular) best:       R²=0.653")

        if all_results:
            best = max(all_results, key=lambda x: x["r2"])
            print(f"\nBest seq model: {best['arch']} seq={best['seq_length']} → "
                  f"R²={best['r2']:.4f}")

        # Save
        results_path = run_dir / "results.json"
        results_path.write_text(json.dumps(all_results, indent=2, default=str))

        if len(all_results) > 1:
            plot_results(all_results, run_dir)

        elapsed = time.time() - t0
        print(f"\nTotal time: {elapsed:.1f}s")
        print(f"Output: {run_dir}")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
