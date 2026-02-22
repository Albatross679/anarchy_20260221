#!/usr/bin/env python3
"""Postprocessing for neural-based model output instances.

Parses output directories from neural models (CNN, LSTM) and generates:
  - Train and validation loss over epochs
  - R² over epochs
  - Learning rate schedule over epochs
  - Text report: training time, hyperparameters, architecture, all metrics
  - Per-building performance breakdown (if predictions.parquet exists)

Usage:
    python postprocess_neural.py output/energy_cnn_20260221_204822
    python postprocess_neural.py output/energy_cnn_20260221_* output/energy_lstm_*
    python postprocess_neural.py output/energy_*  # auto-filters to neural models
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Model type detection ──────────────────────────────────────────────

NEURAL_MODEL_KEYS = {
    "cnn": "CNN",
    "lstm": "LSTM",
    "transformer": "Transformer",
}


def detect_model_type(config: dict) -> str:
    """Return human-readable model name from config.json."""
    for key, name in NEURAL_MODEL_KEYS.items():
        if key in config:
            return name
    run_name = config.get("name", "").lower()
    for key, name in NEURAL_MODEL_KEYS.items():
        if key in run_name:
            return name
    return "Unknown Neural Model"


def get_model_hparams(config: dict, model_name: str) -> dict:
    """Extract the model-specific hyperparameter block."""
    key_map = {v: k for k, v in NEURAL_MODEL_KEYS.items()}
    key = key_map.get(model_name)
    if key and key in config:
        return config[key]
    return {}


# ── Console log parsing ───────────────────────────────────────────────

def parse_training_time(log_text: str) -> str | None:
    """Extract 'Done in Xs' from console.log."""
    m = re.search(r"Done in ([\d.]+)s", log_text)
    return f"{float(m.group(1)):.1f}s" if m else None


def parse_epoch_metrics(log_text: str) -> pd.DataFrame:
    """Parse epoch-level metrics from console.log.

    Expected format:
      Epoch   1/100  train_loss=0.748396  val_loss=0.773831  R²=-0.0775  lr=1.00e-03  patience=0/999
    """
    pattern = (
        r"Epoch\s+(\d+)/(\d+)\s+"
        r"train_loss=([\d.e+-]+)\s+"
        r"val_loss=([\d.e+-]+)\s+"
        r"R[²2]=([\d.e+-]+)\s+"
        r"lr=([\d.e+-]+)\s+"
        r"patience=(\d+)/(\d+)"
    )
    rows = []
    for m in re.finditer(pattern, log_text):
        rows.append({
            "epoch": int(m.group(1)),
            "total_epochs": int(m.group(2)),
            "train_loss": float(m.group(3)),
            "val_loss": float(m.group(4)),
            "r2": float(m.group(5)),
            "lr": float(m.group(6)),
            "patience": int(m.group(7)),
            "max_patience": int(m.group(8)),
        })
    return pd.DataFrame(rows)


def parse_final_metrics_from_log(log_text: str) -> dict:
    """Parse final evaluation metrics from console.log."""
    metrics = {}
    for key, pattern in [
        ("rmse", r"RMSE:\s*([\d.e+-]+)"),
        ("mae", r"MAE:\s*([\d.e+-]+)"),
        ("r2", r"R[²2]:\s*([\d.e+-]+)"),
        ("mape_pct", r"MAPE:\s*([\d.e+-]+)%"),
    ]:
        m = re.search(pattern, log_text)
        if m:
            metrics[key] = float(m.group(1))
    return metrics


def parse_architecture(log_text: str) -> str | None:
    """Extract model architecture printout from console.log."""
    # Find the model class definition between '--- Model ---' and '--- Training ---'
    m = re.search(r"--- Model ---\n(.*?)--- Training ---", log_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def parse_data_summary(log_text: str) -> dict:
    """Extract data pipeline stats from console.log."""
    info = {}
    patterns = {
        "raw_readings": r"Raw meter readings:\s*([\d,]+)",
        "final_rows": r"Final dataset:\s*([\d,]+)\s*rows",
        "n_buildings": r"(\d+)\s*buildings",
        "train_rows": r"Train:\s*([\d,]+)\s*rows",
        "test_rows": r"Test:\s*([\d,]+)\s*rows",
        "train_windows": r"Train windows:\s*([\d,]+)",
        "test_windows": r"Test windows:\s*([\d,]+)",
        "parameters": r"Parameters:\s*([\d,]+)",
        "device": r"Device:\s*(\w+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, log_text)
        if m:
            info[key] = m.group(1)
    return info


# ── Per-building metrics ──────────────────────────────────────────────

def compute_per_building_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute R², RMSE, MAE per building from predictions.parquet."""
    target_col = "energy_per_sqft" if "energy_per_sqft" in df.columns else "readingvalue"
    group_col = "simscode" if "simscode" in df.columns else "buildingnumber"
    # Drop rows with NaN in target or predicted
    df = df.dropna(subset=[target_col, "predicted", "residual"])
    results = []
    for bld, grp in df.groupby(group_col):
        actual = grp[target_col]
        pred = grp["predicted"]
        n = len(grp)
        if n < 2:
            continue
        results.append({
            "building": bld,
            "n_samples": n,
            "rmse": np.sqrt(mean_squared_error(actual, pred)),
            "mae": mean_absolute_error(actual, pred),
            "r2": r2_score(actual, pred),
            "mean_residual": grp["residual"].mean(),
            "std_residual": grp["residual"].std(),
        })
    return pd.DataFrame(results).sort_values("r2", ascending=False).reset_index(drop=True)


# ── Plotting ──────────────────────────────────────────────────────────

def plot_loss_curves(epochs_df: pd.DataFrame, model_name: str, out_dir: Path):
    """Plot train and validation loss over epochs."""
    if epochs_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_df["epoch"], epochs_df["train_loss"], label="Train Loss", linewidth=1.5)
    ax.plot(epochs_df["epoch"], epochs_df["val_loss"], label="Validation Loss", linewidth=1.5)

    # Mark best validation epoch
    best_idx = epochs_df["val_loss"].idxmin()
    best_epoch = epochs_df.loc[best_idx, "epoch"]
    best_val = epochs_df.loc[best_idx, "val_loss"]
    ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.5)
    ax.scatter([best_epoch], [best_val], color="red", zorder=5, s=50,
               label=f"Best val loss={best_val:.4f} (epoch {best_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Train & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "postprocess_epoch_loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_r2_curve(epochs_df: pd.DataFrame, model_name: str, out_dir: Path):
    """Plot R² over epochs."""
    if epochs_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_df["epoch"], epochs_df["r2"], color="green", linewidth=1.5, label="Validation R²")

    best_idx = epochs_df["r2"].idxmax()
    best_epoch = epochs_df.loc[best_idx, "epoch"]
    best_r2 = epochs_df.loc[best_idx, "r2"]
    ax.scatter([best_epoch], [best_r2], color="red", zorder=5, s=50,
               label=f"Best R²={best_r2:.4f} (epoch {best_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title(f"{model_name} — Validation R² Over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(0, epochs_df["r2"].min() - 0.05), 1.0)
    fig.tight_layout()
    path = out_dir / "postprocess_epoch_r2.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_lr_schedule(epochs_df: pd.DataFrame, model_name: str, out_dir: Path):
    """Plot learning rate schedule over epochs."""
    if epochs_df.empty or "lr" not in epochs_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs_df["epoch"], epochs_df["lr"], color="purple", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"{model_name} — Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "postprocess_epoch_lr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined(epochs_df: pd.DataFrame, model_name: str, out_dir: Path):
    """Combined 3-panel plot: loss, R², learning rate."""
    if epochs_df.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Loss
    ax = axes[0]
    ax.plot(epochs_df["epoch"], epochs_df["train_loss"], label="Train Loss")
    ax.plot(epochs_df["epoch"], epochs_df["val_loss"], label="Val Loss")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Training Dashboard")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R²
    ax = axes[1]
    ax.plot(epochs_df["epoch"], epochs_df["r2"], color="green")
    ax.set_ylabel("Validation R²")
    ax.grid(True, alpha=0.3)

    # LR
    ax = axes[2]
    ax.plot(epochs_df["epoch"], epochs_df["lr"], color="purple")
    ax.set_ylabel("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "postprocess_training_dashboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_building_r2(bld_df: pd.DataFrame, model_name: str, out_dir: Path):
    """Histogram of per-building R² values."""
    if bld_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(bld_df["r2"], bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(bld_df["r2"].median(), color="red", linestyle="--",
               label=f'Median R²={bld_df["r2"].median():.4f}')
    ax.set_xlabel("R² per Building")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} — Per-Building R² Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "postprocess_per_building_r2.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residual_by_building(bld_df: pd.DataFrame, model_name: str, out_dir: Path):
    """Bar chart of mean residual per building (top 20 worst)."""
    if bld_df.empty:
        return
    worst = bld_df.nlargest(20, "std_residual")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(worst))
    ax.bar(x, worst["mean_residual"], yerr=worst["std_residual"], capsize=3, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(worst["building"], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Building")
    ax.set_ylabel("Mean Residual (± 1σ)")
    ax.set_title(f"{model_name} — Top 20 Buildings by Residual Variance")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "postprocess_residual_by_building.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pred_vs_actual(preds: pd.DataFrame, model_name: str, out_dir: Path):
    """Scatter plot of predicted vs actual values with identity line."""
    target_col = "energy_per_sqft" if "energy_per_sqft" in preds.columns else "readingvalue"
    df = preds.dropna(subset=[target_col, "predicted"])
    if df.empty:
        return
    actual = df[target_col].values
    predicted = df["predicted"].values

    # Subsample for plotting if too many points
    n = len(actual)
    if n > 50_000:
        idx = np.random.default_rng(42).choice(n, 50_000, replace=False)
        actual, predicted = actual[idx], predicted[idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual, predicted, alpha=0.15, s=3, rasterized=True)
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name} — Predicted vs Actual")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "postprocess_pred_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residual_distribution(preds: pd.DataFrame, model_name: str, out_dir: Path):
    """Histogram of residuals with normal fit overlay."""
    residuals = preds["residual"].dropna().values
    if len(residuals) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals, bins=100, density=True, edgecolor="black", alpha=0.7, label="Residuals")

    # Overlay normal fit
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
            "r-", linewidth=2, label=f"Normal (μ={mu:.4f}, σ={sigma:.4f})")
    ax.axvline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} — Residual Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "postprocess_residual_dist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residual_vs_predicted(preds: pd.DataFrame, model_name: str, out_dir: Path):
    """Residual vs predicted scatter — checks heteroscedasticity."""
    df = preds.dropna(subset=["predicted", "residual"])
    if df.empty:
        return
    predicted = df["predicted"].values
    residuals = df["residual"].values

    n = len(predicted)
    if n > 50_000:
        idx = np.random.default_rng(42).choice(n, 50_000, replace=False)
        predicted, residuals = predicted[idx], residuals[idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(predicted, residuals, alpha=0.15, s=3, rasterized=True)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(f"{model_name} — Residual vs Predicted")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "postprocess_residual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_error_by_hour(preds: pd.DataFrame, model_name: str, out_dir: Path):
    """Mean absolute error by hour of day — exposes temporal bias."""
    if "hour_of_day" not in preds.columns:
        if "readingtime" in preds.columns:
            preds = preds.copy()
            preds["hour_of_day"] = pd.to_datetime(preds["readingtime"]).dt.hour
        else:
            return
    df = preds.dropna(subset=["residual", "hour_of_day"])
    if df.empty:
        return

    hourly = df.groupby("hour_of_day")["residual"].agg(["mean", "std", "count"]).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Mean residual by hour (bias)
    ax = axes[0]
    ax.bar(hourly["hour_of_day"], hourly["mean"], color="steelblue", alpha=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("Mean Residual")
    ax.set_title(f"{model_name} — Residual by Hour of Day")
    ax.grid(True, alpha=0.3, axis="y")

    # Std residual by hour (uncertainty)
    ax = axes[1]
    ax.bar(hourly["hour_of_day"], hourly["std"], color="darkorange", alpha=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Residual Std Dev")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = out_dir / "postprocess_error_by_hour.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Text report ───────────────────────────────────────────────────────

def generate_text_report(
    model_name: str,
    config: dict,
    hparams: dict,
    final_metrics: dict,
    training_time: str | None,
    epochs_df: pd.DataFrame,
    data_info: dict,
    architecture: str | None,
    bld_df: pd.DataFrame | None,
    out_dir: Path,
):
    """Write a comprehensive text summary."""
    lines = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"POSTPROCESSING REPORT — {model_name}")
    lines.append(f"Output directory: {out_dir}")
    lines.append(sep)

    # Training summary
    lines.append("")
    lines.append("--- Training Summary ---")
    lines.append(f"  Model type:      {model_name}")
    lines.append(f"  Training time:   {training_time or 'N/A'}")
    lines.append(f"  Seed:            {config.get('seed', 'N/A')}")
    lines.append(f"  Utility:         {config.get('data', {}).get('utility_filter', 'N/A')}")
    lines.append(f"  Parameters:      {data_info.get('parameters', 'N/A')}")
    lines.append(f"  Device:          {data_info.get('device', 'N/A')}")

    # Architecture
    if architecture:
        lines.append("")
        lines.append("--- Architecture ---")
        for line in architecture.split("\n"):
            lines.append(f"  {line}")

    # Hyperparameters
    lines.append("")
    lines.append("--- Hyperparameters ---")
    if hparams:
        for k, v in hparams.items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append("  (not found in config)")

    # Data config
    data_cfg = config.get("data", {})
    lines.append("")
    lines.append("--- Data Configuration ---")
    lines.append(f"  Sequence length:   {data_cfg.get('seq_length', 'N/A')}")
    lines.append(f"  Stride:            {data_cfg.get('stride', 'N/A')}")
    lines.append(f"  Batch size:        {data_cfg.get('batch_size', 'N/A')}")
    lines.append(f"  Normalize feats:   {data_cfg.get('normalize_features', 'N/A')}")
    lines.append(f"  Normalize target:  {data_cfg.get('normalize_target', 'N/A')}")
    lines.append(f"  Split type:        {'Temporal' if data_cfg.get('temporal_split') else 'Random'}")
    lines.append(f"  Split date:        {data_cfg.get('split_date', 'N/A')}")

    # Data pipeline
    if data_info:
        lines.append("")
        lines.append("--- Data Pipeline ---")
        for k, v in data_info.items():
            lines.append(f"  {k}: {v}")

    # Epoch-level summary
    if not epochs_df.empty:
        lines.append("")
        lines.append("--- Epoch Training Summary ---")
        lines.append(f"  Total epochs:       {len(epochs_df)}")
        lines.append(f"  Total configured:   {epochs_df.iloc[0]['total_epochs']}")

        best_loss_idx = epochs_df["val_loss"].idxmin()
        best_r2_idx = epochs_df["r2"].idxmax()
        lines.append(f"  Best val loss:      {epochs_df.loc[best_loss_idx, 'val_loss']:.6f} (epoch {epochs_df.loc[best_loss_idx, 'epoch']})")
        lines.append(f"  Best val R²:        {epochs_df.loc[best_r2_idx, 'r2']:.4f} (epoch {epochs_df.loc[best_r2_idx, 'epoch']})")
        lines.append(f"  Final train loss:   {epochs_df.iloc[-1]['train_loss']:.6f}")
        lines.append(f"  Final val loss:     {epochs_df.iloc[-1]['val_loss']:.6f}")
        lines.append(f"  Final val R²:       {epochs_df.iloc[-1]['r2']:.4f}")

        # Overfitting analysis
        gap = epochs_df.iloc[-1]["val_loss"] - epochs_df.iloc[-1]["train_loss"]
        lines.append(f"  Train-Val gap:      {gap:.6f} ({'overfitting' if gap > 0 else 'underfitting'})")
        lines.append(f"  LR at start:        {epochs_df.iloc[0]['lr']:.2e}")
        lines.append(f"  LR at end:          {epochs_df.iloc[-1]['lr']:.2e}")

        # Convergence analysis
        if len(epochs_df) >= 10:
            last10 = epochs_df.tail(10)
            val_std = last10["val_loss"].std()
            lines.append(f"  Val loss std (last 10): {val_std:.6f} ({'converged' if val_std < 0.01 else 'still fluctuating'})")

    # Final evaluation metrics
    lines.append("")
    lines.append("--- Final Evaluation Metrics ---")
    if final_metrics:
        lines.append(f"  RMSE:           {final_metrics.get('rmse', 'N/A')}")
        lines.append(f"  MAE:            {final_metrics.get('mae', 'N/A')}")
        lines.append(f"  R²:             {final_metrics.get('r2', 'N/A')}")
        lines.append(f"  MAPE:           {final_metrics.get('mape_pct', 'N/A')}%")
    else:
        lines.append("  (no metrics available)")

    # Per-building summary
    if bld_df is not None and not bld_df.empty:
        lines.append("")
        lines.append("--- Per-Building Performance ---")
        lines.append(f"  Buildings evaluated:  {len(bld_df)}")
        lines.append(f"  Median R²:            {bld_df['r2'].median():.4f}")
        lines.append(f"  Mean R²:              {bld_df['r2'].mean():.4f}")
        lines.append(f"  R² range:             [{bld_df['r2'].min():.4f}, {bld_df['r2'].max():.4f}]")
        lines.append(f"  Buildings with R²>0.9: {(bld_df['r2'] > 0.9).sum()}")
        lines.append(f"  Buildings with R²<0:   {(bld_df['r2'] < 0).sum()}")

        lines.append("")
        lines.append("  Top 5 buildings (best R²):")
        for _, row in bld_df.head(5).iterrows():
            lines.append(f"    {row['building']:>10s}  R²={row['r2']:.4f}  RMSE={row['rmse']:.6e}  n={row['n_samples']}")

        lines.append("")
        lines.append("  Bottom 5 buildings (worst R²):")
        for _, row in bld_df.tail(5).iterrows():
            lines.append(f"    {row['building']:>10s}  R²={row['r2']:.4f}  RMSE={row['rmse']:.6e}  n={row['n_samples']}")

    lines.append("")
    lines.append(sep)
    report = "\n".join(lines)

    path = out_dir / "postprocess_summary.txt"
    path.write_text(report)
    print(f"  Saved: {path}")
    print()
    print(report)


# ── Main processing ──────────────────────────────────────────────────

def process_neural_output(out_dir: Path):
    """Process a single neural-based output directory."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {out_dir}")
    print(f"{'=' * 60}")

    # Load config
    config_path = out_dir / "config.json"
    if not config_path.exists():
        print(f"  ERROR: No config.json found in {out_dir}")
        return
    config = json.loads(config_path.read_text())
    model_name = detect_model_type(config)
    print(f"  Model type: {model_name}")

    hparams = get_model_hparams(config, model_name)

    # Load console.log
    log_path = out_dir / "console.log"
    log_text = log_path.read_text() if log_path.exists() else ""
    training_time = parse_training_time(log_text)
    architecture = parse_architecture(log_text)
    data_info = parse_data_summary(log_text)

    # Final metrics
    final_metrics = parse_final_metrics_from_log(log_text)

    # Parse epoch-by-epoch training logs
    epochs_df = parse_epoch_metrics(log_text)
    if not epochs_df.empty:
        print(f"  Parsed {len(epochs_df)} epochs")

        # Save epoch metrics CSV for further analysis
        epoch_csv_path = out_dir / "postprocess_epoch_metrics.csv"
        epochs_df.to_csv(epoch_csv_path, index=False)
        print(f"  Saved: {epoch_csv_path}")
    else:
        print(f"  No epoch-level metrics found in console.log")

    # Generate plots
    plot_loss_curves(epochs_df, model_name, out_dir)
    plot_r2_curve(epochs_df, model_name, out_dir)
    plot_lr_schedule(epochs_df, model_name, out_dir)
    plot_combined(epochs_df, model_name, out_dir)

    # Per-building and prediction-level analysis
    bld_df = None
    parquet_path = out_dir / "predictions.parquet"
    if parquet_path.exists():
        print(f"  Loading predictions.parquet...")
        preds = pd.read_parquet(parquet_path)
        bld_df = compute_per_building_metrics(preds)
        print(f"  Computed metrics for {len(bld_df)} buildings")
        plot_per_building_r2(bld_df, model_name, out_dir)
        plot_residual_by_building(bld_df, model_name, out_dir)

        # New prediction-level plots
        plot_pred_vs_actual(preds, model_name, out_dir)
        plot_residual_distribution(preds, model_name, out_dir)
        plot_residual_vs_predicted(preds, model_name, out_dir)
        plot_error_by_hour(preds, model_name, out_dir)

        # Save per-building CSV
        bld_csv_path = out_dir / "postprocess_per_building_metrics.csv"
        bld_df.to_csv(bld_csv_path, index=False)
        print(f"  Saved: {bld_csv_path}")
    else:
        print(f"  No predictions.parquet — skipping per-building analysis")

    # Text report
    generate_text_report(
        model_name, config, hparams, final_metrics,
        training_time, epochs_df, data_info, architecture,
        bld_df, out_dir,
    )


# ── Multi-model comparison ───────────────────────────────────────────

def generate_comparison(out_dirs: list[Path]):
    """Print a comparison table across multiple neural model runs."""
    if len(out_dirs) < 2:
        return

    rows = []
    for d in out_dirs:
        config_path = d / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        model_name = detect_model_type(config)
        log_text = (d / "console.log").read_text() if (d / "console.log").exists() else ""
        final_m = parse_final_metrics_from_log(log_text)
        epochs_df = parse_epoch_metrics(log_text)
        data_info = parse_data_summary(log_text)

        best_r2 = epochs_df["r2"].max() if not epochs_df.empty else None

        rows.append({
            "directory": d.name,
            "model": model_name,
            "params": data_info.get("parameters"),
            "epochs": len(epochs_df),
            "best_val_r2": best_r2,
            "final_rmse": final_m.get("rmse"),
            "final_r2": final_m.get("r2"),
            "time": parse_training_time(log_text),
        })

    if not rows:
        return

    comp_df = pd.DataFrame(rows).sort_values("final_r2", ascending=False, na_position="last")
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON (sorted by final R²)")
    print(f"{'=' * 80}")
    print(comp_df.to_string(index=False))
    print()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess neural-based model output directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dirs", nargs="+", type=Path,
        help="One or more output directories to process",
    )
    args = parser.parse_args()

    # Filter to valid neural model directories
    valid_dirs = []
    for d in args.dirs:
        if not d.is_dir():
            print(f"Skipping (not a directory): {d}")
            continue
        if not (d / "config.json").exists():
            print(f"Skipping (no config.json): {d}")
            continue
        config = json.loads((d / "config.json").read_text())
        model_name = detect_model_type(config)
        if model_name == "Unknown Neural Model":
            print(f"Skipping (not a neural model): {d}")
            continue
        valid_dirs.append(d)

    if not valid_dirs:
        print("No valid neural model output directories found.")
        sys.exit(1)

    for d in valid_dirs:
        process_neural_output(d)

    generate_comparison(valid_dirs)


if __name__ == "__main__":
    main()
