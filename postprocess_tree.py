#!/usr/bin/env python3
"""Postprocessing for tree-based model output instances.

Parses output directories from tree-based models (XGBoost, LightGBM, CatBoost,
DART, Random Forest, Extra Trees, NGBoost, QRF) and generates:
  - RMSE over training rounds (for boosting models)
  - Final metrics summary table
  - Text report: training time, hyperparameters, all metrics
  - Per-building performance breakdown (if predictions.parquet exists)

Usage:
    python postprocess_tree.py output/energy_xgboost_20260222_001425
    python postprocess_tree.py output/energy_xgboost_20260222_001425 output/energy_lightgbm_20260222_001846
    python postprocess_tree.py output/energy_*  # all tree outputs
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

TREE_MODEL_KEYS = {
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "lgb": "LightGBM",
    "cb": "CatBoost",
    "catboost": "CatBoost",
    "dart": "DART",
    "rf": "Random Forest",
    "et": "Extra Trees",
    "ngb": "NGBoost",
    "qrf": "Quantile RF",
}

BOOSTING_MODELS = {"XGBoost", "LightGBM", "CatBoost", "DART"}


def detect_model_type(config: dict) -> str:
    """Return human-readable model name from config.json."""
    for key, name in TREE_MODEL_KEYS.items():
        if key in config:
            return name
    # Fallback: infer from run name
    run_name = config.get("name", "").lower()
    for key, name in TREE_MODEL_KEYS.items():
        if key in run_name or name.lower().replace(" ", "_") in run_name:
            return name
    return "Unknown Tree Model"


def get_model_hparams(config: dict, model_name: str) -> dict:
    """Extract the model-specific hyperparameter block."""
    key_candidates = {
        "XGBoost": ["xgb"],
        "LightGBM": ["lgbm", "lgb"],
        "CatBoost": ["cb", "catboost"],
        "DART": ["dart"],
        "Random Forest": ["rf"],
        "Extra Trees": ["et"],
        "NGBoost": ["ngb"],
        "Quantile RF": ["qrf"],
    }
    for key in key_candidates.get(model_name, []):
        if key in config:
            return config[key]
    return {}


# ── Console log parsing ───────────────────────────────────────────────

def parse_training_time(log_text: str) -> str | None:
    """Extract 'Done in Xs' from console.log."""
    m = re.search(r"Done in ([\d.]+)s", log_text)
    return f"{float(m.group(1)):.1f}s" if m else None


def parse_round_metrics_xgb(log_text: str) -> pd.DataFrame:
    """Parse XGBoost/DART round logs: [N] validation_0-rmse:X validation_1-rmse:Y"""
    pattern = r"\[(\d+)\]\s*validation_0-rmse:([\d.e+-]+)\s*validation_1-rmse:([\d.e+-]+)"
    rows = []
    for m in re.finditer(pattern, log_text):
        rows.append({
            "round": int(m.group(1)),
            "train_rmse": float(m.group(2)),
            "val_rmse": float(m.group(3)),
        })
    return pd.DataFrame(rows)


def parse_round_metrics_lgb(log_text: str) -> pd.DataFrame:
    """Parse LightGBM round logs: [N] train's rmse: X validation's rmse: Y"""
    pattern = r"\[(\d+)\]\s*train's rmse:\s*([\d.e+-]+)\s*validation's rmse:\s*([\d.e+-]+)"
    rows = []
    for m in re.finditer(pattern, log_text):
        rows.append({
            "round": int(m.group(1)),
            "train_rmse": float(m.group(2)),
            "val_rmse": float(m.group(3)),
        })
    return pd.DataFrame(rows)


def parse_round_metrics_catboost(log_text: str) -> pd.DataFrame:
    """Parse CatBoost round logs: N: learn: X test: Y best: Z (iter)"""
    pattern = r"(\d+):\s*learn:\s*([\d.e+-]+)\s*test:\s*([\d.e+-]+)"
    rows = []
    for m in re.finditer(pattern, log_text):
        rows.append({
            "round": int(m.group(1)),
            "train_rmse": float(m.group(2)),
            "val_rmse": float(m.group(3)),
        })
    return pd.DataFrame(rows)


def parse_round_metrics(log_text: str, model_name: str) -> pd.DataFrame:
    """Dispatch to correct parser based on model type."""
    if model_name in ("XGBoost", "DART"):
        return parse_round_metrics_xgb(log_text)
    elif model_name == "LightGBM":
        return parse_round_metrics_lgb(log_text)
    elif model_name == "CatBoost":
        return parse_round_metrics_catboost(log_text)
    return pd.DataFrame()


def parse_final_metrics_from_log(log_text: str) -> dict:
    """Fallback: parse final metrics from console.log evaluation section."""
    metrics = {}
    for key, pattern in [
        ("rmse", r"RMSE:\s*([\d.e+-]+)"),
        ("mae", r"MAE:\s*([\d.e+-]+)"),
        ("r2", r"R[²2]:\s*([\d.e+-]+)"),
        ("mape_pct", r"MAPE:\s*([\d.e+-]+)%"),
        ("n_trees_used", r"Trees:\s*(\d+)"),
    ]:
        m = re.search(pattern, log_text)
        if m:
            val = m.group(1)
            metrics[key] = int(val) if key == "n_trees_used" else float(val)
    return metrics


# ── Per-building metrics ──────────────────────────────────────────────

def compute_per_building_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute R², RMSE, MAE per building from predictions.parquet."""
    # Target is energy_per_sqft (predicted is in same scale)
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

def plot_round_metrics(rounds_df: pd.DataFrame, model_name: str, out_dir: Path):
    """Plot RMSE over training rounds for boosting models."""
    if rounds_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds_df["round"], rounds_df["train_rmse"], label="Train RMSE", linewidth=1.5)
    ax.plot(rounds_df["round"], rounds_df["val_rmse"], label="Validation RMSE", linewidth=1.5)
    ax.set_xlabel("Round (Boosting Iteration)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{model_name} — RMSE Over Training Rounds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "postprocess_round_rmse.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_building_r2(bld_df: pd.DataFrame, model_name: str, out_dir: Path):
    """Histogram of per-building R² values."""
    if bld_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(bld_df["r2"], bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(bld_df["r2"].median(), color="red", linestyle="--", label=f'Median R²={bld_df["r2"].median():.4f}')
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


def plot_feature_importance(final_metrics: dict, model_name: str, out_dir: Path):
    """Horizontal bar chart of top feature importances from metrics.json."""
    top_features = final_metrics.get("top_features", {})
    if not top_features:
        return
    features = list(top_features.keys())
    importances = [float(v) for v in top_features.values()]

    fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.4)))
    y = np.arange(len(features))
    ax.barh(y, importances, align="center", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} — Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    path = out_dir / "postprocess_feature_importance.png"
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
        # Try to derive from readingtime
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
    rounds_df: pd.DataFrame,
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

    # Training time
    lines.append("")
    lines.append("--- Training Summary ---")
    lines.append(f"  Model type:     {model_name}")
    lines.append(f"  Training time:  {training_time or 'N/A'}")
    lines.append(f"  Seed:           {config.get('seed', 'N/A')}")
    lines.append(f"  Utility:        {config.get('data', {}).get('utility_filter', 'N/A')}")

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
    lines.append(f"  Split type:       {'Temporal' if data_cfg.get('temporal_split') else 'Random'}")
    lines.append(f"  Split date:       {data_cfg.get('split_date', 'N/A')}")
    lines.append(f"  Lag hours:        {data_cfg.get('lag_hours', 'N/A')}")
    lines.append(f"  Rolling windows:  {data_cfg.get('rolling_windows', 'N/A')}")
    lines.append(f"  Interactions:     {data_cfg.get('add_interactions', False)}")

    # Final metrics
    lines.append("")
    lines.append("--- Final Evaluation Metrics ---")
    if final_metrics:
        lines.append(f"  RMSE:           {final_metrics.get('rmse', 'N/A')}")
        lines.append(f"  MAE:            {final_metrics.get('mae', 'N/A')}")
        lines.append(f"  R²:             {final_metrics.get('r2', 'N/A')}")
        lines.append(f"  MAPE:           {final_metrics.get('mape_pct', 'N/A')}%")
        lines.append(f"  Test samples:   {final_metrics.get('n_test', 'N/A')}")
        lines.append(f"  Trees used:     {final_metrics.get('n_trees_used', 'N/A')}")
    else:
        lines.append("  (no metrics available)")

    # Top features
    top_features = final_metrics.get("top_features", {})
    if top_features:
        lines.append("")
        lines.append("--- Top 10 Feature Importances ---")
        for i, (feat, imp) in enumerate(top_features.items(), 1):
            lines.append(f"  {i:2d}. {feat:30s}  {imp}")

    # Round-level summary
    if not rounds_df.empty:
        lines.append("")
        lines.append("--- Training Rounds Summary ---")
        lines.append(f"  Total rounds logged:  {len(rounds_df)}")
        lines.append(f"  First round:          {rounds_df.iloc[0]['round']}")
        lines.append(f"  Last round:           {rounds_df.iloc[-1]['round']}")
        lines.append(f"  Best val RMSE:        {rounds_df['val_rmse'].min():.6e} (round {rounds_df.loc[rounds_df['val_rmse'].idxmin(), 'round']})")
        lines.append(f"  Final train RMSE:     {rounds_df.iloc[-1]['train_rmse']:.6e}")
        lines.append(f"  Final val RMSE:       {rounds_df.iloc[-1]['val_rmse']:.6e}")
        # Overfitting gap
        gap = rounds_df.iloc[-1]["val_rmse"] - rounds_df.iloc[-1]["train_rmse"]
        lines.append(f"  Train-Val gap:        {gap:.6e} ({'overfitting' if gap > 0 else 'underfitting'})")

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

def process_tree_output(out_dir: Path):
    """Process a single tree-based output directory."""
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

    # Load metrics.json (if exists)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        final_metrics = json.loads(metrics_path.read_text())
    else:
        final_metrics = parse_final_metrics_from_log(log_text)

    # Parse round-by-round training logs
    rounds_df = pd.DataFrame()
    if model_name in BOOSTING_MODELS:
        rounds_df = parse_round_metrics(log_text, model_name)
        if not rounds_df.empty:
            print(f"  Parsed {len(rounds_df)} training rounds")
        else:
            print(f"  No round-level metrics found in console.log")

    # Plot RMSE over rounds
    plot_round_metrics(rounds_df, model_name, out_dir)

    # Feature importance plot (from metrics.json)
    plot_feature_importance(final_metrics, model_name, out_dir)

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

        # Save per-building metrics CSV
        bld_csv_path = out_dir / "postprocess_per_building_metrics.csv"
        bld_df.to_csv(bld_csv_path, index=False)
        print(f"  Saved: {bld_csv_path}")
    else:
        print(f"  No predictions.parquet — skipping per-building analysis")

    # Text report
    generate_text_report(
        model_name, config, hparams, final_metrics,
        training_time, rounds_df, bld_df, out_dir,
    )


# ── Multi-model comparison ───────────────────────────────────────────

def generate_comparison(out_dirs: list[Path]):
    """Print a comparison table across multiple tree model runs."""
    if len(out_dirs) < 2:
        return

    rows = []
    for d in out_dirs:
        config_path = d / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        model_name = detect_model_type(config)

        metrics_path = d / "metrics.json"
        log_path = d / "console.log"
        log_text = log_path.read_text() if log_path.exists() else ""

        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
        else:
            m = parse_final_metrics_from_log(log_text)

        rows.append({
            "directory": d.name,
            "model": model_name,
            "rmse": m.get("rmse"),
            "mae": m.get("mae"),
            "r2": m.get("r2"),
            "trees": m.get("n_trees_used"),
            "time": parse_training_time(log_text),
        })

    if not rows:
        return

    comp_df = pd.DataFrame(rows).sort_values("r2", ascending=False, na_position="last")
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON (sorted by R²)")
    print(f"{'=' * 80}")
    print(comp_df.to_string(index=False))
    print()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess tree-based model output directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dirs", nargs="+", type=Path,
        help="One or more output directories to process",
    )
    args = parser.parse_args()

    # Filter to valid directories with config.json
    valid_dirs = []
    for d in args.dirs:
        if not d.is_dir():
            print(f"Skipping (not a directory): {d}")
            continue
        if not (d / "config.json").exists():
            print(f"Skipping (no config.json): {d}")
            continue
        # Check it's actually a tree model
        config = json.loads((d / "config.json").read_text())
        model_name = detect_model_type(config)
        if model_name == "Unknown Tree Model":
            # Could be neural — skip
            name = config.get("name", "")
            if any(k in name for k in ("cnn", "lstm", "transformer")):
                print(f"Skipping (neural model): {d}")
                continue
        valid_dirs.append(d)

    if not valid_dirs:
        print("No valid tree-based output directories found.")
        sys.exit(1)

    for d in valid_dirs:
        process_tree_output(d)

    generate_comparison(valid_dirs)


if __name__ == "__main__":
    main()
