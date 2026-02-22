#!/usr/bin/env python3
"""SHAP-based explainability for energy investment prioritization.

Generates:
    - Global feature importance (bar chart)
    - Per-building SHAP waterfall charts
    - Plain-language explanations ("Building X wastes energy because...")

Usage:
    python -m src.explainer                              # electricity, top 10
    python -m src.explainer --utility COOLING --top-n 20
    python -m src.explainer --building 42                # single building
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ── Configuration ────────────────────────────────────────────────────

UTILITY_DIRS = {
    "ELECTRICITY": "output/electricity_xgboost_20260222_021830",
    "COOLING": "output/cooling_xgboost_20260222_021836",
    "GAS": "output/gas_xgboost_20260222_021833",
    "HEAT": "output/heat_xgboost_20260222_021837",
    "STEAM": "output/steam_xgboost_20260222_021840",
}

BUILDING_METADATA = "data/building_metadata.csv"
OUT_DIR = Path("doc/explainability")

# Human-readable feature names
FEATURE_LABELS = {
    "energy_lag_4": "Energy (1h ago)",
    "energy_lag_24": "Energy (6h ago)",
    "energy_lag_96": "Energy (1 day ago)",
    "energy_lag_672": "Energy (1 week ago)",
    "rolling_mean_96": "24h Avg Energy",
    "rolling_std_96": "24h Energy Volatility",
    "rolling_mean_672": "7-day Avg Energy",
    "rolling_std_672": "7-day Energy Volatility",
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity",
    "dew_point_2m": "Dew Point",
    "direct_radiation": "Solar Radiation",
    "wind_speed_10m": "Wind Speed",
    "cloud_cover": "Cloud Cover",
    "apparent_temperature": "Feels-like Temp",
    "precipitation": "Precipitation",
    "grossarea": "Building Size",
    "floorsaboveground": "# Floors",
    "building_age": "Building Age",
    "hour_of_day": "Hour of Day",
    "minute_of_hour": "Minute",
    "day_of_week": "Day of Week",
    "is_weekend": "Weekend",
    "temp_x_area": "Temp × Area",
    "humidity_x_area": "Humidity × Area",
    "hdd": "Heating Degree-Days",
    "cdd": "Cooling Degree-Days",
}


# ── Model + data loading ────────────────────────────────────────────


def load_model_and_data(
    utility: str, utility_dirs: dict | None = None
) -> tuple:
    """Load XGBoost model and predictions for a utility.

    Returns:
        model: XGBoost Booster
        preds: predictions DataFrame
        feature_cols: list of feature column names
        config: training config dict
    """
    utility_dirs = utility_dirs or UTILITY_DIRS
    dir_path = Path(utility_dirs[utility])

    # Load config to get feature columns
    with open(dir_path / "config.json") as f:
        config = json.load(f)

    # Load predictions
    preds = pd.read_parquet(dir_path / "predictions.parquet")

    # Load model — check multiple possible filenames
    model_path = None
    for candidate in [
        dir_path / "checkpoints" / "model_best.json",
        dir_path / "checkpoints" / "model.ubj",
        dir_path / "checkpoints" / "model.json",
        dir_path / "model_best.json",
    ]:
        if candidate.exists():
            model_path = candidate
            break
    if model_path is None:
        raise FileNotFoundError(f"No model found in {dir_path}")

    model = xgb.Booster()
    model.load_model(str(model_path))

    # Determine feature columns (everything that's not target/meta)
    meta_cols = {
        "simscode", "buildingnumber", "readingtime", "readingvalue",
        "energy_per_sqft", "predicted", "residual",
        "predicted_q10", "predicted_q50", "predicted_q90",
        "prediction_interval_width", "predicted_std",
    }
    feature_cols = [c for c in preds.columns if c not in meta_cols]

    return model, preds, feature_cols, config


# ── SHAP computation ────────────────────────────────────────────────


def compute_shap_values(
    model, X: pd.DataFrame, max_samples: int = 5000
) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer.

    Subsamples to max_samples for speed.
    """
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer(X_sample)

    return shap_values


def compute_building_shap(
    model, preds: pd.DataFrame, feature_cols: list, building_id: str,
    max_samples: int = 500,
) -> shap.Explanation:
    """Compute SHAP values for a single building's predictions."""
    bldg_data = preds[preds["buildingnumber"].astype(str) == str(building_id)]
    if bldg_data.empty:
        raise ValueError(f"Building {building_id} not found in predictions")

    X = bldg_data[feature_cols]
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)

    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer(X)

    return shap_values


# ── Plain-language explanation ───────────────────────────────────────


def explain_building(
    shap_values: shap.Explanation, building_id: str, utility: str
) -> str:
    """Generate a plain-language explanation for why a building ranks high.

    Example: "Building 42 wastes energy because: high baseline load
    (SHAP +0.3), poor weather response (SHAP +0.2)"
    """
    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = shap_values.feature_names

    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    top_features = []
    for i in sorted_idx[:5]:
        fname = feature_names[i]
        label = FEATURE_LABELS.get(fname, fname)
        mean_shap = shap_values.values[:, i].mean()
        direction = "increases" if mean_shap > 0 else "decreases"
        top_features.append(
            f"{label} {direction} predicted consumption (SHAP {mean_shap:+.6f})"
        )

    explanation = (
        f"Building {building_id} ({utility}):\n"
        f"  Key energy drivers:\n"
    )
    for i, feat in enumerate(top_features, 1):
        explanation += f"    {i}. {feat}\n"

    # Overall assessment
    mean_shap_sum = shap_values.values.mean(axis=0).sum()
    if mean_shap_sum > 0:
        explanation += (
            f"  Assessment: Model predicts HIGHER consumption than baseline "
            f"(net SHAP effect: {mean_shap_sum:+.6f})\n"
        )
    else:
        explanation += (
            f"  Assessment: Model predicts LOWER consumption than baseline "
            f"(net SHAP effect: {mean_shap_sum:+.6f})\n"
        )

    return explanation


# ── Visualization ────────────────────────────────────────────────────


def plot_global_importance(
    shap_values: shap.Explanation, utility: str, out_dir: Path
):
    """Bar chart of mean |SHAP| values (global feature importance)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=15, show=False, ax=ax)
    ax.set_title(f"Global Feature Importance — {utility}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / f"global_importance_{utility.lower()}.png", dpi=150)
    plt.close()
    print(f"  Saved global importance plot for {utility}")


def plot_building_waterfall(
    shap_values: shap.Explanation, building_id: str, utility: str, out_dir: Path
):
    """SHAP waterfall chart for a single building (mean across its time steps)."""
    # Average SHAP values across time steps for this building
    mean_shap = shap_values.values.mean(axis=0)
    base_value = shap_values.base_values.mean() if hasattr(shap_values.base_values, 'mean') else shap_values.base_values[0]

    # Create a single-row Explanation for waterfall
    avg_explanation = shap.Explanation(
        values=mean_shap,
        base_values=base_value,
        feature_names=shap_values.feature_names,
        data=shap_values.data.mean(axis=0) if hasattr(shap_values.data, 'mean') else None,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(avg_explanation, max_display=12, show=False)
    plt.title(f"Building {building_id} — {utility}\n(averaged across time steps)", fontsize=11)
    plt.tight_layout()
    plt.savefig(
        out_dir / f"waterfall_{building_id}_{utility.lower()}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved waterfall for building {building_id} ({utility})")


def plot_beeswarm(shap_values: shap.Explanation, utility: str, out_dir: Path):
    """Beeswarm summary plot showing SHAP value distribution."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"SHAP Value Distribution — {utility}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / f"beeswarm_{utility.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved beeswarm plot for {utility}")


# ── Main pipeline ───────────────────────────────────────────────────


def run_explainability(
    utility: str = "ELECTRICITY",
    top_n: int = 10,
    building_ids: list[str] | None = None,
    out_dir: Path = OUT_DIR,
    max_global_samples: int = 5000,
    max_building_samples: int = 500,
):
    """Full explainability pipeline for one utility."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExplainability Analysis — {utility}")
    print("=" * 50)

    # Load model + data
    print("Loading model and predictions...")
    model, preds, feature_cols, config = load_model_and_data(utility)
    preds["buildingnumber"] = preds["buildingnumber"].astype(str)

    X_all = preds[feature_cols]
    print(f"  {len(preds)} predictions, {len(feature_cols)} features, "
          f"{preds['buildingnumber'].nunique()} buildings")

    # Global SHAP
    print("Computing global SHAP values...")
    global_shap = compute_shap_values(model, X_all, max_global_samples)
    plot_global_importance(global_shap, utility, out_dir)
    plot_beeswarm(global_shap, utility, out_dir)

    # Determine which buildings to explain
    if building_ids is None:
        # Use top buildings by |mean residual|
        bldg_residuals = (
            preds.groupby("buildingnumber")["residual"]
            .mean().abs()
            .sort_values(ascending=False)
        )
        building_ids = bldg_residuals.head(top_n).index.tolist()
    building_ids = [str(b) for b in building_ids]

    # Per-building SHAP
    print(f"\nGenerating per-building explanations ({len(building_ids)} buildings)...")
    explanations = []
    for bldg_id in building_ids:
        try:
            bldg_shap = compute_building_shap(
                model, preds, feature_cols, bldg_id, max_building_samples
            )
            plot_building_waterfall(bldg_shap, bldg_id, utility, out_dir)
            explanation = explain_building(bldg_shap, bldg_id, utility)
            explanations.append(explanation)
            print(f"    {explanation.strip()}")
        except Exception as e:
            print(f"    [ERROR] Building {bldg_id}: {e}")

    # Save explanations
    explanation_path = out_dir / f"explanations_{utility.lower()}.txt"
    explanation_path.write_text("\n\n".join(explanations))
    print(f"\n  Saved explanations to {explanation_path}")

    # Save SHAP summary as JSON
    mean_abs_shap = np.abs(global_shap.values).mean(axis=0)
    feature_importance = dict(zip(global_shap.feature_names, mean_abs_shap.tolist()))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: -x[1]))

    summary = {
        "utility": utility,
        "n_buildings": int(preds["buildingnumber"].nunique()),
        "n_features": len(feature_cols),
        "n_samples_for_shap": min(len(X_all), max_global_samples),
        "feature_importance_shap": feature_importance,
        "top_buildings_explained": building_ids,
    }
    summary_path = out_dir / f"shap_summary_{utility.lower()}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved SHAP summary to {summary_path}")

    return global_shap, explanations


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    if shap is None:
        print("ERROR: shap package not installed. Run: pip install shap")
        return
    if xgb is None:
        print("ERROR: xgboost package not installed. Run: pip install xgboost")
        return

    parser = argparse.ArgumentParser(description="SHAP explainability for energy models")
    parser.add_argument("--utility", default="ELECTRICITY", choices=list(UTILITY_DIRS.keys()))
    parser.add_argument("--top-n", type=int, default=10, help="Number of buildings to explain")
    parser.add_argument("--building", type=str, nargs="*", help="Specific building IDs")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples for global SHAP")
    args = parser.parse_args()

    run_explainability(
        utility=args.utility,
        top_n=args.top_n,
        building_ids=args.building,
        out_dir=Path(args.out_dir),
        max_global_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
