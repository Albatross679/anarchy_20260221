#!/usr/bin/env python3
"""
Pre-compute gas-specific features with cross-utility signals.

Builds the standard gas feature matrix, then enriches it with concurrent
readings from co-located utilities (ELECTRICITY, HEAT, STEAM, COOLING).
Buildings without a given co-utility get NaN (XGBoost handles missing natively).

Cross-utility features per co-utility:
    {utility}_concurrent     — energy_per_sqft at same (building, timestamp)
    {utility}_lag_4          — 1-hour lag  (4 × 15-min intervals)
    {utility}_lag_96         — 24-hour lag (96 × 15-min intervals)
    {utility}_rolling_mean_96 — 24-hour rolling mean

Usage:
    python src/prepare_gas_features.py
    python src/prepare_gas_features.py --out-dir data
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import DataConfig, MLBaseConfig
from src.data_loader import (
    build_feature_matrix,
    load_building_metadata,
    load_meter_data,
    aggregate_building_meters,
    resample_to_15min,
)
from src.feature_engineer import engineer_features

CO_UTILITIES = ["ELECTRICITY", "HEAT", "STEAM", "COOLING"]

# Lag/rolling parameters for cross-utility signals
CROSS_LAG_INTERVALS = [4, 96]       # 1h, 24h  (in 15-min intervals)
CROSS_ROLLING_WINDOWS = [96]        # 24h

# Standard tree feature engineering parameters
DEFAULT_LAG_HOURS = [1, 6, 24, 168]
DEFAULT_ROLLING_WINDOWS = [24, 168]
DEFAULT_ADD_INTERACTIONS = True

METADATA_COLS = [
    "simscode",
    "readingtime",
    "energy_per_sqft",
    "readingvalue",
    "grossarea",
    "buildingnumber",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute gas features with cross-utility signals"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory for parquet files (default: data)",
    )
    return parser.parse_args()


def load_co_utility_signals(utility: str, data_cfg: DataConfig) -> pd.DataFrame:
    """Load a co-utility's meter data and compute energy_per_sqft per (building, timestamp).

    Returns a DataFrame with columns: simscode, readingtime, {utility}_concurrent
    """
    print(f"  Loading {utility} meter data...")
    co_cfg = DataConfig(utility_filter=utility)

    try:
        meter_df = load_meter_data(co_cfg)
    except Exception as e:
        print(f"    SKIP {utility}: {e}")
        return pd.DataFrame()

    if len(meter_df) == 0:
        print(f"    SKIP {utility}: no data")
        return pd.DataFrame()

    # Aggregate meters per building per timestamp
    meter_df = aggregate_building_meters(meter_df)
    # Upsample to 15-min intervals
    meter_df = resample_to_15min(meter_df)

    # Join with building metadata to get grossarea
    building_df = load_building_metadata(data_cfg)
    meter_df = meter_df.merge(
        building_df[["buildingnumber", "grossarea"]],
        left_on="simscode",
        right_on="buildingnumber",
        how="inner",
    )

    # Compute energy per sqft for the co-utility
    col_name = f"{utility.lower()}_concurrent"
    meter_df[col_name] = meter_df["readingvalue"] / meter_df["grossarea"]

    # Keep only what we need for the join
    result = meter_df[["simscode", "readingtime", col_name]].copy()

    print(f"    {utility}: {len(result):,} rows, {result['simscode'].nunique()} buildings")
    return result


def add_cross_utility_lags(df: pd.DataFrame, col_name: str) -> list[str]:
    """Add lag and rolling features for a cross-utility concurrent column.

    Returns list of new column names created.
    """
    new_cols = []
    base = col_name  # e.g. "electricity_concurrent"

    df.sort_values(["simscode", "readingtime"], inplace=True)

    for lag in CROSS_LAG_INTERVALS:
        lag_col = f"{base.replace('_concurrent', '')}_lag_{lag}"
        df[lag_col] = df.groupby("simscode")[col_name].shift(lag)
        new_cols.append(lag_col)

    for window in CROSS_ROLLING_WINDOWS:
        roll_col = f"{base.replace('_concurrent', '')}_rolling_mean_{window}"
        df[roll_col] = df.groupby("simscode")[col_name].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        new_cols.append(roll_col)

    return new_cols


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig()
    t0 = time.time()

    # 1. Build gas feature matrix
    print("=" * 60)
    print("Building GAS feature matrix")
    print("=" * 60)

    cfg = MLBaseConfig()
    cfg.data = DataConfig(utility_filter="GAS")
    df = build_feature_matrix(cfg, run_cleaning=True)

    if len(df) == 0:
        print("ERROR: GAS feature matrix is empty after pipeline")
        sys.exit(1)

    gas_buildings = set(df["simscode"].unique())
    print(f"\nGas buildings: {len(gas_buildings)}")

    # 2. Load and join cross-utility signals
    print(f"\n{'=' * 60}")
    print("Adding cross-utility features")
    print(f"{'=' * 60}")

    cross_feature_cols = []
    coverage_stats = {}

    for utility in CO_UTILITIES:
        co_df = load_co_utility_signals(utility, data_cfg)

        if len(co_df) == 0:
            coverage_stats[utility] = {"buildings": 0, "pct": 0.0}
            continue

        col_name = f"{utility.lower()}_concurrent"

        # Left-join onto gas data
        before_len = len(df)
        df = df.merge(
            co_df,
            on=["simscode", "readingtime"],
            how="left",
        )
        cross_feature_cols.append(col_name)

        # Compute coverage
        co_buildings = set(co_df["simscode"].unique())
        overlap = gas_buildings & co_buildings
        coverage_stats[utility] = {
            "buildings": len(overlap),
            "pct": 100.0 * len(overlap) / len(gas_buildings) if gas_buildings else 0,
        }

        # Add lags and rolling features for this co-utility
        new_cols = add_cross_utility_lags(df, col_name)
        cross_feature_cols.extend(new_cols)

        non_null_pct = 100.0 * df[col_name].notna().mean()
        print(f"    Joined: {len(df):,} rows, {non_null_pct:.1f}% non-null for {col_name}")

    # 3. Run standard gas-specific feature engineering
    print(f"\n{'=' * 60}")
    print("Engineering standard features")
    print(f"{'=' * 60}")

    df, std_feature_cols = engineer_features(
        df,
        weather_features=data_cfg.weather_features,
        building_features=data_cfg.building_features,
        time_features=data_cfg.time_features,
        lag_hours=DEFAULT_LAG_HOURS,
        rolling_windows=DEFAULT_ROLLING_WINDOWS,
        add_interactions=DEFAULT_ADD_INTERACTIONS,
    )

    # Combine standard + cross-utility feature columns
    # Cross-utility cols that survived the dropna in engineer_features
    surviving_cross = [c for c in cross_feature_cols if c in df.columns]
    all_feature_cols = std_feature_cols + surviving_cross

    # Deduplicate while preserving order
    seen = set()
    all_feature_cols = [c for c in all_feature_cols if not (c in seen or seen.add(c))]

    print(f"  After engineering: {len(df):,} rows, {len(all_feature_cols)} features")
    print(f"  Standard features: {len(std_feature_cols)}")
    print(f"  Cross-utility features: {len(surviving_cross)}")

    # 4. Save parquet
    keep_cols = [c for c in METADATA_COLS if c in df.columns] + all_feature_cols
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

    df_out = df[keep_cols]

    parquet_path = out_dir / "tree_features_gas_cross.parquet"
    df_out.to_parquet(parquet_path, index=False)
    print(f"\nSaved: {parquet_path} ({len(df_out):,} rows, {len(keep_cols)} cols)")

    # 5. Update manifest
    manifest_path = out_dir / "tree_features_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"description": "Pre-computed tree-model features", "utilities": {}}

    manifest["utilities"]["GAS_CROSS"] = {
        "parquet_file": str(parquet_path),
        "rows": len(df_out),
        "buildings": int(df_out["simscode"].nunique()),
        "feature_cols": all_feature_cols,
        "all_cols": keep_cols,
        "cross_utility_features": surviving_cross,
        "cross_utility_coverage": coverage_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest updated: {manifest_path} (key: GAS_CROSS)")

    # 6. Print coverage summary
    print(f"\n{'=' * 60}")
    print("Cross-utility coverage for GAS buildings")
    print(f"{'=' * 60}")
    for utility, stats in coverage_stats.items():
        print(f"  {utility:15s}: {stats['buildings']:3d} buildings ({stats['pct']:.1f}%)")

    print(f"\nCross-utility feature columns ({len(surviving_cross)}):")
    for col in surviving_cross:
        non_null = df_out[col].notna().sum()
        pct = 100.0 * non_null / len(df_out)
        print(f"  {col:40s}: {pct:5.1f}% non-null")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"\nTo train with cross-utility features:")
    print(f"  python xgb/train.py --precomputed --utility GAS \\")
    print(f"    --precomputed-path {parquet_path}")


if __name__ == "__main__":
    main()
