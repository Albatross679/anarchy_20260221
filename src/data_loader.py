"""
Data loading and feature engineering for energy consumption prediction.

Functions:
    load_meter_data       -- read and filter meter CSVs
    load_meter_data_raw   -- read all CSVs without filtering (for cleaner)
    aggregate_building_meters -- sum meters per building per timestamp
    load_building_metadata -- read building metadata, derive age
    load_weather_data     -- read weather CSV
    engineer_time_features -- add hour, day-of-week, weekend flag
    build_feature_matrix  -- orchestrate full join pipeline
    build_multi_utility_matrix -- clean once, build per-utility matrices
    split_data            -- temporal or random train/test split
"""

import json

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import DataConfig, MLBaseConfig


def load_meter_data(cfg: DataConfig) -> pd.DataFrame:
    """Read meter CSVs, filter to target utility, clean values."""
    frames = []
    for f in cfg.meter_files:
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Filter to target utility
    df = df[df["utility"] == cfg.utility_filter].copy()

    # Parse datetime
    df["readingtime"] = pd.to_datetime(df["readingtime"])

    # Clean: drop NaN and negative readings
    df = df.dropna(subset=["readingvalue", "simscode"])
    df = df[df["readingvalue"] >= 0]

    # Ensure simscode is clean string for joining (float->int->str to avoid "338.0")
    df["simscode"] = df["simscode"].astype(int).astype(str)

    return df


def load_meter_data_raw(cfg: DataConfig) -> pd.DataFrame:
    """Read all meter CSVs without utility filtering or NaN dropping.

    Used by clean_meter_data which needs the full raw DataFrame across
    all utilities.
    """
    frames = []
    for f in cfg.meter_files:
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Parse datetime
    df["readingtime"] = pd.to_datetime(df["readingtime"])

    # Minimal cleaning: drop rows with no simscode at all
    df = df.dropna(subset=["simscode"])

    # Ensure simscode is clean string
    df["simscode"] = df["simscode"].astype(int).astype(str)

    return df


def aggregate_building_meters(df: pd.DataFrame) -> pd.DataFrame:
    """Sum readings across meters for each (building, timestamp)."""
    agg = (
        df.groupby(["simscode", "readingtime"], as_index=False)
        .agg(readingvalue=("readingvalue", "sum"))
    )
    return agg


def load_building_metadata(cfg: DataConfig) -> pd.DataFrame:
    """Read building metadata, derive building_age, clean."""
    df = pd.read_csv(cfg.building_metadata_file)

    # Parse construction date and derive age
    df["constructiondate"] = pd.to_datetime(df["constructiondate"], errors="coerce")
    df["building_age"] = 2025 - df["constructiondate"].dt.year

    # Fill missing age with median
    median_age = df["building_age"].median()
    df["building_age"] = df["building_age"].fillna(median_age)

    # Drop buildings with no usable area
    df = df[df["grossarea"].fillna(0).astype(float) > 0].copy()

    # Ensure join key is string
    df["buildingnumber"] = df["buildingnumber"].astype(str).str.strip()

    # Convert numeric columns
    for col in ["grossarea", "floorsaboveground", "floorsbelowground"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_weather_data(cfg: DataConfig) -> pd.DataFrame:
    """Read weather CSV, parse datetime, keep feature columns."""
    df = pd.read_csv(cfg.weather_file)
    df["date"] = pd.to_datetime(df["date"])

    # Keep only the feature columns + date
    keep_cols = ["date"] + [c for c in cfg.weather_features if c in df.columns]
    df = df[keep_cols].copy()

    # Convert to numeric
    for col in cfg.weather_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Upsample hourly meter readings to 15-minute intervals.

    Each hourly reading is split into 4 equal 15-min readings at
    :00, :15, :30, :45.  readingvalue is divided by 4.
    """
    offsets = pd.to_timedelta([0, 15, 30, 45], unit="min")
    repeated = pd.concat([df] * 4, ignore_index=True)
    n = len(df)
    offset_col = np.concatenate([np.full(n, off.total_seconds()) for off in offsets])
    repeated["readingtime"] = (
        repeated["readingtime"] + pd.to_timedelta(offset_col, unit="s")
    )
    repeated["readingvalue"] = repeated["readingvalue"] / 4.0
    repeated = repeated.sort_values(["simscode", "readingtime"]).reset_index(drop=True)
    return repeated


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from readingtime."""
    df = df.copy()
    df["hour_of_day"] = df["readingtime"].dt.hour
    df["minute_of_hour"] = df["readingtime"].dt.minute
    df["day_of_week"] = df["readingtime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def build_feature_matrix(
    cfg: MLBaseConfig,
    run_cleaning: bool = False,
) -> pd.DataFrame:
    """
    Orchestrate full data pipeline:
    1. Load and aggregate meter data
    2. Load building metadata
    3. Load weather data
    4. Join meter <-> building on simscode == buildingnumber
    5. Join with weather on readingtime floored to hour
    6. Engineer time features
    7. Normalize energy by gross area

    Args:
        cfg: Full experiment config.
        run_cleaning: If True, runs data_cleaner.clean_meter_data before
            the standard pipeline. Default False preserves backward compat.
    """
    data_cfg = cfg.data

    # Load building metadata (needed for both cleaning and joining)
    print("Loading building metadata...")
    building_df = load_building_metadata(data_cfg)
    print(f"  Buildings with valid area: {len(building_df):,}")

    # Decide data source: pre-cleaned CSV > run_cleaning > raw
    use_cleaned = getattr(data_cfg, "use_cleaned_data", False)
    cleaned_dir = getattr(data_cfg, "cleaned_data_dir", "data")

    if use_cleaned:
        cleaned_path = Path(cleaned_dir) / f"cleaned_{data_cfg.utility_filter.lower()}.csv"
        if cleaned_path.exists():
            print(f"Loading pre-cleaned data from {cleaned_path}...")
            meter_df = pd.read_csv(cleaned_path, low_memory=False)
            meter_df["readingtime"] = pd.to_datetime(meter_df["readingtime"])
            meter_df = meter_df.dropna(subset=["readingvalue", "simscode"])
            meter_df = meter_df[meter_df["readingvalue"] >= 0]
            meter_df["simscode"] = meter_df["simscode"].astype(int).astype(str)
            print(f"  Cleaned meter readings: {len(meter_df):,}")
        else:
            print(f"  WARNING: cleaned file {cleaned_path} not found, falling back to raw data")
            use_cleaned = False

    if not use_cleaned and run_cleaning:
        from src.data_cleaner import clean_meter_data

        print("Loading raw meter data for cleaning...")
        raw_df = load_meter_data_raw(data_cfg)
        print(f"  Raw meter readings (all utilities): {len(raw_df):,}")

        cleaner_cfg = data_cfg.cleaner
        cleaned_df, report = clean_meter_data(
            raw_df,
            building_df,
            lookup_path=cleaner_cfg.lookup_path,
            fuzzy_threshold=cleaner_cfg.fuzzy_threshold,
            gap_limit=cleaner_cfg.gap_limit_intervals,
        )

        # Filter to target utility after cleaning
        meter_df = cleaned_df[cleaned_df["utility"] == data_cfg.utility_filter].copy()
        meter_df = meter_df.dropna(subset=["readingvalue"])
        meter_df = meter_df[meter_df["readingvalue"] >= 0]
        print(f"  After cleaning + utility filter: {len(meter_df):,}")

    if not use_cleaned and not run_cleaning:
        # Original path
        print("Loading meter data...")
        meter_df = load_meter_data(data_cfg)
        print(f"  Raw meter readings: {len(meter_df):,}")

    meter_df = aggregate_building_meters(meter_df)
    print(f"  After aggregation: {len(meter_df):,}")

    # Upsample hourly readings to 15-minute intervals
    meter_df = resample_to_15min(meter_df)
    print(f"  After 15-min resample: {len(meter_df):,}")

    # Determine join key: use resolved_buildingnumber if available
    if "resolved_buildingnumber" in meter_df.columns:
        join_key = "resolved_buildingnumber"
        # Drop rows with no resolved building for the join
        meter_df = meter_df.dropna(subset=[join_key])
        print(f"  Using resolved building linkage ({join_key})")
    else:
        join_key = "simscode"

    # Join meter with building metadata
    merged = meter_df.merge(
        building_df[["buildingnumber", "grossarea", "floorsaboveground", "building_age"]],
        left_on=join_key,
        right_on="buildingnumber",
        how="inner",
    )
    print(f"  After building join: {len(merged):,}")
    n_buildings = merged["simscode"].nunique()
    print(f"  Unique buildings: {n_buildings}")

    # Load weather and join on floored hour
    print("Loading weather data...")
    weather_df = load_weather_data(data_cfg)
    merged["weather_hour"] = merged["readingtime"].dt.floor("h")
    merged = merged.merge(
        weather_df, left_on="weather_hour", right_on="date", how="inner"
    )
    merged.drop(columns=["weather_hour", "date"], inplace=True)
    print(f"  After weather join: {len(merged):,}")

    # Engineer time features
    merged = engineer_time_features(merged)

    # Normalize energy by gross area (kWh per sqft)
    merged["energy_per_sqft"] = merged["readingvalue"] / merged["grossarea"]

    # Drop rows with NaN in feature columns
    feature_cols = data_cfg.weather_features + data_cfg.building_features + data_cfg.time_features
    available_features = [c for c in feature_cols if c in merged.columns]
    merged = merged.dropna(subset=available_features + ["energy_per_sqft"])

    # Remove extreme outliers (IQR-based on energy_per_sqft)
    q1 = merged["energy_per_sqft"].quantile(0.01)
    q99 = merged["energy_per_sqft"].quantile(0.99)
    before = len(merged)
    merged = merged[(merged["energy_per_sqft"] >= q1) & (merged["energy_per_sqft"] <= q99)]
    print(f"  Outlier removal: {before - len(merged):,} rows removed (1st-99th percentile)")

    print(f"  Final dataset: {len(merged):,} rows, {merged['simscode'].nunique()} buildings")

    return merged


def build_multi_utility_matrix(
    cfg: MLBaseConfig,
    utilities: Optional[List[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], "CleaningReport"]:
    """Clean data once, then build per-utility feature matrices.

    Args:
        cfg: Full experiment config.
        utilities: List of utility types to process. Defaults to the five
            main energy utilities.

    Returns:
        (dict of utility -> DataFrame, CleaningReport)
    """
    from src.data_cleaner import clean_meter_data

    if utilities is None:
        utilities = ["ELECTRICITY", "GAS", "HEAT", "STEAM", "COOLING"]

    data_cfg = cfg.data

    # Load raw data and building metadata
    print("Loading raw meter data for multi-utility cleaning...")
    raw_df = load_meter_data_raw(data_cfg)
    building_df = load_building_metadata(data_cfg)

    # Clean once across all utilities
    cleaner_cfg = data_cfg.cleaner
    cleaned_df, report = clean_meter_data(
        raw_df,
        building_df,
        lookup_path=cleaner_cfg.lookup_path,
        fuzzy_threshold=cleaner_cfg.fuzzy_threshold,
        gap_limit=cleaner_cfg.gap_limit_intervals,
    )

    # Build per-utility feature matrices
    results = {}
    weather_df = load_weather_data(data_cfg)

    for utility in utilities:
        print(f"\nBuilding matrix for {utility}...")
        udf = cleaned_df[cleaned_df["utility"] == utility].copy()
        udf = udf.dropna(subset=["readingvalue"])
        udf = udf[udf["readingvalue"] >= 0]

        if len(udf) == 0:
            print(f"  {utility}: no data after cleaning, skipping")
            continue

        udf = aggregate_building_meters(udf)
        udf = resample_to_15min(udf)

        # Determine join key
        if "resolved_buildingnumber" in udf.columns:
            join_key = "resolved_buildingnumber"
            udf = udf.dropna(subset=[join_key])
        else:
            join_key = "simscode"

        merged = udf.merge(
            building_df[["buildingnumber", "grossarea", "floorsaboveground", "building_age"]],
            left_on=join_key,
            right_on="buildingnumber",
            how="inner",
        )

        if len(merged) == 0:
            print(f"  {utility}: no matched buildings, skipping")
            continue

        # Weather join
        merged["weather_hour"] = merged["readingtime"].dt.floor("h")
        merged = merged.merge(
            weather_df, left_on="weather_hour", right_on="date", how="inner"
        )
        merged.drop(columns=["weather_hour", "date"], inplace=True)

        merged = engineer_time_features(merged)
        merged["energy_per_sqft"] = merged["readingvalue"] / merged["grossarea"]

        feature_cols = data_cfg.weather_features + data_cfg.building_features + data_cfg.time_features
        available_features = [c for c in feature_cols if c in merged.columns]
        merged = merged.dropna(subset=available_features + ["energy_per_sqft"])

        # Percentile-based outlier removal per utility
        if len(merged) > 100:
            q1 = merged["energy_per_sqft"].quantile(0.01)
            q99 = merged["energy_per_sqft"].quantile(0.99)
            merged = merged[(merged["energy_per_sqft"] >= q1) & (merged["energy_per_sqft"] <= q99)]

        print(f"  {utility}: {len(merged):,} rows, {merged['simscode'].nunique()} buildings")
        results[utility] = merged

    return results, report


def split_data(df: pd.DataFrame, cfg: MLBaseConfig):
    """
    Split into train/test.
    Temporal split (default): train on data before split_date, test on data after.
    Random split: random 80/20.

    Returns (X_train, X_test, y_train, y_test, feature_cols)
    """
    data_cfg = cfg.data
    feature_cols = data_cfg.weather_features + data_cfg.building_features + data_cfg.time_features
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col = "energy_per_sqft"

    if data_cfg.temporal_split:
        split_dt = pd.Timestamp(data_cfg.split_date)
        train_mask = df["readingtime"] < split_dt
        test_mask = df["readingtime"] >= split_dt
        train_df = df[train_mask]
        test_df = df[test_mask]
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=1 - data_cfg.random_split_ratio, random_state=cfg.seed
        )

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    return X_train, X_test, y_train, y_test, feature_cols


def load_precomputed_tree_features(
    utility: str,
    features_dir: str = "data",
    parquet_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load pre-computed tree features from parquet + manifest.

    Args:
        utility: Utility type (e.g. "ELECTRICITY").
        features_dir: Directory containing parquet and manifest files.
        parquet_path: Optional explicit path to a parquet file. When provided,
            the manifest is searched for an entry whose parquet_file matches,
            or for a ``{UTILITY}_CROSS`` key, falling back to the standard
            utility key.

    Returns:
        (df, feature_cols) ready for train/test split.
    """
    features_dir = Path(features_dir)

    if parquet_path is not None:
        parquet_path = Path(parquet_path)
    else:
        parquet_path = features_dir / f"tree_features_{utility.lower()}.parquet"

    manifest_path = features_dir / "tree_features_manifest.json"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Pre-computed features not found: {parquet_path}\n"
            f"Run: python src/prepare_tree_features.py --utilities {utility}"
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run: python src/prepare_tree_features.py --utilities {utility}"
        )

    df = pd.read_parquet(parquet_path)
    manifest = json.loads(manifest_path.read_text())

    # Find the manifest entry: try matching parquet_file first, then
    # {UTILITY}_CROSS, then {UTILITY}.
    utility_key = None
    for key, entry in manifest.get("utilities", {}).items():
        if str(parquet_path) == entry.get("parquet_file"):
            utility_key = key
            break
    if utility_key is None:
        for candidate in [f"{utility.upper()}_CROSS", utility.upper()]:
            if candidate in manifest.get("utilities", {}):
                utility_key = candidate
                break

    if utility_key is None:
        raise KeyError(
            f"No manifest entry found for {parquet_path} or utility {utility.upper()}. "
            f"Available: {list(manifest['utilities'].keys())}"
        )

    feature_cols = manifest["utilities"][utility_key]["feature_cols"]

    print(f"Loaded pre-computed features from {parquet_path}")
    print(f"  Manifest key: {utility_key}")
    print(f"  Rows: {len(df):,} | Features: {len(feature_cols)}")
    print(f"  Buildings: {df['simscode'].nunique()}")

    return df, feature_cols
