"""
Data loading and feature engineering for energy consumption prediction.

Functions:
    load_meter_data       -- read and filter meter CSVs
    aggregate_building_meters -- sum meters per building per timestamp
    load_building_metadata -- read building metadata, derive age
    load_weather_data     -- read weather CSV
    engineer_time_features -- add hour, day-of-week, weekend flag
    build_feature_matrix  -- orchestrate full join pipeline
    split_data            -- temporal or random train/test split
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DataConfig, EnergyModelConfig


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


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from readingtime."""
    df = df.copy()
    df["hour_of_day"] = df["readingtime"].dt.hour
    df["day_of_week"] = df["readingtime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def build_feature_matrix(cfg: EnergyModelConfig) -> pd.DataFrame:
    """
    Orchestrate full data pipeline:
    1. Load and aggregate meter data
    2. Load building metadata
    3. Load weather data
    4. Join meter <-> building on simscode == buildingnumber
    5. Join with weather on readingtime floored to hour
    6. Engineer time features
    7. Normalize energy by gross area
    """
    data_cfg = cfg.data

    # Load and aggregate meter data
    print("Loading meter data...")
    meter_df = load_meter_data(data_cfg)
    print(f"  Raw meter readings: {len(meter_df):,}")
    meter_df = aggregate_building_meters(meter_df)
    print(f"  After aggregation: {len(meter_df):,}")

    # Load building metadata
    print("Loading building metadata...")
    building_df = load_building_metadata(data_cfg)
    print(f"  Buildings with valid area: {len(building_df):,}")

    # Join meter with building metadata
    merged = meter_df.merge(
        building_df[["buildingnumber", "grossarea", "floorsaboveground", "building_age"]],
        left_on="simscode",
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


def split_data(df: pd.DataFrame, cfg: EnergyModelConfig):
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
