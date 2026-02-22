"""
Shared feature engineering for tree-based models.

Canonical implementation of lag, rolling, and interaction features used by all
8 tree models (xgb, lgbm, rf, extratrees, cb, dart, ngb, qrf).
"""

from typing import List, Tuple

import pandas as pd


def engineer_features(
    df: pd.DataFrame,
    *,
    weather_features: List[str],
    building_features: List[str],
    time_features: List[str],
    lag_hours: List[int] = (1, 6, 24, 168),
    rolling_windows: List[int] = (24, 168),
    add_interactions: bool = True,
    hdd_base: float = 65.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add lag, rolling, interaction, and degree-day features per building.

    Groups by simscode, sorts by readingtime, then adds:
    1. Lag features (shift by N 15-min intervals)
    2. Rolling mean/std over configurable windows
    3. Interaction features (temp x area, humidity x area)
    4. HDD/CDD (heating/cooling degree values from temperature)

    Drops rows with NaN from lags/rolling (beginning of each building's data).

    Returns:
        (df_clean, feature_cols) where feature_cols includes all original
        + engineered feature column names.
    """
    df = df.copy()

    # Determine base feature columns (before engineering)
    base_features = weather_features + building_features + time_features
    base_features = [c for c in base_features if c in df.columns]

    engineered_cols: list[str] = []

    # Sort globally for efficiency
    df = df.sort_values(["simscode", "readingtime"]).reset_index(drop=True)

    # 1. Lag features (per building)
    intervals_per_hour = 4  # 15-min data
    for hours in lag_hours:
        n_intervals = hours * intervals_per_hour
        col_name = f"energy_lag_{n_intervals}"
        df[col_name] = df.groupby("simscode")["energy_per_sqft"].shift(n_intervals)
        engineered_cols.append(col_name)

    # 2. Rolling statistics (per building)
    for hours in rolling_windows:
        n_intervals = hours * intervals_per_hour
        grp = df.groupby("simscode")["energy_per_sqft"]

        mean_col = f"rolling_mean_{n_intervals}"
        std_col = f"rolling_std_{n_intervals}"

        df[mean_col] = grp.transform(
            lambda x: x.rolling(n_intervals, min_periods=1).mean()
        )
        df[std_col] = grp.transform(
            lambda x: x.rolling(n_intervals, min_periods=1).std()
        )
        engineered_cols.extend([mean_col, std_col])

    # 3. Interaction features
    if add_interactions:
        if "temperature_2m" in df.columns and "grossarea" in df.columns:
            df["temp_x_area"] = df["temperature_2m"] * df["grossarea"]
            engineered_cols.append("temp_x_area")
        if "relative_humidity_2m" in df.columns and "grossarea" in df.columns:
            df["humidity_x_area"] = df["relative_humidity_2m"] * df["grossarea"]
            engineered_cols.append("humidity_x_area")

    # 4. Heating / Cooling Degree values (base 65 °F)
    if "temperature_2m" in df.columns:
        df["hdd"] = (hdd_base - df["temperature_2m"]).clip(lower=0)
        df["cdd"] = (df["temperature_2m"] - hdd_base).clip(lower=0)
        engineered_cols.extend(["hdd", "cdd"])

    # Drop rows with NaN from lag/rolling features
    all_feature_cols = base_features + engineered_cols
    df = df.dropna(subset=all_feature_cols).reset_index(drop=True)

    return df, all_feature_cols
