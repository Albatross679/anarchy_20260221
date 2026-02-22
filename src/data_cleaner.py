"""
Data cleaning pipeline for smart meter data.

Cleaning steps (executed in order):
    1. drop_nan_simscode      -- remove rows with NaN join keys
    2. exclude_utilities       -- drop OIL28SEC (100% zeros)
    3. exclude_unmatched_buildings -- drop simscodes 8, 43, 93 (no metadata)
    4. apply_hard_caps         -- drop sensor-fault outliers per utility
    5. impute_short_gaps       -- ffill/bfill gaps up to 8 hours
    6. drop_dead_meters        -- drop meters that are 100% NaN
    7. drop_sparse_meters      -- drop meters with >50% NaN (long outages)

Orchestrator:
    clean_meter_data          -- runs all steps, returns (cleaned_df, report)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UTILITY_HARD_CAPS: Dict[str, float] = {
    "ELECTRICITY": 10_000,
    "ELECTRICAL_POWER": 10_000,
    "GAS": 50_000,
    "STEAM": 1_000_000,
    "STEAMRATE": 1_000_000,
    "HEAT": 10_000,
    "COOLING": 10_000,
    "COOLING_POWER": 10_000,
}

EXCLUDED_UTILITIES = {"OIL28SEC"}
EXCLUDED_SIMSCODES = {"8", "43", "93"}
FFILL_GAP_LIMIT = 8  # 8 hours at hourly intervals (applied before 15-min resample)


# ---------------------------------------------------------------------------
# CleaningReport
# ---------------------------------------------------------------------------


@dataclass
class CleaningReport:
    rows_before: int = 0
    rows_after: int = 0
    nan_simscode_dropped: int = 0
    excluded_utility_dropped: int = 0
    excluded_buildings_dropped: int = 0
    outliers_removed: Dict[str, int] = field(default_factory=dict)
    intervals_filled: int = 0
    dead_meters_dropped: int = 0
    dead_meter_rows_dropped: int = 0
    sparse_meters_dropped: int = 0
    sparse_meter_rows_dropped: int = 0
    gaps_remaining: int = 0

    @property
    def total_outliers_removed(self) -> int:
        return sum(self.outliers_removed.values())

    def summary(self) -> str:
        lines = [
            "=== Cleaning Report ===",
            f"Rows before:              {self.rows_before:,}",
            f"Rows after:               {self.rows_after:,}",
            f"NaN simscode dropped:     {self.nan_simscode_dropped:,}",
            f"Excluded utility dropped: {self.excluded_utility_dropped:,}",
            f"Excluded buildings dropped:{self.excluded_buildings_dropped:,}",
            f"Outliers removed (total): {self.total_outliers_removed:,}",
        ]
        for util, count in sorted(self.outliers_removed.items()):
            lines.append(f"  {util}: {count:,}")
        lines.append(f"Intervals filled:         {self.intervals_filled:,}")
        lines.append(f"Dead meters dropped:      {self.dead_meters_dropped:,} ({self.dead_meter_rows_dropped:,} rows)")
        lines.append(f"Sparse meters dropped:    {self.sparse_meters_dropped:,} ({self.sparse_meter_rows_dropped:,} rows)")
        lines.append(f"Gaps remaining (NaN):     {self.gaps_remaining:,}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1: Drop NaN simscode
# ---------------------------------------------------------------------------


def drop_nan_simscode(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Drop rows where simscode is NaN, cast remainder to int->str."""
    mask = df["simscode"].isna()
    report.nan_simscode_dropped = int(mask.sum())
    df = df[~mask].copy()
    # Clean join key: float -> int -> str (avoids "338.0")
    df["simscode"] = df["simscode"].astype(float).astype(int).astype(str)
    return df


# ---------------------------------------------------------------------------
# Step 2: Exclude utilities
# ---------------------------------------------------------------------------


def exclude_utilities(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Drop all rows for utilities in EXCLUDED_UTILITIES."""
    mask = df["utility"].isin(EXCLUDED_UTILITIES)
    report.excluded_utility_dropped = int(mask.sum())
    return df[~mask].copy()


# ---------------------------------------------------------------------------
# Step 3: Exclude unmatched buildings
# ---------------------------------------------------------------------------


def exclude_unmatched_buildings(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Drop rows for simscodes with no building metadata."""
    mask = df["simscode"].isin(EXCLUDED_SIMSCODES)
    report.excluded_buildings_dropped = int(mask.sum())
    return df[~mask].copy()


# ---------------------------------------------------------------------------
# Step 4: Apply hard caps
# ---------------------------------------------------------------------------


def apply_hard_caps(
    df: pd.DataFrame,
    report: CleaningReport,
    caps: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Drop rows where readingvalue exceeds per-utility hard cap.

    Unknown utilities (not in caps dict) are left untouched.
    """
    if caps is None:
        caps = UTILITY_HARD_CAPS

    keep_mask = pd.Series(True, index=df.index)

    for utility, cap in caps.items():
        util_mask = df["utility"] == utility
        over_cap = util_mask & (df["readingvalue"] > cap)
        n_removed = int(over_cap.sum())
        if n_removed > 0:
            report.outliers_removed[utility] = n_removed
            keep_mask &= ~over_cap

    return df[keep_mask].copy()


# ---------------------------------------------------------------------------
# Step 5: Impute short gaps
# ---------------------------------------------------------------------------


def impute_short_gaps(
    df: pd.DataFrame,
    report: CleaningReport,
    gap_limit: int = FFILL_GAP_LIMIT,
) -> pd.DataFrame:
    """Fill short NaN gaps in readingvalue via ffill + bfill within meter groups.

    Gaps longer than gap_limit intervals remain NaN.
    """
    nans_before = int(df["readingvalue"].isna().sum())

    df = df.sort_values(["meterid", "simscode", "utility", "readingtime"]).copy()
    group_cols = ["meterid", "simscode", "utility"]

    df["readingvalue"] = (
        df.groupby(group_cols)["readingvalue"]
        .transform(lambda s: s.ffill(limit=gap_limit).bfill(limit=gap_limit))
    )

    nans_after = int(df["readingvalue"].isna().sum())
    report.intervals_filled = nans_before - nans_after
    report.gaps_remaining = nans_after
    return df


# ---------------------------------------------------------------------------
# Step 6: Drop dead meters (100% NaN)
# ---------------------------------------------------------------------------


def drop_dead_meters(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Drop meters where every readingvalue is NaN (dead/offline meters)."""
    group_cols = ["meterid", "simscode", "utility"]
    nan_frac = df.groupby(group_cols)["readingvalue"].apply(
        lambda s: s.isna().all()
    )
    dead_keys = nan_frac[nan_frac].reset_index()[group_cols]

    if len(dead_keys) == 0:
        return df

    report.dead_meters_dropped = len(dead_keys)

    # Tag rows to drop via merge
    dead_keys["_dead"] = True
    merged = df.merge(dead_keys, on=group_cols, how="left")
    keep = merged["_dead"].isna()
    report.dead_meter_rows_dropped = int((~keep).sum())
    return df[keep.values].copy()


# ---------------------------------------------------------------------------
# Step 7: Drop sparse meters (>50% NaN after imputation)
# ---------------------------------------------------------------------------

SPARSE_THRESHOLD = 0.5  # drop meters with more than 50% NaN


def drop_sparse_meters(
    df: pd.DataFrame,
    report: CleaningReport,
    threshold: float = SPARSE_THRESHOLD,
) -> pd.DataFrame:
    """Drop meters where NaN fraction exceeds threshold (long outages)."""
    group_cols = ["meterid", "simscode", "utility"]
    nan_frac = df.groupby(group_cols)["readingvalue"].apply(
        lambda s: s.isna().mean()
    )
    sparse_keys = nan_frac[nan_frac > threshold].reset_index()[group_cols]

    if len(sparse_keys) == 0:
        return df

    report.sparse_meters_dropped = len(sparse_keys)

    sparse_keys["_sparse"] = True
    merged = df.merge(sparse_keys, on=group_cols, how="left")
    keep = merged["_sparse"].isna()
    report.sparse_meter_rows_dropped = int((~keep).sum())

    # Update remaining NaN count
    remaining = df[keep.values]
    report.gaps_remaining = int(remaining["readingvalue"].isna().sum())
    return remaining.copy()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def clean_meter_data(
    df: pd.DataFrame,
    building_df: pd.DataFrame,
    lookup_path: Optional[str] = None,
    fuzzy_threshold: float = 80.0,
    gap_limit: int = FFILL_GAP_LIMIT,
    caps: Optional[Dict[str, float]] = None,
):
    """Run all cleaning steps in order.

    Parameters match what data_loader.build_feature_matrix expects.
    building_df, lookup_path, and fuzzy_threshold are accepted for interface
    compatibility but unused (building filtering is handled by the loader).

    Returns:
        (cleaned_df, CleaningReport)
    """
    report = CleaningReport()
    report.rows_before = len(df)

    df = drop_nan_simscode(df, report)
    df = exclude_utilities(df, report)
    df = exclude_unmatched_buildings(df, report)
    df = apply_hard_caps(df, report, caps=caps)
    df = impute_short_gaps(df, report, gap_limit=gap_limit)
    df = drop_dead_meters(df, report)
    df = drop_sparse_meters(df, report)

    report.gaps_remaining = int(df["readingvalue"].isna().sum())
    report.rows_after = len(df)
    return df, report
