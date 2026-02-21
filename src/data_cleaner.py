"""
Data cleaning pipeline for energy meter data.

Addresses four data integrity issues identified in the audit:
1. GAS unit mismatch (~248k records in kWh instead of kg)
2. Extreme outliers (readings >10^8 from sensor faults)
3. Building metadata linkage gap (24% unmatched simscodes)
4. Non-random missing data inflating STEAM under mean imputation

Functions are independently callable and return updated data + report fragments.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAS_KWH_TO_KG = 1 / 29.3  # natural gas lower heating value conversion

UTILITY_HARD_CAPS: Dict[str, float] = {
    "ELECTRICITY": 10_000,
    "ELECTRICAL_POWER": 10_000,
    "GAS": 50_000,
    "STEAM": 1_000_000,
    "STEAMRATE": 1_000_000,
    "HEAT": 10_000,
    "COOLING": 10_000,
    "COOLING_POWER": 10_000,
    "OIL28SEC": 50_000,
}

FFILL_GAP_LIMIT = 8  # 2 hours at 15-min intervals

WINDOW_STAT_COLS = [
    "readingwindowsum",
    "readingwindowmean",
    "readingwindowmin",
    "readingwindowmax",
    "readingwindowstandarddeviation",
]


# ---------------------------------------------------------------------------
# CleaningReport
# ---------------------------------------------------------------------------


@dataclass
class CleaningReport:
    """Collects counts from every cleaning operation."""

    gas_rows_converted: int = 0
    outliers_removed: Dict[str, int] = field(default_factory=dict)
    total_outliers_removed: int = 0
    matched_direct: int = 0
    matched_fuzzy: int = 0
    unmatched_simscodes: int = 0
    intervals_filled: int = 0
    gaps_too_long: int = 0

    def summary(self) -> str:
        lines = ["=== Data Cleaning Report ==="]
        lines.append(f"GAS rows converted (kWh -> kg): {self.gas_rows_converted:,}")
        lines.append(f"Outliers removed (total): {self.total_outliers_removed:,}")
        for util, count in sorted(self.outliers_removed.items()):
            lines.append(f"  {util}: {count:,}")
        lines.append(f"Metadata linkage — direct: {self.matched_direct:,}, "
                      f"fuzzy: {self.matched_fuzzy:,}, "
                      f"unmatched: {self.unmatched_simscodes:,}")
        lines.append(f"Intervals filled (ffill/bfill): {self.intervals_filled:,}")
        lines.append(f"Gaps too long to fill: {self.gaps_too_long:,}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Fix GAS units
# ---------------------------------------------------------------------------


def fix_gas_units(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Convert GAS rows where readingunits == 'kWh' from kWh to kg.

    Also scales window stats columns. Checks both utility AND unit to avoid
    double-conversion.
    """
    df = df.copy()

    mask = (df["utility"] == "GAS") & (df["readingunits"] == "kWh")
    n_converted = mask.sum()

    if n_converted > 0:
        df.loc[mask, "readingvalue"] *= GAS_KWH_TO_KG

        for col in WINDOW_STAT_COLS:
            if col in df.columns:
                df.loc[mask, col] *= GAS_KWH_TO_KG

        df.loc[mask, "readingunits"] = "kg"

    report.gas_rows_converted = int(n_converted)
    return df


# ---------------------------------------------------------------------------
# 2. Apply hard caps
# ---------------------------------------------------------------------------


def apply_hard_caps(
    df: pd.DataFrame,
    report: CleaningReport,
    caps: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Drop rows where readingvalue exceeds per-utility hard cap.

    Values >10^8 are sensor faults, not outliers — drop, don't clip.
    """
    caps = caps if caps is not None else UTILITY_HARD_CAPS
    df = df.copy()

    total_removed = 0
    for utility, cap in caps.items():
        over_mask = (df["utility"] == utility) & (df["readingvalue"] > cap)
        n_over = over_mask.sum()
        if n_over > 0:
            report.outliers_removed[utility] = int(n_over)
            total_removed += n_over
            df = df[~over_mask]

    report.total_outliers_removed = int(total_removed)
    return df


# ---------------------------------------------------------------------------
# 3. Build site lookup (fuzzy matching)
# ---------------------------------------------------------------------------


def _fuzzy_score(a: str, b: str) -> float:
    """Score similarity between two strings, using rapidfuzz if available."""
    try:
        from rapidfuzz.fuzz import token_set_ratio
        return token_set_ratio(a, b)
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100


def build_site_lookup(
    meter_df: pd.DataFrame,
    building_df: pd.DataFrame,
    threshold: float = 80.0,
) -> pd.DataFrame:
    """Generate a candidate simscode -> buildingnumber mapping table.

    Matching strategies:
    1. Direct match: simscode == buildingnumber
    2. Fuzzy name match: sitename ~ buildingname via token_set_ratio
    """
    # Get unique meter sites
    sites = meter_df[["simscode", "sitename"]].drop_duplicates("simscode").copy()
    sites["simscode"] = sites["simscode"].astype(str).str.strip()
    sites["sitename"] = sites["sitename"].astype(str).str.strip()

    buildings = building_df[["buildingnumber", "buildingname"]].copy()
    buildings["buildingnumber"] = buildings["buildingnumber"].astype(str).str.strip()
    buildings["buildingname"] = buildings["buildingname"].astype(str).str.strip()

    building_numbers = set(buildings["buildingnumber"])

    rows = []
    for _, site in sites.iterrows():
        sc = site["simscode"]
        sn = site["sitename"]

        # Try direct match first
        if sc in building_numbers:
            bldg = buildings[buildings["buildingnumber"] == sc].iloc[0]
            rows.append({
                "simscode": sc,
                "sitename": sn,
                "buildingnumber": sc,
                "buildingname": bldg["buildingname"],
                "match_type": "direct",
                "fuzzy_score": 100.0,
            })
            continue

        # Try fuzzy name match
        best_score = 0.0
        best_match = None
        for _, bldg in buildings.iterrows():
            score = _fuzzy_score(sn, bldg["buildingname"])
            if score > best_score:
                best_score = score
                best_match = bldg

        if best_match is not None and best_score >= threshold:
            rows.append({
                "simscode": sc,
                "sitename": sn,
                "buildingnumber": best_match["buildingnumber"],
                "buildingname": best_match["buildingname"],
                "match_type": "fuzzy",
                "fuzzy_score": best_score,
            })
        else:
            rows.append({
                "simscode": sc,
                "sitename": sn,
                "buildingnumber": None,
                "buildingname": None,
                "match_type": "unmatched",
                "fuzzy_score": best_score if best_match is not None else 0.0,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Load or create site lookup
# ---------------------------------------------------------------------------


def load_or_create_site_lookup(
    meter_df: pd.DataFrame,
    building_df: pd.DataFrame,
    path: str = "data/site_to_building_lookup.csv",
    threshold: float = 80.0,
) -> pd.DataFrame:
    """Load existing lookup CSV or create one via build_site_lookup."""
    p = Path(path)
    if p.exists():
        lookup = pd.read_csv(p, dtype=str)
        # Ensure score is float
        if "fuzzy_score" in lookup.columns:
            lookup["fuzzy_score"] = pd.to_numeric(lookup["fuzzy_score"], errors="coerce")
        return lookup

    warnings.warn(
        f"No site lookup found at {path}. Generating one — please review "
        f"and commit the file before production use.",
        stacklevel=2,
    )
    lookup = build_site_lookup(meter_df, building_df, threshold=threshold)
    p.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(p, index=False)
    return lookup


# ---------------------------------------------------------------------------
# 5. Apply metadata linkage
# ---------------------------------------------------------------------------


def apply_metadata_linkage(
    meter_df: pd.DataFrame,
    building_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    report: CleaningReport,
) -> pd.DataFrame:
    """Add resolved_buildingnumber column to meter_df using lookup table.

    Does NOT drop unmatched rows — that happens at the inner join in
    build_feature_matrix.
    """
    meter_df = meter_df.copy()

    # Build mapping from lookup (only matched entries)
    matched = lookup_df[lookup_df["match_type"] != "unmatched"].copy()
    mapping = dict(zip(matched["simscode"], matched["buildingnumber"]))

    meter_df["simscode_str"] = meter_df["simscode"].astype(str).str.strip()
    meter_df["resolved_buildingnumber"] = meter_df["simscode_str"].map(mapping)

    n_direct = int(lookup_df["match_type"].eq("direct").sum())
    n_fuzzy = int(lookup_df["match_type"].eq("fuzzy").sum())
    n_unmatched = int(lookup_df["match_type"].eq("unmatched").sum())

    report.matched_direct = n_direct
    report.matched_fuzzy = n_fuzzy
    report.unmatched_simscodes = n_unmatched

    # Log unmatched simscodes
    unmatched_codes = meter_df.loc[
        meter_df["resolved_buildingnumber"].isna(), "simscode_str"
    ].unique()
    if len(unmatched_codes) > 0:
        warnings.warn(
            f"{len(unmatched_codes)} simscodes have no building match: "
            f"{list(unmatched_codes[:10])}{'...' if len(unmatched_codes) > 10 else ''}",
            stacklevel=2,
        )

    meter_df.drop(columns=["simscode_str"], inplace=True)
    return meter_df


# ---------------------------------------------------------------------------
# 6. Impute missing intervals
# ---------------------------------------------------------------------------


def impute_missing_intervals(
    df: pd.DataFrame,
    report: CleaningReport,
    gap_limit: int = FFILL_GAP_LIMIT,
) -> pd.DataFrame:
    """Per-meter ffill then bfill on readingvalue, respecting gap limits.

    Groups by (meterid, simscode, utility) to avoid filling across meter
    boundaries. Gaps longer than gap_limit remain NaN.
    """
    df = df.copy()

    # Determine grouping columns (use what's available)
    group_cols = [c for c in ["meterid", "simscode", "utility"] if c in df.columns]
    if not group_cols:
        group_cols = ["simscode"]

    na_before = int(df["readingvalue"].isna().sum())

    if na_before == 0:
        return df

    # Sort for fill correctness
    df = df.sort_values(group_cols + ["readingtime"])

    def _fill_group(g):
        g = g.copy()
        g["readingvalue"] = (
            g["readingvalue"]
            .ffill(limit=gap_limit)
            .bfill(limit=gap_limit)
        )
        return g

    df = df.groupby(group_cols, group_keys=False).apply(_fill_group)

    na_after = int(df["readingvalue"].isna().sum())
    report.intervals_filled = na_before - na_after
    report.gaps_too_long = na_after

    return df


# ---------------------------------------------------------------------------
# 7. Top-level orchestrator
# ---------------------------------------------------------------------------


def clean_meter_data(
    df: pd.DataFrame,
    building_df: pd.DataFrame,
    lookup_path: str = "data/site_to_building_lookup.csv",
    fuzzy_threshold: float = 80.0,
    gap_limit: int = FFILL_GAP_LIMIT,
    caps: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Run all cleaning steps in the correct order.

    Order matters: unit fix MUST precede hard caps because GAS values in kWh
    are ~29x larger than in kg.

    Returns (cleaned_df, CleaningReport).
    """
    report = CleaningReport()

    # Step 1: Fix GAS units (before caps!)
    df = fix_gas_units(df, report)

    # Step 2: Apply hard caps
    df = apply_hard_caps(df, report, caps=caps)

    # Step 3-5: Metadata linkage
    lookup_df = load_or_create_site_lookup(
        df, building_df, path=lookup_path, threshold=fuzzy_threshold,
    )
    df = apply_metadata_linkage(df, building_df, lookup_df, report)

    # Step 6: Impute missing intervals
    df = impute_missing_intervals(df, report, gap_limit=gap_limit)

    print(report.summary())
    return df, report
