#!/usr/bin/env python3
"""
Pre-compute tree-model features and save to parquet.

Runs build_feature_matrix() → engineer_features() once per utility, saving the
result so that all 8 tree-based training scripts can skip the pipeline entirely
with --precomputed.

Usage:
    python src/prepare_tree_features.py                          # all utilities
    python src/prepare_tree_features.py --utilities ELECTRICITY GAS
    python src/prepare_tree_features.py --utilities ELECTRICITY --out-dir data
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import DataConfig, MLBaseConfig
from src.data_loader import build_feature_matrix
from src.feature_engineer import engineer_features

ALL_UTILITIES = [
    "ELECTRICITY",
    "GAS",
    "HEAT",
    "STEAM",
    "COOLING",
    # Excluded: OIL28SEC (100% zeros), STEAMRATE (single meter),
    # ELECTRICAL_POWER / COOLING_POWER (no data after cleaning)
]

# Default engineering parameters (match tree model defaults)
DEFAULT_LAG_HOURS = [1, 6, 24, 168]
DEFAULT_ROLLING_WINDOWS = [24, 168]
DEFAULT_ADD_INTERACTIONS = True

# Metadata columns preserved alongside features
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
        description="Pre-compute tree-model features to parquet"
    )
    parser.add_argument(
        "--utilities",
        nargs="+",
        default=None,
        help="Utility types to process (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output directory for parquet files (default: data)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    utilities = args.utilities or ALL_UTILITIES
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared data config for feature lists
    data_cfg = DataConfig()

    manifest = {
        "description": "Pre-computed tree-model features",
        "lag_hours": DEFAULT_LAG_HOURS,
        "rolling_windows": DEFAULT_ROLLING_WINDOWS,
        "add_interactions": DEFAULT_ADD_INTERACTIONS,
        "utilities": {},
    }

    t0 = time.time()

    for utility in utilities:
        print(f"\n{'=' * 60}")
        print(f"Processing {utility}")
        print(f"{'=' * 60}")

        # Build a minimal config for this utility
        cfg = MLBaseConfig()
        cfg.data = DataConfig(utility_filter=utility)

        try:
            # run_cleaning=True ensures the full data_cleaner pipeline
            # (hard caps, gap imputation, dead/sparse meter removal) is
            # applied when no pre-cleaned CSV exists for this utility.
            df = build_feature_matrix(cfg, run_cleaning=True)
        except Exception as e:
            print(f"  SKIP {utility}: {e}")
            continue

        if len(df) == 0:
            print(f"  SKIP {utility}: empty after pipeline")
            continue

        # Engineer features using shared canonical function
        df, feature_cols = engineer_features(
            df,
            weather_features=data_cfg.weather_features,
            building_features=data_cfg.building_features,
            time_features=data_cfg.time_features,
            lag_hours=DEFAULT_LAG_HOURS,
            rolling_windows=DEFAULT_ROLLING_WINDOWS,
            add_interactions=DEFAULT_ADD_INTERACTIONS,
        )
        print(f"  After engineering: {len(df):,} rows, {len(feature_cols)} features")

        # Select columns to save: metadata + features
        keep_cols = [c for c in METADATA_COLS if c in df.columns] + feature_cols
        # Deduplicate while preserving order
        seen = set()
        keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

        df_out = df[keep_cols]

        # Write parquet
        parquet_path = out_dir / f"tree_features_{utility.lower()}.parquet"
        df_out.to_parquet(parquet_path, index=False)
        print(f"  Saved: {parquet_path} ({len(df_out):,} rows, {len(keep_cols)} cols)")

        manifest["utilities"][utility] = {
            "parquet_file": str(parquet_path),
            "rows": len(df_out),
            "buildings": int(df_out["simscode"].nunique()),
            "feature_cols": feature_cols,
            "all_cols": keep_cols,
        }

    # Write manifest
    manifest_path = out_dir / "tree_features_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")

    elapsed = time.time() - t0
    n_done = len(manifest["utilities"])
    print(f"\nDone: {n_done} utilities in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
