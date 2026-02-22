#!/usr/bin/env python3
"""
Training script for Random Forest energy consumption prediction model.

Usage:
    python rf/train.py                                    # defaults
    python rf/train.py --name exp1 --n-estimators 100     # overrides
    python rf/train.py --utility STEAM --max-depth 30     # different utility
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rf.config import (
    EnergyRandomForestConfig,
    setup_console_logging,
    setup_output_dir,
)
from rf.model import (
    create_model,
    engineer_features,
    evaluate_model,
    get_predictions,
    save_model,
    train_model,
)
from src.data_loader import build_feature_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Forest energy model")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--utility", type=str, default=None, help="Utility type")
    parser.add_argument("--n-estimators", type=int, default=None, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth")
    parser.add_argument("--min-samples-split", type=int, default=None, help="Min samples to split")
    parser.add_argument("--min-samples-leaf", type=int, default=None, help="Min samples per leaf")
    parser.add_argument("--max-features", type=str, default=None, help="Max features per split")
    parser.add_argument("--no-temporal-split", action="store_true", help="Use random split")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = EnergyRandomForestConfig()

    # Apply CLI overrides
    if args.name:
        cfg.name = args.name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.utility:
        cfg.data.utility_filter = args.utility
        cfg.name = f"energy_random_forest_{args.utility.lower()}"
    if args.n_estimators is not None:
        cfg.rf.n_estimators = args.n_estimators
    if args.max_depth is not None:
        cfg.rf.max_depth = args.max_depth
    if args.min_samples_split is not None:
        cfg.rf.min_samples_split = args.min_samples_split
    if args.min_samples_leaf is not None:
        cfg.rf.min_samples_leaf = args.min_samples_leaf
    if args.max_features is not None:
        cfg.rf.max_features = args.max_features
    if args.no_temporal_split:
        cfg.data.temporal_split = False

    # Setup output directory and logging
    run_dir = setup_output_dir(cfg)
    cleanup_logging = setup_console_logging(cfg, run_dir)

    print("=" * 60)
    print(f"Random Forest Energy Prediction -- {cfg.name}")
    print(f"Output: {run_dir}")
    print(f"Utility: {cfg.data.utility_filter}")
    print(f"Seed: {cfg.seed}")
    print("=" * 60)

    t0 = time.time()

    try:
        # 1. Build feature matrix
        print("\n--- Data Pipeline ---")
        df = build_feature_matrix(cfg)

        # 2. Engineer features (lags, rolling, interactions)
        print("\n--- Feature Engineering ---")
        df, feature_cols = engineer_features(df, cfg.data)
        print(f"  Features ({len(feature_cols)}): {feature_cols}")
        print(f"  Dataset after engineering: {len(df):,} rows")

        # 3. Split (temporal or random)
        print("\n--- Train/Test Split ---")
        data_cfg = cfg.data
        target_col = "energy_per_sqft"

        if data_cfg.temporal_split:
            split_dt = pd.Timestamp(data_cfg.split_date)
            train_mask = df["readingtime"] < split_dt
            test_mask = df["readingtime"] >= split_dt
        else:
            np.random.seed(cfg.seed)
            train_mask = np.random.rand(len(df)) < data_cfg.random_split_ratio
            test_mask = ~train_mask

        X_train = df.loc[train_mask, feature_cols]
        X_test = df.loc[test_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        y_test = df.loc[test_mask, target_col]
        print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
        print(f"  Features ({len(feature_cols)}): {feature_cols}")

        # 4. Create & train model
        print("\n--- Training ---")
        model, fit_params = create_model(cfg.rf, cfg.seed)
        model = train_model(
            model, X_train, y_train, X_test, y_test, fit_params,
            params=cfg.rf, run_dir=run_dir, tb_cfg=cfg.tensorboard,
        )

        # 5. Evaluate (with SHAP + plots)
        print("\n--- Evaluation ---")
        metrics = evaluate_model(model, X_test, y_test, feature_cols, run_dir)
        print(f"  RMSE:  {metrics['rmse']:.6f}")
        print(f"  MAE:   {metrics['mae']:.6f}")
        print(f"  R2:    {metrics['r2']:.4f}")
        print(f"  MAPE:  {metrics['mape_pct']:.2f}%")
        print(f"  Trees: {metrics['n_trees_used']}")

        # Save metrics
        metrics_path = run_dir / "metrics.json"
        metrics_save = {k: v for k, v in metrics.items() if k != "feature_importance"}
        metrics_save["top_features"] = dict(
            sorted(metrics["feature_importance"].items(), key=lambda x: -x[1])[:10]
        )
        metrics_path.write_text(json.dumps(metrics_save, indent=2))

        # 6. Save model
        print("\n--- Saving ---")
        model_path = run_dir / "checkpoints" / "model_best.joblib"
        save_model(model, model_path)
        print(f"  Model saved: {model_path}")

        # 7. Generate predictions
        df_preds = get_predictions(model, df, feature_cols)
        preds_path = run_dir / "predictions.parquet"
        df_preds.to_parquet(preds_path, index=False)
        print(f"  Predictions saved: {preds_path}")

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"Done in {elapsed:.1f}s")
        print(f"Output directory: {run_dir}")
        print(f"{'=' * 60}")

    finally:
        cleanup_logging()


if __name__ == "__main__":
    main()
