#!/usr/bin/env python3
"""
Training script for NGBoost energy consumption prediction model.

NGBoost provides probabilistic predictions with uncertainty estimates.

Usage:
    python ngb/train.py                                    # defaults
    python ngb/train.py --name exp1 --n-estimators 200     # overrides
    python ngb/train.py --utility STEAM --lr 0.05          # different utility
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ngb.config import (
    EnergyNGBoostConfig,
    setup_console_logging,
    setup_output_dir,
)
from ngb.model import (
    create_model,
    engineer_features,
    evaluate_model,
    get_predictions,
    save_model,
    train_model,
)
from src.data_loader import build_feature_matrix, load_precomputed_tree_features


def parse_args():
    parser = argparse.ArgumentParser(description="Train NGBoost energy model")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--utility", type=str, default=None, help="Utility type")
    parser.add_argument("--n-estimators", type=int, default=None, help="Number of boosting stages")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--minibatch-frac", type=float, default=None, help="Minibatch fraction")
    parser.add_argument("--base-max-depth", type=int, default=None, help="Base learner max depth")
    parser.add_argument("--no-temporal-split", action="store_true", help="Use random split")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--precomputed", action="store_true", help="Load pre-computed features from parquet")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = EnergyNGBoostConfig()

    if args.name:
        cfg.name = args.name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.utility:
        cfg.data.utility_filter = args.utility
    if args.n_estimators is not None:
        cfg.ngb.n_estimators = args.n_estimators
    if args.lr is not None:
        cfg.ngb.learning_rate = args.lr
    if args.minibatch_frac is not None:
        cfg.ngb.minibatch_frac = args.minibatch_frac
    if args.base_max_depth is not None:
        cfg.ngb.base_max_depth = args.base_max_depth
    if args.no_temporal_split:
        cfg.data.temporal_split = False
    if args.no_early_stop:
        cfg.ngb.early_stopping_rounds = 0

    run_dir = setup_output_dir(cfg)
    cleanup_logging = setup_console_logging(cfg, run_dir)

    print("=" * 60)
    print(f"NGBoost Energy Prediction -- {cfg.name}")
    print(f"Output: {run_dir}")
    print(f"Utility: {cfg.data.utility_filter}")
    print(f"Seed: {cfg.seed}")
    print("=" * 60)

    t0 = time.time()

    try:
        # 1. Build feature matrix + engineer features
        if args.precomputed:
            print("\n--- Loading Pre-computed Features ---")
            df, feature_cols = load_precomputed_tree_features(
                cfg.data.utility_filter, cfg.data.precomputed_features_dir,
            )
        else:
            print("\n--- Data Pipeline ---")
            df = build_feature_matrix(cfg)
            print("\n--- Feature Engineering ---")
            df, feature_cols = engineer_features(df, cfg.data)

        print(f"  Features ({len(feature_cols)}): {feature_cols}")
        print(f"  Dataset: {len(df):,} rows")

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

        print("\n--- Training ---")
        model, fit_params = create_model(cfg.ngb, cfg.seed)
        model = train_model(
            model, X_train, y_train, X_test, y_test, fit_params,
            params=cfg.ngb, run_dir=run_dir, tb_cfg=cfg.tensorboard,
        )

        print("\n--- Evaluation ---")
        metrics = evaluate_model(model, X_test, y_test, feature_cols, run_dir)
        print(f"  RMSE:  {metrics['rmse']:.6f}")
        print(f"  MAE:   {metrics['mae']:.6f}")
        print(f"  R2:    {metrics['r2']:.4f}")
        print(f"  MAPE:  {metrics['mape_pct']:.2f}%")
        print(f"  Trees: {metrics['n_trees_used']}")

        metrics_path = run_dir / "metrics.json"
        metrics_save = {k: v for k, v in metrics.items() if k != "feature_importance"}
        metrics_save["top_features"] = dict(
            sorted(metrics["feature_importance"].items(), key=lambda x: -x[1])[:10]
        )
        metrics_path.write_text(json.dumps(metrics_save, indent=2))

        print("\n--- Saving ---")
        model_path = run_dir / "checkpoints" / "model_best.joblib"
        save_model(model, model_path)
        print(f"  Model saved: {model_path}")

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
