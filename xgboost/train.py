#!/usr/bin/env python3
"""
Training script for energy consumption prediction model.

Usage:
    python script/train.py                          # defaults
    python script/train.py --name exp1 --seed 123   # overrides
    python script/train.py --utility STEAM           # different utility
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    EnergyModelConfig,
    save_config,
    setup_console_logging,
    setup_output_dir,
)
from src.data_loader import build_feature_matrix, split_data
from src.model import (
    create_model,
    evaluate_model,
    get_predictions,
    save_model,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train energy consumption model")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--utility", type=str, default=None, help="Utility type to model")
    parser.add_argument("--n-estimators", type=int, default=None, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--no-temporal-split", action="store_true", help="Use random split instead")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = EnergyModelConfig()

    # Apply CLI overrides
    if args.name:
        cfg.name = args.name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.utility:
        cfg.data.utility_filter = args.utility
        cfg.name = f"energy_{args.utility.lower()}"
    if args.n_estimators is not None:
        cfg.xgb.n_estimators = args.n_estimators
    if args.max_depth is not None:
        cfg.xgb.max_depth = args.max_depth
    if args.lr is not None:
        cfg.xgb.learning_rate = args.lr
    if args.no_temporal_split:
        cfg.data.temporal_split = False

    # Setup output directory and logging
    run_dir = setup_output_dir(cfg)
    cleanup_logging = setup_console_logging(cfg, run_dir)

    print("=" * 60)
    print(f"Energy Consumption Prediction — {cfg.name}")
    print(f"Output: {run_dir}")
    print(f"Utility: {cfg.data.utility_filter}")
    print(f"Seed: {cfg.seed}")
    print("=" * 60)

    t0 = time.time()

    try:
        # Build feature matrix
        print("\n--- Data Pipeline ---")
        df = build_feature_matrix(cfg)

        # Split
        print("\n--- Train/Test Split ---")
        X_train, X_test, y_train, y_test, feature_cols = split_data(df, cfg)

        # Create and train model
        print("\n--- Training ---")
        model, fit_params = create_model(cfg.xgb, seed=cfg.seed)
        model = train_model(model, X_train, y_train, X_test, y_test, fit_params)

        # Evaluate
        print("\n--- Evaluation ---")
        metrics = evaluate_model(model, X_test, y_test)
        print(f"  RMSE:  {metrics['rmse']:.6f}")
        print(f"  MAE:   {metrics['mae']:.6f}")
        print(f"  R²:    {metrics['r2']:.4f}")
        print(f"  MAPE:  {metrics['mape_pct']:.2f}%")
        print(f"  Trees: {metrics['n_trees_used']}")

        print("\n  Feature importance:")
        sorted_imp = sorted(metrics["feature_importance"].items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_imp:
            print(f"    {feat:30s} {imp:.4f}")

        # Save model
        if cfg.checkpointing.enabled:
            model_path = run_dir / "checkpoints" / cfg.checkpointing.best_filename
            save_model(model, model_path)
            print(f"\n  Model saved: {model_path}")

        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved: {metrics_path}")

        # Generate predictions with residuals for scoring pipeline
        print("\n--- Generating Predictions ---")
        df_with_preds = get_predictions(model, df, feature_cols)
        preds_path = run_dir / "predictions.parquet"
        df_with_preds.to_parquet(preds_path, index=False)
        print(f"  Predictions saved: {preds_path}")
        print(f"  Columns: {list(df_with_preds.columns)}")

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"Done in {elapsed:.1f}s")
        print(f"Output directory: {run_dir}")
        print(f"{'=' * 60}")

    finally:
        cleanup_logging()


if __name__ == "__main__":
    main()
