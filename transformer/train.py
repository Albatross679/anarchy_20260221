#!/usr/bin/env python3
"""
Training script for Transformer energy consumption prediction model.

Usage:
    python transformer/train.py                                    # defaults
    python transformer/train.py --name exp1 --seq-length 48        # overrides
    python transformer/train.py --utility STEAM --epochs 100       # different utility
    python transformer/train.py --d-model 128 --n-heads 8          # transformer-specific
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from transformer.config import (
    EnergyTransformerConfig,
    setup_console_logging,
    setup_output_dir,
)
from transformer.model import (
    create_datasets,
    create_model,
    evaluate_model,
    get_predictions,
    save_model,
    train_model,
)
from src.data_loader import build_feature_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer energy model")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--utility", type=str, default=None, help="Utility type")
    parser.add_argument("--seq-length", type=int, default=None, help="Window size (timesteps)")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--no-temporal-split", action="store_true", help="Use random split")
    # Transformer-specific args
    parser.add_argument("--d-model", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--n-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=None, help="Number of encoder layers")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = EnergyTransformerConfig()

    # Apply CLI overrides
    if args.name:
        cfg.name = args.name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.utility:
        cfg.data.utility_filter = args.utility
    if args.seq_length is not None:
        cfg.data.seq_length = args.seq_length
    if args.epochs is not None:
        cfg.transformer.epochs = args.epochs
    if args.lr is not None:
        cfg.transformer.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.no_temporal_split:
        cfg.data.temporal_split = False
    # Transformer-specific overrides
    if args.d_model is not None:
        cfg.transformer.d_model = args.d_model
    if args.n_heads is not None:
        cfg.transformer.n_heads = args.n_heads
    if args.n_layers is not None:
        cfg.transformer.n_layers = args.n_layers

    # Setup output directory and logging
    run_dir = setup_output_dir(cfg)
    cleanup_logging = setup_console_logging(cfg, run_dir)

    print("=" * 60)
    print(f"Transformer Energy Prediction — {cfg.name}")
    print(f"Output: {run_dir}")
    print(f"Utility: {cfg.data.utility_filter}")
    print(f"Seq length: {cfg.data.seq_length}")
    print(f"Seed: {cfg.seed}")
    print("=" * 60)

    t0 = time.time()

    try:
        # 1. Build feature matrix (reuses src.data_loader)
        print("\n--- Data Pipeline ---")
        df = build_feature_matrix(cfg)

        # 2. Split into train/test DataFrames
        print("\n--- Train/Test Split ---")
        data_cfg = cfg.data
        feature_cols = (
            data_cfg.weather_features
            + data_cfg.building_features
            + data_cfg.time_features
        )
        feature_cols = [c for c in feature_cols if c in df.columns]

        if data_cfg.temporal_split:
            split_dt = pd.Timestamp(data_cfg.split_date)
            df_train = df[df["readingtime"] < split_dt].copy()
            df_test = df[df["readingtime"] >= split_dt].copy()
        else:
            np.random.seed(cfg.seed)
            mask = np.random.rand(len(df)) < data_cfg.random_split_ratio
            df_train = df[mask].copy()
            df_test = df[~mask].copy()

        print(f"Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")

        # 3. Create windowed datasets
        print("\n--- Creating Sequence Datasets ---")
        train_ds, test_ds, scaler_stats = create_datasets(
            df_train, df_test, feature_cols, data_cfg
        )
        print(f"  Train windows: {len(train_ds):,}")
        print(f"  Test windows:  {len(test_ds):,}")

        # 4. Create model
        print("\n--- Model ---")
        n_features = len(feature_cols)
        model, device = create_model(
            cfg.transformer, n_features=n_features, seq_length=data_cfg.seq_length
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        print(f"  Device: {device}")
        print(model)

        # 5. Train (logs to TensorBoard in run_dir/tensorboard/)
        print("\n--- Training ---")
        model = train_model(
            model, train_ds, test_ds,
            params=cfg.transformer,
            data_cfg=data_cfg,
            device=device,
            run_dir=run_dir,
            tb_cfg=cfg.tensorboard,
            resume_from=args.resume_from,
        )

        # 6. Evaluate (logs eval metrics to TensorBoard)
        print("\n--- Evaluation ---")
        metrics = evaluate_model(
            model, test_ds, data_cfg, device, scaler_stats,
            run_dir=run_dir, params=cfg.transformer,
        )
        print(f"  RMSE:  {metrics['rmse']:.6f}")
        print(f"  MAE:   {metrics['mae']:.6f}")
        print(f"  R²:    {metrics['r2']:.4f}")
        print(f"  MAPE:  {metrics['mape_pct']:.2f}%")

        # 7. Save model
        if cfg.checkpointing.enabled:
            model_path = run_dir / "checkpoints" / cfg.checkpointing.best_filename
            save_model(model, scaler_stats, model_path)
            print(f"\n  Model saved: {model_path}")

        # 8. Generate predictions
        print("\n--- Generating Predictions ---")
        df_with_preds = get_predictions(
            model, df, feature_cols, data_cfg, device, scaler_stats
        )
        preds_path = run_dir / "predictions.parquet"
        df_with_preds.to_parquet(preds_path, index=False)
        print(f"  Predictions saved: {preds_path}")

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"Done in {elapsed:.1f}s")
        print(f"Output directory: {run_dir}")
        print(f"TensorBoard:  tensorboard --logdir {run_dir / 'tensorboard'}")
        print(f"{'=' * 60}")

    finally:
        cleanup_logging()


if __name__ == "__main__":
    main()
