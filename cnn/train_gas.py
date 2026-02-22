#!/usr/bin/env python3
"""
Training script for CNN gas consumption prediction model.

Loads pre-engineered gas parquet, filters always-off buildings,
and trains a 1D CNN using the same architecture as the electricity CNN.

Usage:
    python cnn/train_gas.py                            # defaults
    python cnn/train_gas.py --seq-length 96            # overrides
    python cnn/train_gas.py --epochs 100 --lr 3e-4     # training overrides
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

from cnn.config_gas import (
    EnergyCNNGasConfig,
    setup_console_logging,
    setup_output_dir,
)
from cnn.model import (
    create_datasets,
    create_model,
    evaluate_model,
    get_predictions,
    save_model,
    train_model,
)


# ---------------------------------------------------------------------------
# Gas-specific data loading (same as lstm/train.py)
# ---------------------------------------------------------------------------

_META_COLS = {"simscode", "readingtime", "energy_per_sqft", "readingvalue",
              "grossarea", "buildingnumber", "building_idx"}


def load_gas_parquet(cfg):
    """Load pre-engineered gas parquet and preprocess.

    Returns:
        df_train, df_test, feature_cols, df_off
    """
    data_cfg = cfg.data

    print(f"\n--- Loading Data ---")
    df = pd.read_parquet(data_cfg.parquet_path)
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols, "
          f"{df['simscode'].nunique()} buildings")

    # Drop sparse cross-utility columns
    sparse = [c for c in df.columns
              if any(c.startswith(p) for p in data_cfg.sparse_prefixes)]
    if sparse:
        print(f"  Dropping {len(sparse)} sparse cross-utility columns")
        df = df.drop(columns=sparse)

    # Separate always-off buildings
    bldg_zero = df.groupby("simscode")["energy_per_sqft"].apply(
        lambda s: (np.abs(s) <= data_cfg.zero_threshold).mean())
    always_off = bldg_zero[bldg_zero > data_cfg.always_off_threshold].index
    active = bldg_zero[bldg_zero <= data_cfg.always_off_threshold].index
    df_off = df[df["simscode"].isin(always_off)].copy()
    df = df[df["simscode"].isin(active)].reset_index(drop=True)
    print(f"  Active: {len(active)} buildings ({len(df):,} rows)")
    print(f"  Always-off: {len(always_off)} buildings ({len(df_off):,} rows)")

    # All non-meta, non-target columns are features (CNN uses all as channels)
    feature_cols = [c for c in df.columns
                    if c not in _META_COLS and c != "energy_per_sqft"]

    # Fill NaN
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Temporal split
    split_dt = pd.Timestamp(data_cfg.split_date)
    df_train = df[df["readingtime"] < split_dt].copy()
    df_test = df[df["readingtime"] >= split_dt].copy()
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")
    print(f"  Features ({len(feature_cols)}): {feature_cols[:8]}...")

    return df_train, df_test, feature_cols, df_off


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN gas energy model")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--seq-length", type=int, default=None, help="Window size (timesteps)")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--stride", type=int, default=None, help="Sliding window stride")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--parquet", type=str, default=None, help="Path to gas parquet")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = EnergyCNNGasConfig()

    # Apply CLI overrides
    if args.name:
        cfg.name = args.name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.seq_length is not None:
        cfg.data.seq_length = args.seq_length
    if args.epochs is not None:
        cfg.cnn.epochs = args.epochs
    if args.lr is not None:
        cfg.cnn.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.stride is not None:
        cfg.data.stride = args.stride
    if args.patience is not None:
        cfg.cnn.early_stopping_patience = args.patience
    if args.parquet:
        cfg.data.parquet_path = args.parquet
    if args.no_early_stop:
        cfg.cnn.early_stopping_patience = 999

    import torch
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Setup output directory and logging
    run_dir = setup_output_dir(cfg)
    cleanup_logging = setup_console_logging(cfg, run_dir)

    hours_covered = cfg.data.seq_length * 15 / 60

    print("=" * 60)
    print(f"CNN Gas Consumption Prediction — {cfg.name}")
    print(f"Output: {run_dir}")
    print(f"Utility: {cfg.data.utility_filter}")
    print(f"Seq length: {cfg.data.seq_length} ({hours_covered:.1f}h)")
    print(f"Stride: {cfg.data.stride}")
    print(f"Conv channels: {cfg.cnn.conv_channels}")
    print(f"Kernel sizes: {cfg.cnn.kernel_sizes}")
    print(f"LR: {cfg.cnn.learning_rate}, weight_decay={cfg.cnn.weight_decay}")
    print(f"Batch size: {cfg.data.batch_size}")
    print(f"Patience: {cfg.cnn.early_stopping_patience}")
    print(f"Seed: {cfg.seed}")
    print("=" * 60)

    t0 = time.time()

    try:
        # 1. Load gas parquet and preprocess
        df_train, df_test, feature_cols, df_off = load_gas_parquet(cfg)

        # 2. Create windowed datasets (reuses CNN model's EnergySequenceDataset)
        print("\n--- Creating Sequence Datasets ---")
        train_ds, test_ds, scaler_stats = create_datasets(
            df_train, df_test, feature_cols, cfg.data
        )
        print(f"  Train windows: {len(train_ds):,}")
        print(f"  Test windows:  {len(test_ds):,}")

        # 3. Create model
        print("\n--- Model ---")
        n_features = len(feature_cols)
        model, device = create_model(
            cfg.cnn, n_features=n_features, seq_length=cfg.data.seq_length
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        print(f"  Device: {device}")
        print(model)

        # 4. Train
        print("\n--- Training ---")
        model = train_model(
            model, train_ds, test_ds,
            params=cfg.cnn,
            data_cfg=cfg.data,
            device=device,
            run_dir=run_dir,
            save_every=cfg.checkpointing.save_every_n_epochs,
            tb_cfg=cfg.tensorboard,
            resume_from=args.resume_from,
        )

        # 5. Evaluate
        print("\n--- Evaluation ---")
        metrics = evaluate_model(
            model, test_ds, cfg.data, device, scaler_stats,
            run_dir=run_dir, params=cfg.cnn,
        )
        print(f"  RMSE:  {metrics['rmse']:.6f}")
        print(f"  MAE:   {metrics['mae']:.6f}")
        print(f"  R²:    {metrics['r2']:.4f}")
        print(f"  MAPE:  {metrics['mape_pct']:.2f}%")

        # 6. Generate predictions on full dataset (active buildings only)
        print("\n--- Generating Predictions ---")
        df_active = pd.concat([df_train, df_test], ignore_index=True)
        df_with_preds = get_predictions(
            model, df_active, feature_cols, cfg.data, device, scaler_stats
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
