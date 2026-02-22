#!/usr/bin/env python3
"""
Training script for two-stage XGBoost gas consumption model.

Stage 1: Classify on/off (binary) — XGBClassifier
Stage 2: Predict magnitude for "on" samples — XGBRegressor (log1p target)
Combined: prediction = 0 if off, else expm1(regressor output)

Fixes applied vs v1:
    1. Target-encode building identity (simscode → per-building mean)
    2. Log-transform regressor target (log1p) to compress 138x dynamic range
    3. Drop sparse cross-utility columns (steam/cooling/heat >60% NaN)
    4. Separate always-off buildings (>99.9% zero) — predict as 0

Usage:
    python xgb_gas/train.py
    python xgb_gas/train.py --parquet data/tree_features_gas_cross.parquet
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

from xgb_gas.config import (
    EnergyXGBoostGasConfig,
    setup_console_logging,
    setup_output_dir,
)
from xgb_gas.model import (
    create_binary_labels,
    create_classifier,
    create_regressor,
    evaluate_classifier,
    evaluate_combined,
    evaluate_regressor,
    get_predictions,
    save_models,
    train_classifier,
    train_regressor,
)

# Non-feature columns to exclude when deriving feature list from parquet
_META_COLS = {"simscode", "readingtime", "energy_per_sqft", "readingvalue",
              "grossarea", "buildingnumber"}

# Cross-utility prefixes with >60% NaN — too sparse to be useful
_SPARSE_PREFIXES = ("heat_", "steam_", "cooling_")

# Buildings with >99.9% zero readings are trivially predictable
_ALWAYS_OFF_THRESHOLD = 0.999


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-stage XGBoost gas model")
    parser.add_argument("--parquet", type=str,
                        default="data/tree_features_gas_cross.parquet",
                        help="Path to pre-engineered parquet file")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--zero-threshold", type=float, default=None, help="Cutoff for binary label")
    # Classifier overrides
    parser.add_argument("--cls-n-estimators", type=int, default=None, help="Classifier trees")
    parser.add_argument("--cls-max-depth", type=int, default=None, help="Classifier max depth")
    parser.add_argument("--cls-lr", type=float, default=None, help="Classifier learning rate")
    # Regressor overrides
    parser.add_argument("--reg-n-estimators", type=int, default=None, help="Regressor trees")
    parser.add_argument("--reg-max-depth", type=int, default=None, help="Regressor max depth")
    parser.add_argument("--reg-lr", type=float, default=None, help="Regressor learning rate")
    # Split
    parser.add_argument("--no-temporal-split", action="store_true", help="Use random split")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = EnergyXGBoostGasConfig()

    # Apply CLI overrides
    if args.name:
        cfg.name = args.name
    if args.seed is not None:
        cfg.seed = args.seed
    if args.zero_threshold is not None:
        cfg.zero_threshold = args.zero_threshold
    if args.cls_n_estimators is not None:
        cfg.classifier.n_estimators = args.cls_n_estimators
    if args.cls_max_depth is not None:
        cfg.classifier.max_depth = args.cls_max_depth
    if args.cls_lr is not None:
        cfg.classifier.learning_rate = args.cls_lr
    if args.reg_n_estimators is not None:
        cfg.regressor.n_estimators = args.reg_n_estimators
    if args.reg_max_depth is not None:
        cfg.regressor.max_depth = args.reg_max_depth
    if args.reg_lr is not None:
        cfg.regressor.learning_rate = args.reg_lr
    if args.no_temporal_split:
        cfg.data.temporal_split = False

    # Setup output directory and logging
    run_dir = setup_output_dir(cfg)
    cleanup_logging = setup_console_logging(cfg, run_dir)

    print("=" * 60)
    print(f"Two-Stage XGBoost Gas Model (v2) — {cfg.name}")
    print(f"Output: {run_dir}")
    print(f"Utility: {cfg.data.utility_filter}")
    print(f"Zero threshold: {cfg.zero_threshold}")
    print(f"Seed: {cfg.seed}")
    print("=" * 60)

    t0 = time.time()

    try:
        # ==============================================================
        # 1. Load data
        # ==============================================================
        parquet_path = Path(args.parquet)
        print(f"\n--- Loading Data ---")
        print(f"  Source: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded: {len(df):,} rows, {df.shape[1]} columns, "
              f"{df['simscode'].nunique()} buildings")

        # ==============================================================
        # FIX 3: Drop sparse cross-utility columns
        # ==============================================================
        sparse_cols = [c for c in df.columns
                       if any(c.startswith(p) for p in _SPARSE_PREFIXES)]
        if sparse_cols:
            print(f"\n--- Fix 3: Dropping {len(sparse_cols)} sparse cross-utility columns ---")
            for c in sparse_cols:
                pct = df[c].isna().mean() * 100
                print(f"    {c}: {pct:.1f}% NaN")
            df = df.drop(columns=sparse_cols)

        # Derive feature columns (everything except metadata)
        feature_cols = [c for c in df.columns if c not in _META_COLS]

        # Fill remaining NaN (electricity_* at ~5%) with 0
        nan_counts = df[feature_cols].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            print(f"  Filling NaN in {len(nan_cols)} columns with 0:")
            for col, cnt in nan_cols.items():
                print(f"    {col}: {cnt:,} NaN ({cnt/len(df)*100:.1f}%)")
            df[feature_cols] = df[feature_cols].fillna(0.0)

        # ==============================================================
        # FIX 4: Separate always-off buildings
        # ==============================================================
        target_col = "energy_per_sqft"
        bldg_zero_rate = df.groupby("simscode")[target_col].apply(
            lambda s: (np.abs(s) <= cfg.zero_threshold).mean()
        )
        always_off = bldg_zero_rate[bldg_zero_rate > _ALWAYS_OFF_THRESHOLD].index
        active_buildings = bldg_zero_rate[bldg_zero_rate <= _ALWAYS_OFF_THRESHOLD].index

        n_off = len(always_off)
        n_active = len(active_buildings)
        rows_before = len(df)

        df_off = df[df["simscode"].isin(always_off)].copy()
        df = df[df["simscode"].isin(active_buildings)].reset_index(drop=True)

        print(f"\n--- Fix 4: Separating always-off buildings ---")
        print(f"  Always-off (>{_ALWAYS_OFF_THRESHOLD*100:.1f}% zero): "
              f"{n_off} buildings, {len(df_off):,} rows")
        print(f"  Active: {n_active} buildings, {len(df):,} rows")

        # ==============================================================
        # FIX 1: Target-encode building identity
        # ==============================================================
        # We need to split first to avoid leakage, so do split, then encode

        # Analyze zero distribution on active buildings
        n_zeros = (np.abs(df[target_col]) <= cfg.zero_threshold).sum()
        pct_zeros = n_zeros / len(df) * 100
        print(f"\n--- Zero Analysis (active buildings only) ---")
        print(f"  Total samples: {len(df):,}")
        print(f"  Zero samples:  {n_zeros:,} ({pct_zeros:.1f}%)")
        print(f"  Non-zero:      {len(df) - n_zeros:,} ({100 - pct_zeros:.1f}%)")

        # ==============================================================
        # Train/test split
        # ==============================================================
        print("\n--- Train/Test Split ---")
        data_cfg = cfg.data

        if data_cfg.temporal_split:
            split_dt = pd.Timestamp(data_cfg.split_date)
            train_mask = df["readingtime"] < split_dt
            test_mask = df["readingtime"] >= split_dt
        else:
            np.random.seed(cfg.seed)
            train_mask = np.random.rand(len(df)) < data_cfg.random_split_ratio
            test_mask = ~train_mask

        # ==============================================================
        # FIX 1: Target-encode simscode using TRAINING set only
        # ==============================================================
        print("\n--- Fix 1: Target-encoding building identity ---")
        train_bldg_mean = df.loc[train_mask].groupby("simscode")[target_col].mean()
        global_mean = df.loc[train_mask, target_col].mean()
        df["building_target_enc"] = df["simscode"].map(train_bldg_mean).fillna(global_mean)
        feature_cols.append("building_target_enc")
        print(f"  Encoded {len(train_bldg_mean)} buildings (train-set means)")
        print(f"  Range: {train_bldg_mean.min():.8f} — {train_bldg_mean.max():.8f}")

        print(f"\n  Final features ({len(feature_cols)}): {feature_cols}")

        X_train = df.loc[train_mask, feature_cols]
        X_test = df.loc[test_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        y_test = df.loc[test_mask, target_col]
        print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

        # Create binary labels
        y_train_cls = create_binary_labels(y_train, cfg.zero_threshold)
        y_test_cls = create_binary_labels(y_test, cfg.zero_threshold)
        print(f"  Train on/off: {y_train_cls.sum():,} on / {(y_train_cls == 0).sum():,} off")
        print(f"  Test on/off:  {y_test_cls.sum():,} on / {(y_test_cls == 0).sum():,} off")

        # Auto scale_pos_weight
        n_neg = (y_train_cls == 0).sum()
        n_pos = y_train_cls.sum()
        if n_pos > 0:
            cfg.classifier.scale_pos_weight = float(n_neg / n_pos)
            print(f"  Auto scale_pos_weight: {cfg.classifier.scale_pos_weight:.2f}")

        # ===============================================================
        # STAGE 1: Classifier
        # ===============================================================
        print("\n" + "=" * 60)
        print("STAGE 1: Training Classifier (on/off)")
        print("=" * 60)

        clf, clf_fit_params = create_classifier(cfg.classifier, cfg.seed)
        clf = train_classifier(
            clf, X_train, y_train_cls, X_test, y_test_cls, clf_fit_params,
            params=cfg.classifier, run_dir=run_dir, tb_cfg=cfg.tensorboard,
        )

        print("\n--- Classifier Evaluation ---")
        cls_metrics = evaluate_classifier(clf, X_test, y_test_cls, feature_cols, run_dir)
        print(f"  Accuracy:  {cls_metrics['accuracy']:.4f}")
        print(f"  Precision: {cls_metrics['precision']:.4f}")
        print(f"  Recall:    {cls_metrics['recall']:.4f}")
        print(f"  F1:        {cls_metrics['f1']:.4f}")
        print(f"  AUC:       {cls_metrics['auc']:.4f}")

        # ===============================================================
        # STAGE 2: Regressor (non-zero, log1p target)
        # ===============================================================
        print("\n" + "=" * 60)
        print("STAGE 2: Training Regressor (non-zero samples)")
        print("=" * 60)

        train_on_mask = y_train_cls == 1
        test_on_mask = y_test_cls == 1
        X_train_on = X_train[train_on_mask]
        X_test_on = X_test[test_on_mask]

        y_train_on = y_train[train_on_mask]
        y_test_on = y_test[test_on_mask]
        print(f"  Non-zero train: {len(X_train_on):,} rows")
        print(f"  Non-zero test:  {len(X_test_on):,} rows")

        reg, reg_fit_params = create_regressor(cfg.regressor, cfg.seed)
        reg = train_regressor(
            reg, X_train_on, y_train_on, X_test_on, y_test_on, reg_fit_params,
            params=cfg.regressor, run_dir=run_dir, tb_cfg=cfg.tensorboard,
        )

        print("\n--- Regressor Evaluation (non-zero subset) ---")
        reg_metrics = evaluate_regressor(
            reg, X_test_on, y_test_on, feature_cols, run_dir,
        )
        print(f"  RMSE:  {reg_metrics['rmse']:.6f}")
        print(f"  MAE:   {reg_metrics['mae']:.6f}")
        print(f"  R²:    {reg_metrics['r2']:.4f}")
        print(f"  MAPE:  {reg_metrics['mape_pct']:.2f}%")
        print(f"  Trees: {reg_metrics['n_trees_used']}")

        # ===============================================================
        # COMBINED EVALUATION (active buildings)
        # ===============================================================
        print("\n" + "=" * 60)
        print("COMBINED: Two-Stage Pipeline Evaluation (active buildings)")
        print("=" * 60)

        combined_metrics = evaluate_combined(
            clf, reg, X_test, y_test, feature_cols, cfg.zero_threshold,
            run_dir, log_target=False,
        )
        best = combined_metrics['best_strategy']
        print(f"  Blending strategies:")
        print(f"    hard (0/1 gate):        R² = {combined_metrics['r2_hard']:.4f}")
        print(f"    soft (prob * reg):       R² = {combined_metrics['r2_soft']:.4f}")
        print(f"    optimal (cutoff={combined_metrics['optimal_prob_cutoff']:.2f}):  "
              f"R² = {combined_metrics['r2_optimal_threshold']:.4f}")
        print(f"  Best strategy: {best}")
        print(f"  Combined R²:   {combined_metrics['combined_r2']:.4f}")
        print(f"  Combined RMSE: {combined_metrics['combined_rmse']:.6f}")
        print(f"  Combined MAE:  {combined_metrics['combined_mae']:.6f}")
        print(f"  R² (non-zero): {combined_metrics['r2_nonzero_subset']:.4f}")
        print(f"  Zero cls accuracy:    {combined_metrics['zero_classification_accuracy']:.4f}")
        print(f"  Nonzero cls accuracy: {combined_metrics['nonzero_classification_accuracy']:.4f}")

        # ===============================================================
        # OVERALL R² including always-off buildings
        # ===============================================================
        # Always-off buildings are predicted as 0 (perfect for them)
        if data_cfg.temporal_split:
            off_test = df_off[df_off["readingtime"] >= pd.Timestamp(data_cfg.split_date)]
        else:
            np.random.seed(cfg.seed + 1)
            off_test_mask = np.random.rand(len(df_off)) >= data_cfg.random_split_ratio
            off_test = df_off[off_test_mask]

        y_all_true = np.concatenate([np.asarray(y_test), np.asarray(off_test[target_col])])

        # Get best predictions for active buildings
        blend_mode = best if best != "optimal_threshold" else "optimal"
        prob_cutoff = combined_metrics.get("optimal_prob_cutoff", 0.5)
        df_test_active = df.loc[test_mask].copy()
        df_preds_active = get_predictions(
            clf, reg, df_test_active, feature_cols, cfg.zero_threshold,
            blend_mode=blend_mode, prob_cutoff=prob_cutoff, log_target=False,
        )

        y_all_pred = np.concatenate([
            np.asarray(df_preds_active["predicted"]),
            np.zeros(len(off_test)),  # always-off → predict 0
        ])

        from sklearn.metrics import r2_score as _r2
        overall_r2 = float(_r2(y_all_true, y_all_pred))
        print(f"\n--- Overall R² (all {n_active + n_off} buildings) ---")
        print(f"  Active buildings R²:    {combined_metrics['combined_r2']:.4f}")
        print(f"  + {n_off} always-off (predicted=0)")
        print(f"  Overall R²:             {overall_r2:.4f}")

        # ===============================================================
        # SAVE
        # ===============================================================
        print("\n--- Saving ---")

        checkpoint_dir = run_dir / "checkpoints"
        save_models(clf, reg, checkpoint_dir)
        print(f"  Classifier saved: {checkpoint_dir / 'classifier_best.json'}")
        print(f"  Regressor saved:  {checkpoint_dir / 'regressor_best.json'}")

        # Save metrics
        all_metrics = {
            "classifier": cls_metrics,
            "regressor": reg_metrics,
            "combined_active": combined_metrics,
            "overall_r2": overall_r2,
            "zero_threshold": cfg.zero_threshold,
            "zero_pct_active": pct_zeros,
            "n_buildings_total": n_active + n_off,
            "n_buildings_active": n_active,
            "n_buildings_always_off": n_off,
            "log_target": False,
            "parquet_source": str(parquet_path),
        }
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(all_metrics, indent=2))
        print(f"  Metrics saved: {metrics_path}")

        # Save predictions (active + always-off combined)
        df_preds_off = off_test.copy()
        df_preds_off["class_pred"] = 0
        df_preds_off["class_prob"] = 0.0
        df_preds_off["predicted"] = 0.0
        df_preds_off["residual"] = df_preds_off[target_col]
        # Add building_target_enc for always-off so columns match
        df_preds_off["building_target_enc"] = 0.0

        shared_cols = [c for c in df_preds_active.columns if c in df_preds_off.columns]
        df_preds_all = pd.concat(
            [df_preds_active[shared_cols], df_preds_off[shared_cols]],
            ignore_index=True,
        )
        preds_path = run_dir / "predictions.parquet"
        df_preds_all.to_parquet(preds_path, index=False)
        print(f"  Predictions saved: {preds_path} ({len(df_preds_all):,} rows, "
              f"blend_mode={blend_mode})")

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"Done in {elapsed:.1f}s")
        print(f"Output directory: {run_dir}")
        print(f"Overall R²: {overall_r2:.4f} (vs previous 0.633)")
        print(f"{'=' * 60}")

    finally:
        cleanup_logging()


if __name__ == "__main__":
    main()
