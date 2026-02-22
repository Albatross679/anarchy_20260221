"""
XGBoost model for energy consumption prediction with enhanced feature engineering.

Functions:
    engineer_features -- add lag, rolling, and interaction features
    create_model      -- instantiate XGBRegressor from config
    train_model       -- fit with early stopping
    evaluate_model    -- compute RMSE/MAE/R²/MAPE + SHAP + plots
    save_model        -- persist to native JSON format
    load_model        -- load from native JSON format
    get_predictions   -- add predicted and residual columns to DataFrame
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback

from xgb.config import XGBoostDataConfig, XGBoostParams, TensorBoardConfig, log_system_metrics_to_tb


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(
    df: pd.DataFrame,
    data_cfg: XGBoostDataConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add lag, rolling, and interaction features per building.

    Groups by simscode, sorts by readingtime, then adds:
    1. Lag features (shift by N 15-min intervals)
    2. Rolling mean/std over configurable windows
    3. Interaction features (temp × area, humidity × area)

    Drops rows with NaN from lags/rolling (beginning of each building's data).

    Returns:
        (df_clean, feature_cols) where feature_cols includes all original
        + engineered feature column names.
    """
    df = df.copy()

    # Determine base feature columns (before engineering)
    base_features = (
        data_cfg.weather_features
        + data_cfg.building_features
        + data_cfg.time_features
    )
    base_features = [c for c in base_features if c in df.columns]

    engineered_cols = []

    # Sort globally for efficiency
    df = df.sort_values(["simscode", "readingtime"]).reset_index(drop=True)

    # 1. Lag features (per building)
    intervals_per_hour = 4  # 15-min data
    for hours in data_cfg.lag_hours:
        n_intervals = hours * intervals_per_hour
        col_name = f"energy_lag_{n_intervals}"
        df[col_name] = df.groupby("simscode")["energy_per_sqft"].shift(n_intervals)
        engineered_cols.append(col_name)

    # 2. Rolling statistics (per building)
    for hours in data_cfg.rolling_windows:
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
    if data_cfg.add_interactions:
        if "temperature_2m" in df.columns and "grossarea" in df.columns:
            df["temp_x_area"] = df["temperature_2m"] * df["grossarea"]
            engineered_cols.append("temp_x_area")
        if "relative_humidity_2m" in df.columns and "grossarea" in df.columns:
            df["humidity_x_area"] = df["relative_humidity_2m"] * df["grossarea"]
            engineered_cols.append("humidity_x_area")

    # Drop rows with NaN from lag/rolling features
    all_feature_cols = base_features + engineered_cols
    df = df.dropna(subset=all_feature_cols).reset_index(drop=True)

    return df, all_feature_cols


# ---------------------------------------------------------------------------
# TensorBoard callback
# ---------------------------------------------------------------------------


class TensorBoardCallback(TrainingCallback):
    """XGBoost callback that logs per-round metrics and system stats to TensorBoard."""

    def __init__(
        self,
        writer,
        tb_cfg: TensorBoardConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        log_every: int = 10,
    ):
        self.writer = writer
        self.tb_cfg = tb_cfg
        self.X_train = X_train
        self.y_train = np.asarray(y_train)
        self.X_val = X_val
        self.y_val = np.asarray(y_val)
        self.log_every = log_every
        self._start_time = None
        self._last_epoch_time = None

    def after_iteration(self, model, epoch, evals_log) -> bool:
        if self.writer is None:
            return False

        now = time.time()
        if self._start_time is None:
            self._start_time = now
            self._last_epoch_time = now

        elapsed = now - self._start_time
        round_time = now - self._last_epoch_time
        self._last_epoch_time = now

        # Log built-in eval metrics (RMSE) every round
        for dataset, metrics in evals_log.items():
            for metric_name, values in metrics.items():
                val = values[-1]
                if isinstance(val, tuple):
                    val = val[0]
                self.writer.add_scalar(f"{dataset}/{metric_name}", val, epoch)

        # Log combined train/val loss (RMSE) for parity with neural models
        train_rmse = evals_log.get("validation_0", {}).get("rmse", [None])[-1]
        val_rmse = evals_log.get("validation_1", {}).get("rmse", [None])[-1]
        if train_rmse is not None and val_rmse is not None:
            if isinstance(train_rmse, tuple):
                train_rmse = train_rmse[0]
            if isinstance(val_rmse, tuple):
                val_rmse = val_rmse[0]
            self.writer.add_scalars("loss", {"train": train_rmse, "val": val_rmse}, epoch)

        # Compute MAE / R² on val set at configurable interval
        if epoch % self.log_every == 0:
            import xgboost as xgb

            dval = xgb.DMatrix(self.X_val)
            y_pred_val = model.predict(dval, iteration_range=(0, epoch + 1))
            val_mae = float(np.mean(np.abs(self.y_val - y_pred_val)))
            ss_res = np.sum((self.y_val - y_pred_val) ** 2)
            ss_tot = np.sum((self.y_val - np.mean(self.y_val)) ** 2)
            val_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            self.writer.add_scalar("metrics/val_rmse", float(np.sqrt(np.mean((self.y_val - y_pred_val) ** 2))), epoch)
            self.writer.add_scalar("metrics/val_mae", val_mae, epoch)
            self.writer.add_scalar("metrics/val_r2", val_r2, epoch)

            log_system_metrics_to_tb(self.writer, self.tb_cfg, epoch)

        # Timing
        self.writer.add_scalar("time/wall_clock_seconds", elapsed, epoch)
        self.writer.add_scalar("time/round_seconds", round_time, epoch)

        return False  # False = don't stop training


# ---------------------------------------------------------------------------
# Model API
# ---------------------------------------------------------------------------


def create_model(
    params: XGBoostParams, seed: int = 42
) -> Tuple[XGBRegressor, dict]:
    """Create XGBRegressor from config params.

    Returns (model, fit_params).
    """
    kwargs = dict(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        min_child_weight=params.min_child_weight,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        tree_method=params.tree_method,
        eval_metric=params.eval_metric,
        random_state=seed,
        n_jobs=-1,
    )
    if params.early_stopping_rounds > 0:
        kwargs["early_stopping_rounds"] = params.early_stopping_rounds

    model = XGBRegressor(**kwargs)
    return model, {}


def train_model(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fit_params: dict,
    params: Optional[XGBoostParams] = None,
    run_dir: Optional[Path] = None,
    tb_cfg: Optional[TensorBoardConfig] = None,
) -> XGBRegressor:
    """Fit with early stopping on the eval set, with optional TensorBoard logging."""
    if tb_cfg is None:
        tb_cfg = TensorBoardConfig()

    writer = None
    if tb_cfg.enabled and run_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = run_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

        # Log hyperparameters as text at training start
        if tb_cfg.log_hparams_text and params is not None:
            hparam_text = (
                "| Param | Value |\n|---|---|\n"
                f"| n_estimators | {params.n_estimators} |\n"
                f"| max_depth | {params.max_depth} |\n"
                f"| learning_rate | {params.learning_rate} |\n"
                f"| subsample | {params.subsample} |\n"
                f"| colsample_bytree | {params.colsample_bytree} |\n"
                f"| min_child_weight | {params.min_child_weight} |\n"
                f"| reg_alpha | {params.reg_alpha} |\n"
                f"| reg_lambda | {params.reg_lambda} |\n"
                f"| tree_method | {params.tree_method} |\n"
                f"| early_stopping_rounds | {params.early_stopping_rounds} |\n"
                f"| eval_metric | {params.eval_metric} |\n"
                f"| train_rows | {len(X_train):,} |\n"
                f"| test_rows | {len(X_test):,} |\n"
                f"| n_features | {X_train.shape[1]} |\n"
            )
            writer.add_text("hyperparameters", hparam_text, 0)

        # Initialize CPU baseline
        psutil.cpu_percent(interval=None)

    # Add TensorBoard callback (XGBoost 3.x: callbacks go on the model, not fit())
    callbacks = fit_params.pop("callbacks", [])
    if writer is not None:
        callbacks.append(TensorBoardCallback(
            writer, tb_cfg,
            X_train=X_train, y_train=y_train,
            X_val=X_test, y_val=y_test,
            log_every=10,
        ))
    if callbacks:
        model.set_params(callbacks=callbacks)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
        **fit_params,
    )

    if writer is not None:
        writer.close()
        print(f"  TensorBoard logs: {run_dir / 'tensorboard'}")

    return model


def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    run_dir: Optional[Path] = None,
) -> dict:
    """Evaluate model: metrics + feature importance + SHAP + diagnostic plots.

    If run_dir is provided, saves plots to run_dir/plots/.
    """
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    nonzero = y_test != 0
    if nonzero.sum() > 0:
        mape = float(
            np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100
        )
    else:
        mape = float("nan")

    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "n_test": len(y_test),
        "n_trees_used": (
            int(model.best_iteration + 1)
            if hasattr(model, "best_iteration") and model.best_iteration is not None
            else model.n_estimators
        ),
        "feature_importance": importance,
    }

    if run_dir is not None:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _save_plots(model, X_test, y_test, y_pred, feature_cols, plots_dir)

    return metrics


def _save_plots(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_cols: List[str],
    plots_dir: Path,
) -> None:
    """Generate and save diagnostic plots."""

    # 1. Feature importance bar chart (built-in)
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[-20:]  # top 20
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_cols[i] for i in sorted_idx],
        importance[sorted_idx],
    )
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top 20 Feature Importance")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Predicted vs Actual scatter
    y_true_arr = np.asarray(y_test)
    sample_size = min(5000, len(y_true_arr))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true_arr), size=sample_size, replace=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_arr[idx], y_pred[idx], alpha=0.3, s=4)
    lims = [
        min(y_true_arr[idx].min(), y_pred[idx].min()),
        max(y_true_arr[idx].max(), y_pred[idx].max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual (energy/sqft)")
    ax.set_ylabel("Predicted (energy/sqft)")
    ax.set_title(f"Predicted vs Actual  (R²={r2_score(y_true_arr, y_pred):.4f})")
    fig.tight_layout()
    fig.savefig(plots_dir / "pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Residual distribution
    residuals = y_true_arr - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=100, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Residual Distribution  (mean={residuals.mean():.6f}, std={residuals.std():.6f})"
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "residual_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. SHAP plots
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_sample_size = min(1000, len(X_test))
        X_sample = X_test.iloc[
            rng.choice(len(X_test), size=shap_sample_size, replace=False)
        ]
        shap_values = explainer.shap_values(X_sample)

        # SHAP summary (dot) plot
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        # SHAP importance (bar) plot
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.savefig(plots_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("  SHAP plots saved.")
    except ImportError:
        print("  shap not installed — skipping SHAP plots.")
    except Exception as e:
        print(f"  SHAP plot generation failed: {e}")


def save_model(model: XGBRegressor, path: str | Path) -> None:
    """Save model in XGBoost native JSON format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: str | Path) -> XGBRegressor:
    """Load model from XGBoost native JSON format."""
    model = XGBRegressor()
    model.load_model(str(path))
    return model


def get_predictions(
    model: XGBRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Add predicted and residual columns to DataFrame."""
    df = df.copy()
    df["predicted"] = model.predict(df[feature_cols])
    df["residual"] = df["energy_per_sqft"] - df["predicted"]
    return df
