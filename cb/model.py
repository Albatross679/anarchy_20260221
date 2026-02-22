"""
CatBoost model for energy consumption prediction with enhanced feature engineering.

Functions:
    engineer_features -- add lag, rolling, and interaction features
    create_model      -- instantiate CatBoostRegressor from config
    train_model       -- fit with early stopping
    evaluate_model    -- compute RMSE/MAE/R2/MAPE + SHAP + plots
    save_model        -- persist to native .cbm format
    load_model        -- load from native .cbm format
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
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cb.config import CatBoostDataConfig, CatBoostParams, TensorBoardConfig, log_system_metrics_to_tb


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(
    df: pd.DataFrame,
    data_cfg: CatBoostDataConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add lag, rolling, and interaction features per building."""
    df = df.copy()

    base_features = (
        data_cfg.weather_features
        + data_cfg.building_features
        + data_cfg.time_features
    )
    base_features = [c for c in base_features if c in df.columns]

    engineered_cols = []
    df = df.sort_values(["simscode", "readingtime"]).reset_index(drop=True)

    intervals_per_hour = 4
    for hours in data_cfg.lag_hours:
        n_intervals = hours * intervals_per_hour
        col_name = f"energy_lag_{n_intervals}"
        df[col_name] = df.groupby("simscode")["energy_per_sqft"].shift(n_intervals)
        engineered_cols.append(col_name)

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

    if data_cfg.add_interactions:
        if "temperature_2m" in df.columns and "grossarea" in df.columns:
            df["temp_x_area"] = df["temperature_2m"] * df["grossarea"]
            engineered_cols.append("temp_x_area")
        if "relative_humidity_2m" in df.columns and "grossarea" in df.columns:
            df["humidity_x_area"] = df["relative_humidity_2m"] * df["grossarea"]
            engineered_cols.append("humidity_x_area")

    all_feature_cols = base_features + engineered_cols
    df = df.dropna(subset=all_feature_cols).reset_index(drop=True)

    return df, all_feature_cols


# ---------------------------------------------------------------------------
# TensorBoard callback
# ---------------------------------------------------------------------------


class TensorBoardCatBoostCallback:
    """CatBoost callback that logs per-round metrics to TensorBoard.

    CatBoost callbacks use a custom class with an after_iteration method
    (not inheriting from a base callback class).
    """

    def __init__(
        self,
        writer,
        tb_cfg: TensorBoardConfig,
        log_every: int = 10,
    ):
        self.writer = writer
        self.tb_cfg = tb_cfg
        self.log_every = log_every
        self._start_time = None
        self._last_time = None

    def after_iteration(self, info):
        if self.writer is None:
            return True

        now = time.time()
        if self._start_time is None:
            self._start_time = now
            self._last_time = now

        iteration = info.iteration
        elapsed = now - self._start_time
        round_time = now - self._last_time
        self._last_time = now

        # Log metrics from info.metrics
        for dataset, metrics in info.metrics.items():
            for metric_name, values in metrics.items():
                val = values[-1]
                self.writer.add_scalar(f"{dataset}/{metric_name}", val, iteration)

        # Log timing
        self.writer.add_scalar("time/wall_clock_seconds", elapsed, iteration)
        self.writer.add_scalar("time/round_seconds", round_time, iteration)

        # Log system metrics periodically
        if iteration % self.log_every == 0:
            log_system_metrics_to_tb(self.writer, self.tb_cfg, iteration)

        return True  # True = continue training


# ---------------------------------------------------------------------------
# Model API
# ---------------------------------------------------------------------------


def create_model(
    params: CatBoostParams, seed: int = 42
) -> Tuple[CatBoostRegressor, dict]:
    """Create CatBoostRegressor from config params."""
    model = CatBoostRegressor(
        iterations=params.iterations,
        depth=params.depth,
        learning_rate=params.learning_rate,
        l2_leaf_reg=params.l2_leaf_reg,
        random_strength=params.random_strength,
        bagging_temperature=params.bagging_temperature,
        border_count=params.border_count,
        eval_metric=params.eval_metric,
        random_seed=seed,
        verbose=0,  # Suppress default output; we control logging
    )
    return model, {}


def train_model(
    model: CatBoostRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fit_params: dict,
    params: Optional[CatBoostParams] = None,
    run_dir: Optional[Path] = None,
    tb_cfg: Optional[TensorBoardConfig] = None,
) -> CatBoostRegressor:
    """Fit with early stopping on the eval set, with optional TensorBoard logging."""
    if tb_cfg is None:
        tb_cfg = TensorBoardConfig()

    writer = None
    if tb_cfg.enabled and run_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = run_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

        if tb_cfg.log_hparams_text and params is not None:
            hparam_text = (
                "| Param | Value |\n|---|---|\n"
                f"| iterations | {params.iterations} |\n"
                f"| depth | {params.depth} |\n"
                f"| learning_rate | {params.learning_rate} |\n"
                f"| l2_leaf_reg | {params.l2_leaf_reg} |\n"
                f"| random_strength | {params.random_strength} |\n"
                f"| bagging_temperature | {params.bagging_temperature} |\n"
                f"| border_count | {params.border_count} |\n"
                f"| early_stopping_rounds | {params.early_stopping_rounds} |\n"
                f"| eval_metric | {params.eval_metric} |\n"
                f"| train_rows | {len(X_train):,} |\n"
                f"| test_rows | {len(X_test):,} |\n"
                f"| n_features | {X_train.shape[1]} |\n"
            )
            writer.add_text("hyperparameters", hparam_text, 0)

        psutil.cpu_percent(interval=None)

    # Build fit kwargs
    fit_kwargs = dict(
        eval_set=(X_test, y_test),
        verbose=params.verbose if params else 50,
    )
    if params and params.early_stopping_rounds > 0:
        fit_kwargs["early_stopping_rounds"] = params.early_stopping_rounds

    # Add TensorBoard callback
    callbacks = []
    if writer is not None:
        callbacks.append(TensorBoardCatBoostCallback(writer, tb_cfg, log_every=10))
    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    model.fit(X_train, y_train, **fit_kwargs)

    if writer is not None:
        writer.close()
        print(f"  TensorBoard logs: {run_dir / 'tensorboard'}")

    return model


def evaluate_model(
    model: CatBoostRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    run_dir: Optional[Path] = None,
) -> dict:
    """Evaluate model: metrics + feature importance + SHAP + diagnostic plots."""
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

    importance_values = model.get_feature_importance()
    importance = dict(zip(feature_cols, importance_values.tolist()))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "n_test": len(y_test),
        "n_trees_used": model.get_best_iteration() or model.tree_count_,
        "feature_importance": importance,
    }

    if run_dir is not None:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _save_plots(model, X_test, y_test, y_pred, feature_cols, importance_values, plots_dir)

    return metrics


def _save_plots(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_cols: List[str],
    importance: np.ndarray,
    plots_dir: Path,
) -> None:
    """Generate and save diagnostic plots."""

    sorted_idx = np.argsort(importance)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_cols[i] for i in sorted_idx],
        importance[sorted_idx],
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 20 Feature Importance")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

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
    ax.set_title(f"Predicted vs Actual  (R2={r2_score(y_true_arr, y_pred):.4f})")
    fig.tight_layout()
    fig.savefig(plots_dir / "pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

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

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_sample_size = min(1000, len(X_test))
        X_sample = X_test.iloc[
            rng.choice(len(X_test), size=shap_sample_size, replace=False)
        ]
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.savefig(plots_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("  SHAP plots saved.")
    except ImportError:
        print("  shap not installed -- skipping SHAP plots.")
    except Exception as e:
        print(f"  SHAP plot generation failed: {e}")


def save_model(model: CatBoostRegressor, path: str | Path) -> None:
    """Save model in CatBoost native .cbm format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: str | Path) -> CatBoostRegressor:
    """Load model from CatBoost native .cbm format."""
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def get_predictions(
    model: CatBoostRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Add predicted and residual columns to DataFrame."""
    df = df.copy()
    df["predicted"] = model.predict(df[feature_cols])
    df["residual"] = df["energy_per_sqft"] - df["predicted"]
    return df
