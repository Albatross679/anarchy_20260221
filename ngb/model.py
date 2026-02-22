"""
NGBoost model for energy consumption prediction with uncertainty estimates.

NGBoost (Natural Gradient Boosting) provides probabilistic predictions,
outputting both mean and standard deviation for each prediction.

Functions:
    engineer_features -- add lag, rolling, and interaction features
    create_model      -- instantiate NGBRegressor with Normal distribution
    train_model       -- fit with early stopping
    evaluate_model    -- compute RMSE/MAE/R2/MAPE + SHAP + plots
    save_model        -- persist to joblib format
    load_model        -- load from joblib format
    get_predictions   -- add predicted, residual, and predicted_std columns
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from ngb.config import NGBoostDataConfig, NGBoostParams, TensorBoardConfig, log_system_metrics_to_tb


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(
    df: pd.DataFrame,
    data_cfg: NGBoostDataConfig,
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
# Model API
# ---------------------------------------------------------------------------


def create_model(
    params: NGBoostParams, seed: int = 42
) -> Tuple[NGBRegressor, dict]:
    """Create NGBRegressor with Normal distribution and LogScore.

    Uses a DecisionTreeRegressor as the base learner.
    """
    base_learner = DecisionTreeRegressor(
        max_depth=params.base_max_depth,
        min_samples_leaf=params.base_min_samples_leaf,
        random_state=seed,
    )

    model = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        Base=base_learner,
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        minibatch_frac=params.minibatch_frac,
        natural_gradient=True,
        random_state=seed,
        verbose=False,
    )
    return model, {}


def train_model(
    model: NGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fit_params: dict,
    params: Optional[NGBoostParams] = None,
    run_dir: Optional[Path] = None,
    tb_cfg: Optional[TensorBoardConfig] = None,
) -> NGBRegressor:
    """Fit NGBoost with early stopping. Logs final metrics to TensorBoard."""
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
                f"| n_estimators | {params.n_estimators} |\n"
                f"| learning_rate | {params.learning_rate} |\n"
                f"| minibatch_frac | {params.minibatch_frac} |\n"
                f"| base_max_depth | {params.base_max_depth} |\n"
                f"| base_min_samples_leaf | {params.base_min_samples_leaf} |\n"
                f"| early_stopping_rounds | {params.early_stopping_rounds} |\n"
                f"| train_rows | {len(X_train):,} |\n"
                f"| test_rows | {len(X_test):,} |\n"
                f"| n_features | {X_train.shape[1]} |\n"
            )
            writer.add_text("hyperparameters", hparam_text, 0)

    t0 = time.time()

    fit_kwargs = {}
    if params and params.early_stopping_rounds > 0:
        fit_kwargs["X_val"] = X_test
        fit_kwargs["Y_val"] = y_test
        fit_kwargs["early_stopping_rounds"] = params.early_stopping_rounds

    model.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.time() - t0

    # Log final metrics to TensorBoard
    if writer is not None:
        y_pred_val = model.predict(X_test)
        val_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_val)))
        val_mae = float(mean_absolute_error(y_test, y_pred_val))
        val_r2 = float(r2_score(y_test, y_pred_val))

        writer.add_scalar("metrics/val_rmse", val_rmse, 0)
        writer.add_scalar("metrics/val_mae", val_mae, 0)
        writer.add_scalar("metrics/val_r2", val_r2, 0)
        writer.add_scalar("time/wall_clock_seconds", elapsed, 0)

        log_system_metrics_to_tb(writer, tb_cfg, 0)
        writer.close()
        print(f"  TensorBoard logs: {run_dir / 'tensorboard'}")

    return model


def evaluate_model(
    model: NGBRegressor,
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

    # Feature importance: try model attribute, fall back to permutation importance
    # NGBoost feature_importances_ is 2D (one row per dist param), average across rows
    try:
        fi = model.feature_importances_
        fi = np.asarray(fi)
        if fi.ndim > 1:
            fi = fi.mean(axis=0)
        importance_values = fi
        importance = dict(zip(feature_cols, importance_values.tolist()))
    except AttributeError:
        from sklearn.inspection import permutation_importance

        perm_result = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        importance_values = perm_result.importances_mean
        importance = dict(zip(feature_cols, importance_values.tolist()))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "n_test": len(y_test),
        "n_trees_used": model.n_estimators,
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

    # SHAP: try TreeExplainer, fall back to model-agnostic Explainer (small sample)
    try:
        import shap

        shap_sample_size = min(100, len(X_test))
        X_sample = X_test.iloc[
            rng.choice(len(X_test), size=shap_sample_size, replace=False)
        ]

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            bg = X_test.iloc[rng.choice(len(X_test), size=50, replace=False)]
            explainer = shap.Explainer(model.predict, bg)
            shap_values = explainer(X_sample).values

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


def save_model(model: NGBRegressor, path: str | Path) -> None:
    """Save model in joblib format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> NGBRegressor:
    """Load model from joblib format."""
    return joblib.load(path)


def get_predictions(
    model: NGBRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Add predicted, residual, and uncertainty columns to DataFrame.

    NGBoost provides probabilistic predictions, so we also output
    the predicted standard deviation for uncertainty quantification.
    """
    df = df.copy()
    df["predicted"] = model.predict(df[feature_cols])
    df["residual"] = df["energy_per_sqft"] - df["predicted"]

    # Add uncertainty estimate (predicted std dev)
    dist = model.pred_dist(df[feature_cols])
    df["predicted_std"] = dist.scale

    return df
