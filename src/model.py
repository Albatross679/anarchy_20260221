"""
XGBoost model for energy consumption prediction.

Functions:
    create_model     -- instantiate XGBRegressor from config
    train_model      -- fit with early stopping
    evaluate_model   -- compute RMSE, MAE, R², MAPE, feature importance
    save_model       -- persist to JSON
    load_model       -- load from JSON
    get_predictions  -- add predicted and residual columns to a DataFrame
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import XGBoostParams


def create_model(params: XGBoostParams, seed: int = 42) -> tuple[XGBRegressor, dict]:
    """
    Create XGBRegressor from config params.

    Returns (model, fit_params) — fit_params is empty dict (kept for API compat).
    XGBoost 3.x takes early_stopping_rounds and eval_metric in the constructor.
    """
    kwargs = dict(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        min_child_weight=params.min_child_weight,
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
) -> XGBRegressor:
    """Train the model with early stopping on the eval set."""
    eval_set = [(X_test, y_test)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=50,
        **fit_params,
    )

    return model


def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # MAPE (avoid division by zero)
    nonzero_mask = y_test != 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask])) * 100)
    else:
        mape = float("nan")

    # Feature importance
    importance = dict(zip(X_test.columns, model.feature_importances_.tolist()))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "n_test": len(y_test),
        "n_trees_used": int(model.best_iteration + 1) if hasattr(model, "best_iteration") and model.best_iteration is not None else model.n_estimators,
        "feature_importance": importance,
    }

    return metrics


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
    feature_cols: list[str],
) -> pd.DataFrame:
    """Add predicted and residual columns to a DataFrame."""
    df = df.copy()
    df["predicted"] = model.predict(df[feature_cols])
    df["residual"] = df["energy_per_sqft"] - df["predicted"]
    return df
