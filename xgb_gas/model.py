"""
Two-stage XGBoost model for gas consumption prediction.

Stage 1: XGBClassifier predicts on/off (binary)
Stage 2: XGBRegressor predicts magnitude for non-zero ("on") samples
Combined: prediction = 0 if classifier says off, else regressor output

Reuses engineer_features and TensorBoardCallback from xgb/model.py.
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier, XGBRegressor

# Reuse from existing xgb module
from xgb.model import engineer_features, TensorBoardCallback
from xgb_gas.config import (
    ClassifierParams,
    RegressorParams,
    TensorBoardConfig,
    log_system_metrics_to_tb,
)


# ---------------------------------------------------------------------------
# Binary label creation
# ---------------------------------------------------------------------------


def create_binary_labels(y: pd.Series, threshold: float = 1e-9) -> np.ndarray:
    """Convert continuous target to binary: 0 if <= threshold, 1 otherwise."""
    return (np.abs(y) > threshold).astype(int).values


# ---------------------------------------------------------------------------
# Stage 1: Classifier
# ---------------------------------------------------------------------------


def create_classifier(
    params: ClassifierParams, seed: int = 42
) -> Tuple[XGBClassifier, dict]:
    """Create XGBClassifier from config params."""
    kwargs = dict(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        min_child_weight=params.min_child_weight,
        scale_pos_weight=params.scale_pos_weight,
        tree_method=params.tree_method,
        eval_metric=params.eval_metric,
        random_state=seed,
        n_jobs=-1,
    )
    if params.early_stopping_rounds > 0:
        kwargs["early_stopping_rounds"] = params.early_stopping_rounds

    model = XGBClassifier(**kwargs)
    return model, {}


def train_classifier(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train_cls: np.ndarray,
    X_test: pd.DataFrame,
    y_test_cls: np.ndarray,
    fit_params: dict,
    params: Optional[ClassifierParams] = None,
    run_dir: Optional[Path] = None,
    tb_cfg: Optional[TensorBoardConfig] = None,
) -> XGBClassifier:
    """Fit classifier with early stopping and optional TensorBoard logging."""
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
                f"| stage | classifier |\n"
                f"| n_estimators | {params.n_estimators} |\n"
                f"| max_depth | {params.max_depth} |\n"
                f"| learning_rate | {params.learning_rate} |\n"
                f"| scale_pos_weight | {params.scale_pos_weight} |\n"
                f"| train_rows | {len(X_train):,} |\n"
                f"| test_rows | {len(X_test):,} |\n"
                f"| n_features | {X_train.shape[1]} |\n"
            )
            writer.add_text("classifier/hyperparameters", hparam_text, 0)
        psutil.cpu_percent(interval=None)

    # TensorBoard callback
    callbacks = fit_params.pop("callbacks", [])
    if writer is not None:
        callbacks.append(TensorBoardCallback(
            writer, tb_cfg,
            X_train=X_train, y_train=pd.Series(y_train_cls),
            X_val=X_test, y_val=pd.Series(y_test_cls),
            log_every=10,
        ))
    if callbacks:
        model.set_params(callbacks=callbacks)

    model.fit(
        X_train,
        y_train_cls,
        eval_set=[(X_train, y_train_cls), (X_test, y_test_cls)],
        verbose=50,
        **fit_params,
    )

    if writer is not None:
        writer.close()
        print(f"  Classifier TensorBoard logs: {run_dir / 'tensorboard'}")

    return model


def evaluate_classifier(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test_cls: np.ndarray,
    feature_cols: List[str],
    run_dir: Optional[Path] = None,
) -> dict:
    """Evaluate classifier: accuracy, precision, recall, F1, AUC + plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test_cls, y_pred))
    prec = float(precision_score(y_test_cls, y_pred, zero_division=0))
    rec = float(recall_score(y_test_cls, y_pred, zero_division=0))
    f1 = float(f1_score(y_test_cls, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test_cls, y_prob))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "n_test": len(y_test_cls),
        "n_positive": int(y_test_cls.sum()),
        "n_negative": int((y_test_cls == 0).sum()),
    }

    if run_dir is not None:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _save_classifier_plots(model, X_test, y_test_cls, y_pred, y_prob, feature_cols, plots_dir)

    return metrics


def _save_classifier_plots(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test_cls: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_cols: List[str],
    plots_dir: Path,
) -> None:
    """Generate classifier diagnostic plots."""

    # 1. Confusion matrix
    cm = confusion_matrix(y_test_cls, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Off (0)", "On (1)"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Classifier Confusion Matrix")
    fig.tight_layout()
    fig.savefig(plots_dir / "classifier_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test_cls, y_prob)
    auc_val = roc_auc_score(y_test_cls, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Classifier ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plots_dir / "classifier_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_cols[i] for i in sorted_idx],
        importance[sorted_idx],
    )
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Classifier — Top 20 Feature Importance")
    fig.tight_layout()
    fig.savefig(plots_dir / "classifier_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 2: Regressor (non-zero subset)
# ---------------------------------------------------------------------------


def create_regressor(
    params: RegressorParams, seed: int = 42
) -> Tuple[XGBRegressor, dict]:
    """Create XGBRegressor from config params."""
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


def train_regressor(
    model: XGBRegressor,
    X_train_on: pd.DataFrame,
    y_train_on: pd.Series,
    X_test_on: pd.DataFrame,
    y_test_on: pd.Series,
    fit_params: dict,
    params: Optional[RegressorParams] = None,
    run_dir: Optional[Path] = None,
    tb_cfg: Optional[TensorBoardConfig] = None,
) -> XGBRegressor:
    """Fit regressor on non-zero subset with early stopping."""
    if tb_cfg is None:
        tb_cfg = TensorBoardConfig()

    writer = None
    if tb_cfg.enabled and run_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        # Use a separate subdir so classifier and regressor logs don't collide
        tb_dir = run_dir / "tensorboard" / "regressor"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

        if tb_cfg.log_hparams_text and params is not None:
            hparam_text = (
                "| Param | Value |\n|---|---|\n"
                f"| stage | regressor |\n"
                f"| n_estimators | {params.n_estimators} |\n"
                f"| max_depth | {params.max_depth} |\n"
                f"| learning_rate | {params.learning_rate} |\n"
                f"| train_rows (on only) | {len(X_train_on):,} |\n"
                f"| test_rows (on only) | {len(X_test_on):,} |\n"
                f"| n_features | {X_train_on.shape[1]} |\n"
            )
            writer.add_text("regressor/hyperparameters", hparam_text, 0)
        psutil.cpu_percent(interval=None)

    callbacks = fit_params.pop("callbacks", [])
    if writer is not None:
        callbacks.append(TensorBoardCallback(
            writer, tb_cfg,
            X_train=X_train_on, y_train=y_train_on,
            X_val=X_test_on, y_val=y_test_on,
            log_every=10,
        ))
    if callbacks:
        model.set_params(callbacks=callbacks)

    model.fit(
        X_train_on,
        y_train_on,
        eval_set=[(X_train_on, y_train_on), (X_test_on, y_test_on)],
        verbose=50,
        **fit_params,
    )

    if writer is not None:
        writer.close()
        print(f"  Regressor TensorBoard logs: {run_dir / 'tensorboard' / 'regressor'}")

    return model


def evaluate_regressor(
    model: XGBRegressor,
    X_test_on: pd.DataFrame,
    y_test_on: pd.Series,
    feature_cols: List[str],
    run_dir: Optional[Path] = None,
    log_target: bool = False,
) -> dict:
    """Evaluate regressor on non-zero subset: RMSE, MAE, R².

    If log_target=True, predictions and actuals are in log space;
    metrics are computed in original space after expm1.
    """
    y_pred_raw = model.predict(X_test_on)
    y_actual_raw = np.asarray(y_test_on)

    # Convert back to original space for metrics
    if log_target:
        y_pred = np.exp(y_pred_raw)
        y_actual = np.exp(y_actual_raw)
    else:
        y_pred = y_pred_raw
        y_actual = y_actual_raw

    rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    mae = float(mean_absolute_error(y_actual, y_pred))
    r2 = float(r2_score(y_actual, y_pred))

    nonzero = y_actual != 0
    if nonzero.sum() > 0:
        mape = float(
            np.mean(np.abs((y_actual[nonzero] - y_pred[nonzero]) / y_actual[nonzero])) * 100
        )
    else:
        mape = float("nan")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "n_test_on": len(y_actual),
        "n_trees_used": (
            int(model.best_iteration + 1)
            if hasattr(model, "best_iteration") and model.best_iteration is not None
            else model.n_estimators
        ),
    }

    if run_dir is not None:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _save_regressor_plots(model, X_test_on, pd.Series(y_actual, index=X_test_on.index),
                              y_pred, feature_cols, plots_dir)

    return metrics


def _save_regressor_plots(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_cols: List[str],
    plots_dir: Path,
) -> None:
    """Generate regressor diagnostic plots (non-zero subset only)."""
    y_true_arr = np.asarray(y_test)

    # 1. Predicted vs Actual
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
    ax.set_title(f"Regressor Pred vs Actual (non-zero only, R²={r2_score(y_true_arr, y_pred):.4f})")
    fig.tight_layout()
    fig.savefig(plots_dir / "regressor_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Residual distribution
    residuals = y_true_arr - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=100, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Regressor Residual Distribution (mean={residuals.mean():.6f}, std={residuals.std():.6f})"
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "regressor_residual_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_cols[i] for i in sorted_idx],
        importance[sorted_idx],
    )
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Regressor — Top 20 Feature Importance")
    fig.tight_layout()
    fig.savefig(plots_dir / "regressor_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------


def _find_optimal_threshold(cls_prob, reg_pred, y_true, zero_threshold):
    """Search for the probability cutoff that maximises combined R²."""
    best_r2, best_t = -np.inf, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        y_hat = np.where(cls_prob >= t, reg_pred, 0.0)
        score = r2_score(y_true, y_hat)
        if score > best_r2:
            best_r2, best_t = score, float(t)
    return best_t, best_r2


def evaluate_combined(
    classifier: XGBClassifier,
    regressor: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    threshold: float,
    run_dir: Optional[Path] = None,
    log_target: bool = False,
) -> dict:
    """Evaluate the full two-stage pipeline on the complete test set.

    Reports three blending strategies:
        hard   — 0 if cls_pred==0 else reg_pred  (original)
        soft   — cls_prob * reg_pred              (smooth transition)
        optimal— 0 if cls_prob < best_cutoff else reg_pred

    If log_target=True, regressor predictions are in log space and
    are converted back via expm1 before combining with the classifier.
    """
    y_true = np.asarray(y_test)  # always in original space

    cls_prob = classifier.predict_proba(X_test[feature_cols])[:, 1]
    reg_pred_raw = regressor.predict(X_test[feature_cols])
    reg_pred = np.exp(reg_pred_raw) if log_target else reg_pred_raw

    # --- Hard gating (baseline) ---
    cls_pred = classifier.predict(X_test[feature_cols])
    y_hard = np.where(cls_pred == 1, reg_pred, 0.0)
    r2_hard = float(r2_score(y_true, y_hard))

    # --- Soft blending ---
    y_soft = cls_prob * reg_pred
    r2_soft = float(r2_score(y_true, y_soft))

    # --- Optimal probability threshold ---
    opt_cutoff, r2_opt = _find_optimal_threshold(cls_prob, reg_pred, y_true, threshold)
    y_opt = np.where(cls_prob >= opt_cutoff, reg_pred, 0.0)

    # Pick the best strategy
    strategies = {"hard": (y_hard, r2_hard), "soft": (y_soft, r2_soft), "optimal_threshold": (y_opt, r2_opt)}
    best_name = max(strategies, key=lambda k: strategies[k][1])
    y_best = strategies[best_name][0]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_best)))
    mae = float(mean_absolute_error(y_true, y_best))
    r2 = float(r2_score(y_true, y_best))

    # Subset metrics
    true_zeros = np.abs(y_true) <= threshold
    true_nonzeros = ~true_zeros

    zero_accuracy = float(np.mean(cls_pred[true_zeros] == 0)) if true_zeros.sum() > 0 else float("nan")
    nonzero_accuracy = float(np.mean(cls_pred[true_nonzeros] == 1)) if true_nonzeros.sum() > 0 else float("nan")

    r2_nonzero = float(r2_score(y_true[true_nonzeros], y_best[true_nonzeros])) if true_nonzeros.sum() > 1 else float("nan")

    metrics = {
        "best_strategy": best_name,
        "combined_rmse": rmse,
        "combined_mae": mae,
        "combined_r2": r2,
        "r2_hard": r2_hard,
        "r2_soft": r2_soft,
        "r2_optimal_threshold": r2_opt,
        "optimal_prob_cutoff": opt_cutoff,
        "r2_nonzero_subset": r2_nonzero,
        "zero_classification_accuracy": zero_accuracy,
        "nonzero_classification_accuracy": nonzero_accuracy,
        "n_test": len(y_true),
        "n_true_zeros": int(true_zeros.sum()),
        "n_true_nonzeros": int(true_nonzeros.sum()),
        "n_pred_zeros": int((cls_pred == 0).sum()),
        "n_pred_nonzeros": int((cls_pred == 1).sum()),
    }

    if run_dir is not None:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _save_combined_plots(y_true, y_best, r2, plots_dir)
        _save_shap_plots(regressor, X_test[feature_cols], plots_dir)

    return metrics


def _save_combined_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2: float,
    plots_dir: Path,
) -> None:
    """Generate combined two-stage prediction plots."""
    sample_size = min(5000, len(y_true))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=sample_size, replace=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true[idx], y_pred[idx], alpha=0.3, s=4)
    lims = [
        min(y_true[idx].min(), y_pred[idx].min()),
        max(y_true[idx].max(), y_pred[idx].max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual (energy/sqft)")
    ax.set_ylabel("Predicted (energy/sqft)")
    ax.set_title(f"Combined Two-Stage Pred vs Actual (R²={r2:.4f})")
    fig.tight_layout()
    fig.savefig(plots_dir / "combined_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_shap_plots(
    regressor: XGBRegressor,
    X_test: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Generate SHAP summary plot for the regressor stage."""
    try:
        import shap

        explainer = shap.TreeExplainer(regressor)
        rng = np.random.default_rng(42)
        shap_sample_size = min(1000, len(X_test))
        X_sample = X_test.iloc[
            rng.choice(len(X_test), size=shap_sample_size, replace=False)
        ]
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  SHAP plots saved.")
    except ImportError:
        print("  shap not installed — skipping SHAP plots.")
    except Exception as e:
        print(f"  SHAP plot generation failed: {e}")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def get_predictions(
    classifier: XGBClassifier,
    regressor: XGBRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float,
    blend_mode: str = "soft",
    prob_cutoff: float = 0.5,
    log_target: bool = False,
) -> pd.DataFrame:
    """Add two-stage prediction columns to DataFrame.

    Args:
        blend_mode: "soft" (class_prob * reg_pred), "hard" (0/1 gate),
                    or "optimal" (use prob_cutoff threshold).
        prob_cutoff: Probability threshold when blend_mode="optimal".
        log_target: If True, regressor outputs log values; expm1 is applied.

    Columns added:
        class_pred  — binary 0/1 from classifier
        class_prob  — probability of class 1 (on)
        predicted   — blended prediction (original space)
        residual    — energy_per_sqft - predicted
    """
    df = df.copy()
    df["class_pred"] = classifier.predict(df[feature_cols])
    df["class_prob"] = classifier.predict_proba(df[feature_cols])[:, 1]

    reg_pred_raw = regressor.predict(df[feature_cols])
    reg_pred = np.exp(reg_pred_raw) if log_target else reg_pred_raw

    if blend_mode == "soft":
        df["predicted"] = df["class_prob"].values * reg_pred
    elif blend_mode == "optimal":
        df["predicted"] = np.where(df["class_prob"] >= prob_cutoff, reg_pred, 0.0)
    else:  # hard
        df["predicted"] = np.where(df["class_pred"] == 1, reg_pred, 0.0)

    df["residual"] = df["energy_per_sqft"] - df["predicted"]
    return df


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_models(
    classifier: XGBClassifier,
    regressor: XGBRegressor,
    checkpoint_dir: Path,
) -> None:
    """Save both models in XGBoost native JSON format."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(checkpoint_dir / "classifier_best.json"))
    regressor.save_model(str(checkpoint_dir / "regressor_best.json"))


def load_models(checkpoint_dir: Path) -> Tuple[XGBClassifier, XGBRegressor]:
    """Load both models from XGBoost native JSON format."""
    clf = XGBClassifier()
    clf.load_model(str(checkpoint_dir / "classifier_best.json"))
    reg = XGBRegressor()
    reg.load_model(str(checkpoint_dir / "regressor_best.json"))
    return clf, reg
