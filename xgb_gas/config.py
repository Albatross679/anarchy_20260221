"""
Configuration for two-stage XGBoost gas model.

Stage 1: XGBClassifier (on/off detection)
Stage 2: XGBRegressor (magnitude for non-zero samples)

Reuses XGBoostDataConfig from xgb/config.py for identical feature engineering.
"""

from dataclasses import dataclass, field

from xgb.config import XGBoostDataConfig

from src.config import (
    MLBaseConfig,
    OutputDir,
    ConsoleLogging,
    Checkpointing,
    TensorBoardConfig,
    # Re-export helpers
    save_config,
    load_config,
    setup_output_dir,
    setup_console_logging,
    get_system_metrics,
    log_system_metrics_to_tb,
)


@dataclass
class ClassifierParams:
    """Hyperparameters for XGBClassifier (Stage 1: on/off detection)."""

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    scale_pos_weight: float = 1.0  # adjusted at runtime based on class imbalance
    tree_method: str = "hist"
    early_stopping_rounds: int = 50
    eval_metric: str = "logloss"


@dataclass
class RegressorParams:
    """Hyperparameters for XGBRegressor (Stage 2: magnitude prediction)."""

    n_estimators: int = 1000
    max_depth: int = 7
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"


@dataclass
class EnergyXGBoostGasConfig(MLBaseConfig):
    """Top-level config for two-stage XGBoost gas model."""

    name: str = "gas_xgboost_twostage"
    model_type: str = "xgboost_twostage"
    zero_threshold: float = 1e-5

    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "plots": "plots", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(default_factory=Checkpointing)
    data: XGBoostDataConfig = field(
        default_factory=lambda: XGBoostDataConfig(utility_filter="GAS")
    )
    classifier: ClassifierParams = field(default_factory=ClassifierParams)
    regressor: RegressorParams = field(default_factory=RegressorParams)
