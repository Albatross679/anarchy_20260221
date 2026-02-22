"""
Configuration for XGBoost energy consumption model.

Inherits composable pieces from src.config and adds XGBoost-specific parameters
with enhanced feature engineering settings for tabular models.
"""

from dataclasses import dataclass, field
from typing import List

from src.config import (
    MLBaseConfig,
    OutputDir,
    ConsoleLogging,
    Checkpointing,
    TensorBoardConfig,
    DataConfig,
    # Re-export helpers so xgb/train.py can import from one place
    save_config,
    load_config,
    setup_output_dir,
    setup_console_logging,
    get_system_metrics,
    log_system_metrics_to_tb,
)


@dataclass
class XGBoostDataConfig(DataConfig):
    """Extends DataConfig with tabular feature engineering settings."""

    # Lag features: hours to look back (converted to 15-min intervals internally)
    lag_hours: List[int] = field(default_factory=lambda: [1, 6, 24, 168])

    # Rolling window statistics: hours for rolling mean/std
    rolling_windows: List[int] = field(default_factory=lambda: [24, 168])

    # Interaction features: temp × area, humidity × area
    add_interactions: bool = True


@dataclass
class XGBoostParams:
    """Hyperparameters for XGBRegressor."""

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
class EnergyXGBoostConfig(MLBaseConfig):
    """Top-level config for XGBoost energy model."""

    name: str = "energy_xgboost"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "plots": "plots", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.json")
    )
    data: XGBoostDataConfig = field(default_factory=XGBoostDataConfig)
    xgb: XGBoostParams = field(default_factory=XGBoostParams)
