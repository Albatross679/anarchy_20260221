"""
Configuration for DART (XGBoost with Dropout) energy consumption model.

Inherits composable pieces from src.config and adds DART-specific parameters.
DART extends XGBoost's gradient boosting with dropout regularization.
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
    save_config,
    load_config,
    setup_output_dir,
    setup_console_logging,
    get_system_metrics,
    log_system_metrics_to_tb,
)


@dataclass
class DARTDataConfig(DataConfig):
    """Extends DataConfig with tabular feature engineering settings."""

    lag_hours: List[int] = field(default_factory=lambda: [1, 6, 24, 168])
    rolling_windows: List[int] = field(default_factory=lambda: [24, 168])
    add_interactions: bool = True


@dataclass
class DARTParams:
    """Hyperparameters for XGBRegressor with DART booster."""

    n_estimators: int = 1000
    max_depth: int = 7
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    eval_metric: str = "rmse"

    # DART-specific parameters
    rate_drop: float = 0.1
    skip_drop: float = 0.5
    sample_type: str = "uniform"
    normalize_type: str = "tree"

    # Early stopping disabled by default (DART early stopping is unreliable)
    early_stopping_rounds: int = 0


@dataclass
class EnergyDARTConfig(MLBaseConfig):
    """Top-level config for DART energy model."""

    name: str = "electricity_dart"
    model_type: str = "dart"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "plots": "plots", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.json")
    )
    data: DARTDataConfig = field(default_factory=DARTDataConfig)
    dart: DARTParams = field(default_factory=DARTParams)
