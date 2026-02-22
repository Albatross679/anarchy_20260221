"""
Configuration for CatBoost energy consumption model.

Inherits composable pieces from src.config and adds CatBoost-specific parameters
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
    save_config,
    load_config,
    setup_output_dir,
    setup_console_logging,
    get_system_metrics,
    log_system_metrics_to_tb,
)


@dataclass
class CatBoostDataConfig(DataConfig):
    """Extends DataConfig with tabular feature engineering settings."""

    lag_hours: List[int] = field(default_factory=lambda: [1, 6, 24, 168])
    rolling_windows: List[int] = field(default_factory=lambda: [24, 168])
    add_interactions: bool = True


@dataclass
class CatBoostParams:
    """Hyperparameters for CatBoostRegressor."""

    iterations: int = 1000
    depth: int = 7
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    random_strength: float = 1.0
    bagging_temperature: float = 1.0
    border_count: int = 254
    early_stopping_rounds: int = 50
    eval_metric: str = "RMSE"
    verbose: int = 50


@dataclass
class EnergyCatBoostConfig(MLBaseConfig):
    """Top-level config for CatBoost energy model."""

    name: str = "electricity_catboost"
    model_type: str = "catboost"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "plots": "plots", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.cbm")
    )
    data: CatBoostDataConfig = field(default_factory=CatBoostDataConfig)
    cb: CatBoostParams = field(default_factory=CatBoostParams)
