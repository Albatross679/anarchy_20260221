"""
Configuration for NGBoost energy consumption model.

Inherits composable pieces from src.config and adds NGBoost-specific parameters.
NGBoost provides probabilistic predictions with uncertainty estimates.
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
class NGBoostDataConfig(DataConfig):
    """Extends DataConfig with tabular feature engineering settings."""

    lag_hours: List[int] = field(default_factory=lambda: [1, 6, 24, 168])
    rolling_windows: List[int] = field(default_factory=lambda: [24, 168])
    add_interactions: bool = True


@dataclass
class NGBoostParams:
    """Hyperparameters for NGBRegressor."""

    n_estimators: int = 500
    learning_rate: float = 0.01
    minibatch_frac: float = 0.8
    base_max_depth: int = 4
    base_min_samples_leaf: int = 5
    early_stopping_rounds: int = 50


@dataclass
class EnergyNGBoostConfig(MLBaseConfig):
    """Top-level config for NGBoost energy model."""

    name: str = "electricity_ngboost"
    model_type: str = "ngboost"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "plots": "plots", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.joblib")
    )
    data: NGBoostDataConfig = field(default_factory=NGBoostDataConfig)
    ngb: NGBoostParams = field(default_factory=NGBoostParams)
