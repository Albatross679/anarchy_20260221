"""
Configuration for Extra Trees energy consumption model.

Inherits composable pieces from src.config and adds Extra Trees-specific parameters
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
class ExtraTreesDataConfig(DataConfig):
    """Extends DataConfig with tabular feature engineering settings."""

    lag_hours: List[int] = field(default_factory=lambda: [1, 6, 24, 168])
    rolling_windows: List[int] = field(default_factory=lambda: [24, 168])
    add_interactions: bool = True


@dataclass
class ExtraTreesParams:
    """Hyperparameters for ExtraTreesRegressor."""

    n_estimators: int = 500
    max_depth: int = 20
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    n_jobs: int = -1


@dataclass
class EnergyExtraTreesConfig(MLBaseConfig):
    """Top-level config for Extra Trees energy model."""

    name: str = "electricity_extra_trees"
    model_type: str = "extra_trees"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "plots": "plots", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.joblib")
    )
    data: ExtraTreesDataConfig = field(default_factory=ExtraTreesDataConfig)
    et: ExtraTreesParams = field(default_factory=ExtraTreesParams)
