"""
Configuration for Random Forest energy consumption model.

Inherits composable pieces from src.config and adds Random Forest-specific parameters
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
    # Re-export helpers so rf/train.py can import from one place
    save_config,
    load_config,
    setup_output_dir,
    setup_console_logging,
    get_system_metrics,
    log_system_metrics_to_tb,
)


@dataclass
class RFDataConfig(DataConfig):
    """Extends DataConfig with tabular feature engineering settings."""

    # Lag features: hours to look back (converted to 15-min intervals internally)
    lag_hours: List[int] = field(default_factory=lambda: [1, 6, 24, 168])

    # Rolling window statistics: hours for rolling mean/std
    rolling_windows: List[int] = field(default_factory=lambda: [24, 168])

    # Interaction features: temp x area, humidity x area
    add_interactions: bool = True


@dataclass
class RandomForestParams:
    """Hyperparameters for RandomForestRegressor."""

    n_estimators: int = 500
    max_depth: int = 20
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    n_jobs: int = -1


@dataclass
class EnergyRandomForestConfig(MLBaseConfig):
    """Top-level config for Random Forest energy model."""

    name: str = "energy_random_forest"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "plots": "plots", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.joblib")
    )
    data: RFDataConfig = field(default_factory=RFDataConfig)
    rf: RandomForestParams = field(default_factory=RandomForestParams)
