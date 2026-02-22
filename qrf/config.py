"""
Configuration for Quantile Regression Forest energy consumption model.

Inherits composable pieces from src.config and adds QRF-specific parameters.
QRF provides prediction intervals via quantile estimation.
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
class QRFDataConfig(DataConfig):
    """Extends DataConfig with tabular feature engineering settings."""

    lag_hours: List[int] = field(default_factory=lambda: [1, 6, 24, 168])
    rolling_windows: List[int] = field(default_factory=lambda: [24, 168])
    add_interactions: bool = True


@dataclass
class QRFParams:
    """Hyperparameters for RandomForestQuantileRegressor."""

    n_estimators: int = 500
    max_depth: int = 20
    min_samples_split: int = 5
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    n_jobs: int = -1
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class EnergyQRFConfig(MLBaseConfig):
    """Top-level config for Quantile Regression Forest energy model."""

    name: str = "electricity_qrf"
    model_type: str = "qrf"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "plots": "plots", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.joblib")
    )
    data: QRFDataConfig = field(default_factory=QRFDataConfig)
    qrf: QRFParams = field(default_factory=QRFParams)
