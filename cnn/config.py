"""
Configuration for 1D CNN energy consumption model.

Inherits composable pieces from src.config and adds CNN-specific parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from src.config import (
    MLBaseConfig,
    OutputDir,
    ConsoleLogging,
    Checkpointing,
    TensorBoardConfig,
    DataConfig,
    # Re-export helpers so cnn/train.py can import from one place
    save_config,
    load_config,
    setup_output_dir,
    setup_console_logging,
    get_system_metrics,
    log_system_metrics_to_tb,
)


@dataclass
class CNNDataConfig(DataConfig):
    """Extends DataConfig with CNN-specific temporal/normalization settings."""

    # Sequence length: number of consecutive 15-min timesteps per sample
    # 96 = 24 hours, 24 = 6 hours
    seq_length: int = 96

    # Stride for sliding window (1 = every possible window)
    stride: int = 4

    # Feature normalization
    normalize_features: bool = True
    normalize_target: bool = True

    # DataLoader settings
    batch_size: int = 1024
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class CNNParams:
    """Hyperparameters for the 1D CNN architecture."""

    # Conv1D layers — list lengths define number of layers
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])
    pool_size: int = 2
    dropout_conv: float = 0.15

    # Fully connected head
    fc_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout_fc: float = 0.3

    # Activation
    activation: str = "gelu"

    # Training
    epochs: int = 80
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine", "step", "none"
    scheduler_step_size: int = 15
    scheduler_gamma: float = 0.5
    early_stopping_patience: int = 15

    # Architecture
    use_batch_norm: bool = True


@dataclass
class EnergyCNNConfig(MLBaseConfig):
    """Top-level config for CNN energy model. Mirrors EnergyModelConfig layout."""

    name: str = "energy_cnn"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.pt")
    )
    data: CNNDataConfig = field(default_factory=CNNDataConfig)
    cnn: CNNParams = field(default_factory=CNNParams)
