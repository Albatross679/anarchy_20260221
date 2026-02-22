"""
Configuration for 1D CNN gas consumption model.

Mirrors cnn/config.py but targets GAS utility with pre-engineered parquet features.
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
class CNNGasDataConfig(DataConfig):
    """Extends DataConfig with CNN gas-specific settings."""

    utility_filter: str = "GAS"

    # Sequence length: 48 = 12 hours (matches LSTM gas best result)
    seq_length: int = 48

    # Stride for sliding window
    stride: int = 4

    # Feature normalization
    normalize_features: bool = True
    normalize_target: bool = True

    # DataLoader settings
    batch_size: int = 1024
    num_workers: int = 4
    pin_memory: bool = True

    # Pre-built parquet with engineered features
    parquet_path: str = "data/tree_features_gas_cross.parquet"

    # Sparse cross-utility prefixes to drop from parquet features
    sparse_prefixes: List[str] = field(
        default_factory=lambda: ["heat_", "steam_", "cooling_"]
    )

    # Buildings with >99.9% zero readings are separated as always-off
    always_off_threshold: float = 0.999
    zero_threshold: float = 1e-5


@dataclass
class CNNGasParams:
    """Hyperparameters for the 1D CNN gas architecture.

    Same architecture as electricity CNN (cnn/config.py).
    """

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
class EnergyCNNGasConfig(MLBaseConfig):
    """Top-level config for CNN gas energy model."""

    name: str = "gas_cnn"
    model_type: str = "cnn"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.pt")
    )
    data: CNNGasDataConfig = field(default_factory=CNNGasDataConfig)
    cnn: CNNGasParams = field(default_factory=CNNGasParams)
