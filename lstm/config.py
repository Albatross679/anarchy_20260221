"""
Configuration for hybrid LSTM energy consumption model.

Inherits composable pieces from src.config and adds LSTM-specific parameters.
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
    # Re-export helpers so lstm/train.py can import from one place
    save_config,
    load_config,
    setup_output_dir,
    setup_console_logging,
    get_system_metrics,
    log_system_metrics_to_tb,
)


@dataclass
class LSTMDataConfig(DataConfig):
    """Extends DataConfig with LSTM-specific temporal/normalization settings."""

    # Sequence length: number of consecutive 15-min timesteps per sample
    seq_length: int = 96  # 24h of 15-min intervals

    # Stride for sliding window (4 = one window per hour)
    stride: int = 4

    # Feature normalization
    normalize_features: bool = True
    normalize_target: bool = True

    # DataLoader settings
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class LSTMParams:
    """Hyperparameters for the hybrid LSTM architecture."""

    # LSTM encoder
    hidden_size: int = 128
    num_layers: int = 2
    dropout_lstm: float = 0.2
    bidirectional: bool = False

    # Static MLP branch
    static_embedding_dim: int = 32
    static_hidden_dims: List[int] = field(default_factory=lambda: [64])
    dropout_static: float = 0.2

    # Fusion head
    head_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout_head: float = 0.3

    # Training
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # "cosine", "step", "none"
    scheduler_step_size: int = 15
    scheduler_gamma: float = 0.5
    early_stopping_patience: int = 10

    # LSTM-specific
    max_grad_norm: float = 1.0  # gradient clipping


@dataclass
class EnergyLSTMConfig(MLBaseConfig):
    """Top-level config for hybrid LSTM energy model."""

    name: str = "energy_lstm"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.pt")
    )
    data: LSTMDataConfig = field(default_factory=LSTMDataConfig)
    lstm: LSTMParams = field(default_factory=LSTMParams)
