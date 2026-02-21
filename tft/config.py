"""
Configuration for Temporal Fusion Transformer energy consumption model.

Inherits composable pieces from src.config and adds TFT-specific parameters.
"""

from dataclasses import dataclass, field

from src.config import (
    MLBaseConfig,
    OutputDir,
    ConsoleLogging,
    Checkpointing,
    DataConfig,
    # Re-export helpers so tft/train.py can import from one place
    save_config,
    load_config,
    setup_output_dir,
    setup_console_logging,
)


@dataclass
class TFTDataConfig(DataConfig):
    """Extends DataConfig with TFT-specific temporal/normalization settings."""

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
class TFTParams:
    """Hyperparameters for the Temporal Fusion Transformer architecture."""

    # TFT core dimensions
    hidden_size: int = 64
    num_heads: int = 4
    num_lstm_layers: int = 1
    dropout: float = 0.1
    activation: str = "elu"  # "elu" or "relu"

    # Training
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # "cosine", "step", "none"
    scheduler_step_size: int = 15
    scheduler_gamma: float = 0.5
    early_stopping_patience: int = 10

    # TFT-specific
    max_grad_norm: float = 1.0  # gradient clipping


@dataclass
class EnergyTFTConfig(MLBaseConfig):
    """Top-level config for Temporal Fusion Transformer energy model."""

    name: str = "energy_tft"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.pt")
    )
    data: TFTDataConfig = field(default_factory=TFTDataConfig)
    tft: TFTParams = field(default_factory=TFTParams)
