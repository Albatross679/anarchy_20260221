"""
Configuration for Transformer energy consumption model.

Inherits composable pieces from src.config and adds Transformer-specific parameters.
"""

from dataclasses import dataclass, field
from typing import List

from src.config import (
    MLBaseConfig,
    OutputDir,
    ConsoleLogging,
    Checkpointing,
    DataConfig,
    # Re-export helpers so transformer/train.py can import from one place
    save_config,
    load_config,
    setup_output_dir,
    setup_console_logging,
)


@dataclass
class TransformerDataConfig(DataConfig):
    """Extends DataConfig with Transformer-specific temporal/normalization settings."""

    # Sequence length: number of consecutive hourly timesteps per sample
    seq_length: int = 24

    # Stride for sliding window (1 = every possible window)
    stride: int = 1

    # Feature normalization
    normalize_features: bool = True
    normalize_target: bool = True

    # DataLoader settings
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TransformerParams:
    """Hyperparameters for the Transformer encoder architecture."""

    # Transformer encoder dimensions
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.1

    # Fully connected head
    fc_dims: List[int] = field(default_factory=lambda: [64])
    dropout_fc: float = 0.3

    # Activation
    activation: str = "gelu"

    # Training
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # "cosine", "step", "none"
    scheduler_step_size: int = 15
    scheduler_gamma: float = 0.5
    early_stopping_patience: int = 10

    # Positional encoding
    use_positional_encoding: bool = True


@dataclass
class EnergyTransformerConfig(MLBaseConfig):
    """Top-level config for Transformer energy model."""

    name: str = "energy_transformer"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.pt")
    )
    data: TransformerDataConfig = field(default_factory=TransformerDataConfig)
    transformer: TransformerParams = field(default_factory=TransformerParams)
