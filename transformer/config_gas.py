"""
Configuration for Transformer gas consumption model.

Mirrors transformer/config.py but targets GAS utility with pre-engineered
parquet features. Training settings aligned with the LSTM gas config
(seq_length=48, stride=4, batch_size=512, patience=15, epochs=100).
Transformer architecture scaled to match seq_experiment.py transformer variant
(d_model=128, n_heads=4, n_layers=4, d_ff=256).
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
class TransformerGasDataConfig(DataConfig):
    """Extends DataConfig with Transformer gas-specific settings.

    Training parameters aligned with LSTM gas config for fair comparison.
    """

    utility_filter: str = "GAS"

    # Sequence length: 48 = 12 hours (matches LSTM gas)
    seq_length: int = 48

    # Stride for sliding window (matches LSTM gas)
    stride: int = 4

    # Feature normalization
    normalize_features: bool = True
    normalize_target: bool = True

    # DataLoader settings (matches LSTM gas)
    batch_size: int = 512
    num_workers: int = 2
    pin_memory: bool = True

    # Pre-built parquet with engineered features
    parquet_path: str = "data/tree_features_gas_cross.parquet"

    # Sparse cross-utility prefixes to drop
    sparse_prefixes: List[str] = field(
        default_factory=lambda: ["heat_", "steam_", "cooling_"]
    )

    # Buildings with >99.9% zero readings are separated as always-off
    always_off_threshold: float = 0.999
    zero_threshold: float = 1e-5


@dataclass
class TransformerGasParams:
    """Hyperparameters for the Transformer gas architecture.

    Architecture scaled up from electricity transformer (d_model 64→128,
    n_layers 3→4, d_ff 128→256) to match seq_experiment.py transformer.
    Training settings aligned with LSTM gas.
    """

    # Transformer encoder dimensions
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.2

    # Fully connected head
    fc_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout_fc: float = 0.3

    # Activation
    activation: str = "gelu"

    # Training (aligned with LSTM gas)
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    scheduler_step_size: int = 15
    scheduler_gamma: float = 0.5
    early_stopping_patience: int = 15

    # Positional encoding
    use_positional_encoding: bool = True


@dataclass
class EnergyTransformerGasConfig(MLBaseConfig):
    """Top-level config for Transformer gas energy model."""

    name: str = "gas_transformer"
    model_type: str = "transformer"
    output: OutputDir = field(
        default_factory=lambda: OutputDir(
            subdirs={"checkpoints": "checkpoints", "tensorboard": "tensorboard"}
        )
    )
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(
        default_factory=lambda: Checkpointing(best_filename="model_best.pt")
    )
    data: TransformerGasDataConfig = field(default_factory=TransformerGasDataConfig)
    transformer: TransformerGasParams = field(default_factory=TransformerGasParams)
