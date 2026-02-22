"""
Configuration for LSTM gas consumption model.

Inherits composable pieces from src.config and adds LSTM-specific parameters.
Defaults replicate seq_experiment.py LSTM seq48 result (R²=0.9723).
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

    utility_filter: str = "GAS"

    # Sequence length: number of consecutive 15-min timesteps per sample
    # 48 = 12 hours (best result from seq_experiment)
    seq_length: int = 48

    # Stride for sliding window (4 = one window per hour)
    stride: int = 4

    # Feature normalization
    normalize_features: bool = True
    normalize_target: bool = True

    # DataLoader settings
    batch_size: int = 512
    num_workers: int = 2
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

    # Static features (constant per building, fed to separate MLP branch)
    static_features: List[str] = field(
        default_factory=lambda: ["grossarea", "floorsaboveground", "building_age"]
    )


@dataclass
class LSTMParams:
    """Hyperparameters for the LSTM architecture.

    Defaults match the seq_experiment.py LSTM that achieved R²=0.9723.
    """

    # LSTM encoder
    hidden_size: int = 256
    num_layers: int = 3
    dropout_lstm: float = 0.3
    bidirectional: bool = False

    # Static MLP branch
    static_embedding_dim: int = 32
    static_hidden_dims: List[int] = field(default_factory=lambda: [64])
    dropout_static: float = 0.3

    # Fusion head (uses GELU activation)
    head_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout_head: float = 0.3

    # Training
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine", "step", "none"
    scheduler_step_size: int = 15
    scheduler_gamma: float = 0.5
    early_stopping_patience: int = 15

    # LSTM-specific
    max_grad_norm: float = 1.0  # gradient clipping


@dataclass
class EnergyLSTMConfig(MLBaseConfig):
    """Top-level config for LSTM gas energy model. Mirrors EnergyCNNConfig layout."""

    name: str = "gas_lstm"
    model_type: str = "lstm"
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
