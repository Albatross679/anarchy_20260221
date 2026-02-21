# sl_config_template.py
"""
Supervised learning configuration.

Hierarchy:
    MLBaseConfig (base_config_template.py)
    +-- SLConfig  -- epochs, batch_size, lr, optimizer, scheduler, early stopping

Usage:
    from base_config_template import MLBaseConfig, Checkpointing, TensorBoard

    @dataclass
    class MyTrainingConfig(SLConfig):
        checkpointing: Checkpointing = field(default_factory=Checkpointing)
        tensorboard: TensorBoard = field(default_factory=TensorBoard)
        vocab_size: int = 10000
        num_classes: int = 10
"""

from dataclasses import dataclass
from typing import Optional

from base_config_template import MLBaseConfig


@dataclass
class SLConfig(MLBaseConfig):
    """Supervised learning configuration."""

    # Training
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    # Optimizer
    optimizer: str = "Adam"  # "Adam", "AdamW", "SGD"

    # Scheduler
    scheduler: Optional[str] = None  # "cosine", "linear", "step", None
    scheduler_min_lr: float = 1e-6

    # Regularization
    grad_clip_norm: Optional[float] = None
    dropout: float = 0.0

    # Early stopping
    early_stopping_patience: int = 0  # 0 = disabled
