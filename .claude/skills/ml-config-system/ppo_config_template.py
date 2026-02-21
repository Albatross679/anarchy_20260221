# ppo_config_template.py
"""
PPO configuration.

Hierarchy:
    MLBaseConfig (base_config_template.py)
    +-- RLConfig (rl_config_template.py)
        +-- PPOConfig  -- clip_epsilon, gae_lambda, entropy_coef, value_coef

Usage:
    from ppo_config_template import PPOConfig
    from base_config_template import Checkpointing, TensorBoard

    @dataclass
    class MyPPOConfig(PPOConfig):
        checkpointing: Checkpointing = field(default_factory=Checkpointing)
        tensorboard: TensorBoard = field(default_factory=TensorBoard)
        reward_scale: float = 1.0
"""

from dataclasses import dataclass
from typing import Optional

from rl_config_template import RLConfig


@dataclass
class PPOConfig(RLConfig):
    """PPO-specific configuration."""

    # Rollout
    frames_per_batch: int = 2048

    # PPO objective
    num_epochs: int = 10
    mini_batch_size: int = 64
    clip_epsilon: float = 0.2

    # Advantage estimation
    gae_lambda: float = 0.95
    normalize_advantage: bool = True

    # Loss coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Gradient
    max_grad_norm: float = 0.5

    # Early stopping
    target_kl: Optional[float] = None
