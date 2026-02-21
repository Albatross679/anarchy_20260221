# rl_config_template.py
"""
Base reinforcement learning configuration.

Hierarchy:
    MLBaseConfig (base_config_template.py)
    +-- RLConfig  -- timesteps, gamma, num_envs, normalization

Usage:
    from rl_config_template import RLConfig

    @dataclass
    class MyRLConfig(RLConfig):
        reward_scale: float = 1.0
"""

from dataclasses import dataclass

from base_config_template import MLBaseConfig


@dataclass
class RLConfig(MLBaseConfig):
    """Base RL configuration. Extend for specific algorithms."""

    total_timesteps: int = 1_000_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    num_envs: int = 1
    normalize_obs: bool = False
    normalize_reward: bool = False
