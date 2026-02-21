---
name: ml-config-system
description: |
  A dataclass-based hierarchical configuration system for ML experiments.
  Plain dataclasses with single inheritance and composable pieces.
  Use when: (1) Setting up experiment configurations using Python dataclasses,
  (2) Creating configs for SL, RL, or PPO training pipelines,
  (3) Extending the hierarchy for new tasks or algorithms.
---

# ML Configuration System

Plain dataclass hierarchy for ML experiments. No mixins, no magic — just inheritance and composition.

## Hierarchy

```
MLBaseConfig                    (name, seed, device, output_dir)
├── SLConfig                    (epochs, batch_size, lr, optimizer, scheduler, early stopping)
├── RLConfig                    (timesteps, gamma, num_envs, normalization)
│   └── PPOConfig               (clip_epsilon, gae_lambda, entropy_coef, value_coef)
└── (your task config inherits from any of these)
```

## Composable Pieces

Standalone dataclasses attached via fields — not part of the hierarchy:

| Piece | Fields | Purpose |
|-------|--------|---------|
| `OutputDir` | `base_dir`, `save_config`, `timestamp_format`, `subdirs` | Timestamped run directory |
| `ConsoleLogging` | `enabled`, `filename`, `tee_to_console`, `separate_streams` | Console output capture |
| `Checkpointing` | `enabled`, `save_best`, `save_last`, `save_frequency`, `metric`, `mode`, filenames | Model saving |
| `TensorBoard` | `enabled`, `log_dir`, `flush_secs`, `log_interval` | Metric logging |

Add more composable pieces as needed (e.g., `WandbConfig`, `EvalConfig`).

## Default Output Directory Structure

Every experiment creates a timestamped run directory:

```
{output.base_dir}/{config.name}_{YYYYMMDD_HHMMSS}/
├── config.json           # full config snapshot
├── console.log           # captured stdout/stderr
├── checkpoints/          # model weights
│   ├── model_best.pt
│   └── model_last.pt
└── tensorboard/          # tfevents files
    └── events.out.tfevents...
```

Created automatically by `setup_output_dir(cfg)`.

## Files

```
ml-config-system/
    SKILL.md                    -- This file (overview + field reference)
    base_config_template.py     -- MLBaseConfig, composable pieces, helpers
    sl_config_template.py       -- SLConfig (supervised learning)
    rl_config_template.py       -- RLConfig, PPOConfig (reinforcement learning)
```

## How to Create a Task-Specific Config

1. Pick a parent class (`SLConfig`, `RLConfig`, `PPOConfig`, or `MLBaseConfig`)
2. Inherit from it
3. Add composable pieces as fields
4. Add task-specific fields
5. Override defaults as needed

```python
from dataclasses import dataclass, field

@dataclass
class MyTaskConfig(PPOConfig):
    # Override parent defaults
    name: str = "my_task"
    total_timesteps: int = 2_000_000
    num_envs: int = 8

    # Attach composable pieces (always include output + console)
    output: OutputDir = field(default_factory=OutputDir)
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(default_factory=Checkpointing)
    tensorboard: TensorBoard = field(default_factory=TensorBoard)

    # Task-specific fields
    reward_scale: float = 1.0
    use_curriculum: bool = False
```

## Setting Up a Run

```python
# Create config
cfg = MyTaskConfig(name="ppo_snake_v2")

# Set up output directory (creates timestamped dir + subdirs + saves config.json)
run_dir = setup_output_dir(cfg)
# -> output/ppo_snake_v2_20260221_143000/

# Set up console logging (captures stdout/stderr to console.log)
cleanup = setup_console_logging(cfg, run_dir)

# ... train ...

# Restore original stdout/stderr
cleanup()
```

## Saving and Loading

```python
from dataclasses import asdict

# Save (also done automatically by setup_output_dir)
save_config(cfg, "output/config.json")

# Load
cfg = load_config(MyTaskConfig, "output/config.json")

# Manual serialization
d = asdict(cfg)  # standard dataclasses.asdict
```

## Field Reference

### MLBaseConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"experiment"` | Experiment name (used in run directory) |
| `seed` | `int` | `42` | Random seed |
| `device` | `str` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"` |
| `output_dir` | `str` | `"output"` | Output directory (fallback if no `OutputDir` piece) |

### OutputDir (composable)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_dir` | `str` | `"output"` | Parent directory for all runs |
| `save_config` | `bool` | `True` | Save config.json to run directory |
| `timestamp_format` | `str` | `"%Y%m%d_%H%M%S"` | Timestamp format for directory naming |
| `subdirs` | `Dict[str, str]` | `{"tensorboard": "tensorboard", "checkpoints": "checkpoints"}` | Subdirectories to create |

### ConsoleLogging (composable)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable console capture |
| `filename` | `str` | `"console.log"` | Log file name in run directory |
| `separate_streams` | `bool` | `False` | Split stdout/stderr into separate files |
| `stdout_filename` | `str` | `"stdout.log"` | Stdout file (when `separate_streams=True`) |
| `stderr_filename` | `str` | `"stderr.log"` | Stderr file (when `separate_streams=True`) |
| `tee_to_console` | `bool` | `True` | Also print to terminal |
| `line_timestamps` | `bool` | `False` | Prefix each line with timestamp |
| `timestamp_format` | `str` | `"%H:%M:%S"` | Timestamp format for line prefixes |
| `flush_frequency` | `int` | `1` | Flush every N writes |

### Checkpointing (composable)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable checkpointing |
| `save_best` | `bool` | `True` | Save best model |
| `save_last` | `bool` | `True` | Save last model |
| `save_frequency` | `int` | `0` | Save every N epochs; `0` = disabled |
| `metric` | `str` | `"loss"` | Metric to track for best model |
| `mode` | `str` | `"min"` | `"min"` or `"max"` |
| `best_filename` | `str` | `"model_best.pt"` | Best model filename |
| `last_filename` | `str` | `"model_last.pt"` | Last model filename |
| `epoch_filename_format` | `str` | `"model_epoch_{epoch}.pt"` | Periodic save filename |

### TensorBoard (composable)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable TensorBoard logging |
| `log_dir` | `str` | `"tensorboard"` | Log directory (relative to run dir) |
| `flush_secs` | `int` | `120` | Flush interval |
| `log_interval` | `int` | `100` | Steps between log writes |

### SLConfig (extends MLBaseConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_epochs` | `int` | `100` | Training epochs |
| `batch_size` | `int` | `32` | Batch size |
| `learning_rate` | `float` | `1e-3` | Learning rate |
| `weight_decay` | `float` | `0.0` | L2 regularization |
| `optimizer` | `str` | `"Adam"` | `"Adam"`, `"AdamW"`, `"SGD"` |
| `scheduler` | `Optional[str]` | `None` | `"cosine"`, `"linear"`, `"step"`, `None` |
| `scheduler_min_lr` | `float` | `1e-6` | Minimum LR for scheduler |
| `grad_clip_norm` | `Optional[float]` | `None` | Max gradient norm; `None` = disabled |
| `dropout` | `float` | `0.0` | Dropout rate |
| `early_stopping_patience` | `int` | `0` | Epochs without improvement; `0` = disabled |

### RLConfig (extends MLBaseConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total_timesteps` | `int` | `1_000_000` | Total training timesteps |
| `gamma` | `float` | `0.99` | Discount factor |
| `learning_rate` | `float` | `3e-4` | Learning rate |
| `num_envs` | `int` | `1` | Parallel environments |
| `normalize_obs` | `bool` | `False` | Normalize observations |
| `normalize_reward` | `bool` | `False` | Normalize rewards |

### PPOConfig (extends RLConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `frames_per_batch` | `int` | `2048` | Frames per rollout batch |
| `num_epochs` | `int` | `10` | PPO epochs per batch |
| `mini_batch_size` | `int` | `64` | Mini-batch size |
| `clip_epsilon` | `float` | `0.2` | PPO clipping range |
| `gae_lambda` | `float` | `0.95` | GAE lambda |
| `normalize_advantage` | `bool` | `True` | Normalize advantages |
| `value_coef` | `float` | `0.5` | Value loss coefficient |
| `entropy_coef` | `float` | `0.01` | Entropy bonus coefficient |
| `max_grad_norm` | `float` | `0.5` | Gradient clipping norm |
| `target_kl` | `Optional[float]` | `None` | KL early stopping; `None` = disabled |

## Extending the Hierarchy

To add a new algorithm (e.g., SAC):

```python
@dataclass
class SACConfig(RLConfig):
    tau: float = 0.005              # soft update coefficient
    alpha: float = 0.2              # entropy temperature
    auto_alpha: bool = True         # auto-tune alpha
    buffer_size: int = 1_000_000    # replay buffer size
    batch_size: int = 256
    learning_rate: float = 3e-4
    num_epochs: int = 1             # gradient steps per env step
```

To add a new composable piece:

```python
@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "my-project"
    entity: Optional[str] = None
    log_interval: int = 100
```
