# base_config_template.py
"""
Base ML configuration and shared utilities.

Provides:
    MLBaseConfig          -- root config (name, seed, device, output_dir)
    OutputDir             -- composable: timestamped run directory with subdirs
    ConsoleLogging        -- composable: console output capture
    Checkpointing         -- composable: model saving
    TensorBoard           -- composable: metric logging
    save_config()         -- serialize config to JSON
    load_config()         -- deserialize JSON to config
    setup_output_dir()    -- create timestamped run directory with subdirs

See also:
    sl_config_template.py -- SLConfig (supervised learning)
    rl_config_template.py -- RLConfig, PPOConfig (reinforcement learning)
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Composable pieces -- attach to any config via fields
# ---------------------------------------------------------------------------


@dataclass
class OutputDir:
    """Output directory configuration.

    Creates a timestamped run directory: {base_dir}/{name}_{YYYYMMDD_HHMMSS}/
    with configurable subdirectories.
    """

    base_dir: str = "output"
    save_config: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    subdirs: Dict[str, str] = field(
        default_factory=lambda: {
            "tensorboard": "tensorboard",
            "checkpoints": "checkpoints",
        }
    )


@dataclass
class ConsoleLogging:
    """Console output capture configuration.

    When enabled, stdout/stderr are captured to a log file in the run directory.
    """

    enabled: bool = True
    filename: str = "console.log"
    separate_streams: bool = False
    stdout_filename: str = "stdout.log"
    stderr_filename: str = "stderr.log"
    tee_to_console: bool = True
    line_timestamps: bool = False
    timestamp_format: str = "%H:%M:%S"
    flush_frequency: int = 1


@dataclass
class Checkpointing:
    """Model checkpointing configuration."""

    enabled: bool = True
    save_best: bool = True
    save_last: bool = True
    save_frequency: int = 0  # epochs or steps; 0 = disabled
    metric: str = "loss"  # metric to track for best model
    mode: str = "min"  # "min" or "max"
    best_filename: str = "model_best.pt"
    last_filename: str = "model_last.pt"
    epoch_filename_format: str = "model_epoch_{epoch}.pt"


@dataclass
class TensorBoard:
    """TensorBoard logging configuration."""

    enabled: bool = True
    log_dir: str = "tensorboard"  # relative to run directory
    flush_secs: int = 120
    log_interval: int = 100  # steps between log writes


# ---------------------------------------------------------------------------
# Base config -- all ML experiments inherit from this
# ---------------------------------------------------------------------------


@dataclass
class MLBaseConfig:
    """Root of the config hierarchy. Every ML config inherits from this."""

    name: str = "experiment"
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0"
    output_dir: str = "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_config(cfg, path: str | Path) -> None:
    """Save a dataclass config to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2, default=str))


def load_config(cls: Type[T], path: str | Path) -> T:
    """Load a JSON file into a dataclass config.

    Nested dataclass fields are reconstructed automatically.
    Unknown keys in the JSON are silently ignored.
    """
    data = json.loads(Path(path).read_text())
    return _from_dict(cls, data)


def setup_output_dir(cfg) -> Path:
    """Create a timestamped run directory and its subdirectories.

    Directory structure:
        {output.base_dir}/{cfg.name}_{timestamp}/
        ├── config.json           (if output.save_config)
        ├── console.log           (if console.enabled)
        ├── checkpoints/          (from output.subdirs)
        └── tensorboard/          (from output.subdirs)

    Returns the path to the run directory.
    """
    import dataclasses

    # Get output config (composable piece or fall back to output_dir string)
    output_cfg = getattr(cfg, "output", None)
    if output_cfg is not None and dataclasses.is_dataclass(output_cfg):
        base_dir = output_cfg.base_dir
        timestamp_fmt = output_cfg.timestamp_format
        subdirs = output_cfg.subdirs
        do_save_config = output_cfg.save_config
    else:
        base_dir = getattr(cfg, "output_dir", "output")
        timestamp_fmt = "%Y%m%d_%H%M%S"
        subdirs = {"tensorboard": "tensorboard", "checkpoints": "checkpoints"}
        do_save_config = True

    # Build timestamped run directory
    timestamp = datetime.now().strftime(timestamp_fmt)
    run_dir = Path(base_dir) / f"{cfg.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for subdir_name in subdirs.values():
        (run_dir / subdir_name).mkdir(exist_ok=True)

    # Save config
    if do_save_config:
        save_config(cfg, run_dir / "config.json")

    return run_dir


class TeeWriter:
    """Write to both a file and original stream (for console logging)."""

    def __init__(self, file_handle, original_stream):
        self.file_handle = file_handle
        self.original_stream = original_stream

    def write(self, text):
        self.original_stream.write(text)
        self.file_handle.write(text)

    def flush(self):
        self.original_stream.flush()
        self.file_handle.flush()


def setup_console_logging(cfg, run_dir: Path):
    """Set up console output capture to the run directory.

    Returns a cleanup function to restore original streams.
    """
    console_cfg = getattr(cfg, "console", None)
    if console_cfg is None or not getattr(console_cfg, "enabled", False):
        return lambda: None

    log_path = run_dir / console_cfg.filename
    log_file = open(log_path, "w")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    if console_cfg.tee_to_console:
        sys.stdout = TeeWriter(log_file, orig_stdout)
        sys.stderr = TeeWriter(log_file, orig_stderr)
    else:
        sys.stdout = log_file
        sys.stderr = log_file

    def cleanup():
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()

    return cleanup


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively construct a dataclass from a dict."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, val in data.items():
        if key not in fields:
            continue
        ft = fields[key].type
        # Resolve Optional[X] -> X
        origin = getattr(ft, "__origin__", None)
        if origin is type(None):
            kwargs[key] = val
        elif dataclasses.is_dataclass(ft):
            kwargs[key] = _from_dict(ft, val) if isinstance(val, dict) else val
        else:
            kwargs[key] = val
    return cls(**kwargs)
