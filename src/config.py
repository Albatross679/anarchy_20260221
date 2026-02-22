"""
Configuration hierarchy for energy investment prioritization.

Builds on MLBaseConfig template with XGBoost-specific and data pipeline configs.
"""

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import psutil
import torch

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Composable pieces
# ---------------------------------------------------------------------------


@dataclass
class OutputDir:
    base_dir: str = "output"
    save_config: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    subdirs: Dict[str, str] = field(
        default_factory=lambda: {"checkpoints": "checkpoints"}
    )


@dataclass
class ConsoleLogging:
    enabled: bool = True
    filename: str = "console.log"
    tee_to_console: bool = True


@dataclass
class Checkpointing:
    enabled: bool = True
    save_best: bool = True
    save_every_n_epochs: int = 5
    metric: str = "rmse"
    mode: str = "min"
    best_filename: str = "model_best.json"


@dataclass
class TensorBoardConfig:
    """TensorBoard logging settings — applied across all model types."""

    enabled: bool = True

    # System metrics logged per epoch
    log_system_metrics: bool = True
    log_cpu: bool = True
    log_gpu_utilization: bool = True
    log_vram: bool = True
    log_gpu_temp: bool = True
    log_gpu_power: bool = True

    # Hyperparameters as text at step 0
    log_hparams_text: bool = True

    # Weight/gradient histograms
    log_histograms: bool = True
    histogram_every_n_epochs: int = 5


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------


@dataclass
class CleanerConfig:
    enabled: bool = True
    lookup_path: str = "data/site_to_building_lookup.csv"
    fuzzy_threshold: float = 80.0
    gap_limit_intervals: int = 8


@dataclass
class DataConfig:
    meter_files: List[str] = field(
        default_factory=lambda: [
            "data/meter-data-sept-2025.csv",
            "data/meter-data-oct-2025.csv",
        ]
    )
    building_metadata_file: str = "data/building_metadata.csv"
    weather_file: str = "data/weather-sept-oct-2025.csv"

    utility_filter: str = "ELECTRICITY"
    cleaner: CleanerConfig = field(default_factory=CleanerConfig)

    # Use pre-cleaned CSVs (data/cleaned_{utility}.csv) instead of raw meter files
    use_cleaned_data: bool = True
    cleaned_data_dir: str = "data"

    weather_features: List[str] = field(
        default_factory=lambda: [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "direct_radiation",
            "wind_speed_10m",
            "cloud_cover",
            "apparent_temperature",
            "precipitation",
        ]
    )
    building_features: List[str] = field(
        default_factory=lambda: [
            "grossarea",
            "floorsaboveground",
            "building_age",
        ]
    )
    time_features: List[str] = field(
        default_factory=lambda: [
            "hour_of_day",
            "minute_of_hour",
            "day_of_week",
            "is_weekend",
        ]
    )

    # Pre-computed tree features (from src/prepare_tree_features.py)
    use_precomputed_features: bool = False
    precomputed_features_dir: str = "data"

    # Temporal split: train on Sept, test on Oct
    temporal_split: bool = True
    split_date: str = "2025-10-01"
    random_split_ratio: float = 0.8


# ---------------------------------------------------------------------------
# XGBoost parameters
# ---------------------------------------------------------------------------


@dataclass
class XGBoostParams:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    tree_method: str = "hist"
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------


@dataclass
class MLBaseConfig:
    name: str = "experiment"
    model_type: str = "model"
    seed: int = 42
    output_dir: str = "output"
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)

    def get_run_name(self) -> str:
        """Build run name as {utility}_{model_type}.

        Uses data.utility_filter if available, otherwise falls back to self.name.
        """
        data_cfg = getattr(self, "data", None)
        utility = getattr(data_cfg, "utility_filter", None)
        if utility:
            return f"{utility.lower()}_{self.model_type}"
        return self.name


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class EnergyModelConfig(MLBaseConfig):
    name: str = "electricity_xgboost"
    model_type: str = "xgboost"
    output: OutputDir = field(default_factory=OutputDir)
    console: ConsoleLogging = field(default_factory=ConsoleLogging)
    checkpointing: Checkpointing = field(default_factory=Checkpointing)
    data: DataConfig = field(default_factory=DataConfig)
    xgb: XGBoostParams = field(default_factory=XGBoostParams)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_config(cfg, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2, default=str))


def load_config(cls: Type[T], path: str | Path) -> T:
    data = json.loads(Path(path).read_text())
    return _from_dict(cls, data)


def setup_output_dir(cfg) -> Path:
    import dataclasses

    output_cfg = getattr(cfg, "output", None)
    if output_cfg is not None and dataclasses.is_dataclass(output_cfg):
        base_dir = output_cfg.base_dir
        timestamp_fmt = output_cfg.timestamp_format
        subdirs = output_cfg.subdirs
        do_save_config = output_cfg.save_config
    else:
        base_dir = getattr(cfg, "output_dir", "output")
        timestamp_fmt = "%Y%m%d_%H%M%S"
        subdirs = {"checkpoints": "checkpoints"}
        do_save_config = True

    timestamp = datetime.now().strftime(timestamp_fmt)
    run_name = cfg.get_run_name() if hasattr(cfg, "get_run_name") else cfg.name
    run_dir = Path(base_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for subdir_name in subdirs.values():
        (run_dir / subdir_name).mkdir(exist_ok=True)

    if do_save_config:
        save_config(cfg, run_dir / "config.json")

    return run_dir


class TeeWriter:
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


def get_system_metrics(tb_cfg: "TensorBoardConfig") -> dict:
    """Collect system metrics according to TensorBoardConfig flags."""
    metrics = {}
    if not tb_cfg.log_system_metrics:
        return metrics

    if tb_cfg.log_cpu:
        metrics["cpu_percent"] = psutil.cpu_percent(interval=None)

    if torch.cuda.is_available():
        if tb_cfg.log_vram:
            metrics["vram_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            metrics["vram_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            metrics["vram_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2

        if tb_cfg.log_gpu_utilization or tb_cfg.log_gpu_temp or tb_cfg.log_gpu_power or tb_cfg.log_vram:
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.total,memory.used,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    parts = [p.strip() for p in result.stdout.strip().split(",")]
                    if tb_cfg.log_gpu_utilization:
                        metrics["gpu_utilization_pct"] = float(parts[0])
                    if tb_cfg.log_vram:
                        metrics["vram_total_mb"] = float(parts[1])
                        metrics["vram_used_mb"] = float(parts[2])
                    if tb_cfg.log_gpu_temp:
                        metrics["gpu_temp_c"] = float(parts[3])
                    if tb_cfg.log_gpu_power:
                        metrics["gpu_power_w"] = float(parts[4])
            except (subprocess.TimeoutExpired, Exception):
                pass

    return metrics


def log_system_metrics_to_tb(writer, tb_cfg: "TensorBoardConfig", epoch: int) -> None:
    """Log system metrics to a TensorBoard SummaryWriter."""
    if not tb_cfg.enabled or not tb_cfg.log_system_metrics:
        return
    metrics = get_system_metrics(tb_cfg)
    for key, val in metrics.items():
        writer.add_scalar(f"system/{key}", val, epoch)


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, val in data.items():
        if key not in fields:
            continue
        ft = fields[key].type
        if dataclasses.is_dataclass(ft):
            kwargs[key] = _from_dict(ft, val) if isinstance(val, dict) else val
        else:
            kwargs[key] = val
    return cls(**kwargs)
