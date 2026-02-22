"""
Hybrid LSTM model for energy consumption prediction.

Dual-branch architecture:
    temporal (B, seq_len, n_temporal)  ->  LSTM  ->  temporal_emb
    static   (B, n_static)            ->  MLP   ->  static_emb
    concat(temporal_emb, static_emb)  ->  MLP head  ->  prediction

Functions:
    create_model     -- instantiate EnergyLSTMHybrid from config
    create_datasets  -- convert DataFrames into windowed PyTorch Datasets
    train_model      -- full training loop with validation and early stopping
    evaluate_model   -- compute RMSE, MAE, R², MAPE on test set
    save_model       -- persist state_dict + scaler stats
    load_model       -- load from checkpoint
    get_predictions  -- add predicted and residual columns to DataFrame
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from lstm.config import LSTMDataConfig, LSTMParams, TensorBoardConfig, log_system_metrics_to_tb


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EnergyLSTMDataset(Dataset):
    """Sliding-window dataset over per-building time series.

    Groups by building (simscode), sorts by time, creates windows of
    ``seq_length`` timesteps.  Target = energy_per_sqft at the last timestep.

    Returns 3-tuples:
        temporal: (seq_length, n_temporal) — weather + time features per timestep
        static:   (n_static,)             — building features (constant per building)
        target:   scalar                  — energy_per_sqft at last timestep
    """

    def __init__(
        self,
        df: pd.DataFrame,
        temporal_cols: list[str],
        static_cols: list[str],
        seq_length: int = 96,
        stride: int = 4,
        scaler_stats: Optional[dict] = None,
        normalize_features: bool = True,
        normalize_target: bool = True,
    ):
        self.seq_length = seq_length
        self.temporal_cols = temporal_cols
        self.static_cols = static_cols
        self.normalize_features = normalize_features
        self.normalize_target = normalize_target

        # Build per-building sorted arrays
        windows_temporal: list[np.ndarray] = []
        windows_static: list[np.ndarray] = []
        windows_y: list[np.ndarray] = []
        self.index_keys: list[tuple] = []

        for code, grp in df.groupby("simscode"):
            grp = grp.sort_values("readingtime")
            temporal = grp[temporal_cols].values.astype(np.float32)
            static = grp[static_cols].iloc[0].values.astype(np.float32)
            targets = grp["energy_per_sqft"].values.astype(np.float32)
            times = grp["readingtime"].values

            n = len(grp)
            for start in range(0, n - seq_length + 1, stride):
                end = start + seq_length
                windows_temporal.append(temporal[start:end])
                windows_static.append(static)
                windows_y.append(targets[end - 1])
                self.index_keys.append((code, times[end - 1]))

        self.X_temporal = np.stack(windows_temporal)  # (N, seq_length, n_temporal)
        self.X_static = np.stack(windows_static)      # (N, n_static)
        self.y = np.array(windows_y)                   # (N,)

        # Compute or apply scaler stats
        if scaler_stats is None:
            self.scaler_stats = {}
            if normalize_features:
                self.scaler_stats["temporal_mean"] = self.X_temporal.mean(axis=(0, 1)).tolist()
                self.scaler_stats["temporal_std"] = (
                    self.X_temporal.std(axis=(0, 1)) + 1e-8
                ).tolist()
                self.scaler_stats["static_mean"] = self.X_static.mean(axis=0).tolist()
                self.scaler_stats["static_std"] = (
                    self.X_static.std(axis=0) + 1e-8
                ).tolist()
            if normalize_target:
                self.scaler_stats["target_mean"] = float(self.y.mean())
                self.scaler_stats["target_std"] = float(self.y.std() + 1e-8)
        else:
            self.scaler_stats = scaler_stats

        # Apply normalization in-place
        if normalize_features and "temporal_mean" in self.scaler_stats:
            mean = np.array(self.scaler_stats["temporal_mean"], dtype=np.float32)
            std = np.array(self.scaler_stats["temporal_std"], dtype=np.float32)
            self.X_temporal = (self.X_temporal - mean) / std

        if normalize_features and "static_mean" in self.scaler_stats:
            mean = np.array(self.scaler_stats["static_mean"], dtype=np.float32)
            std = np.array(self.scaler_stats["static_std"], dtype=np.float32)
            self.X_static = (self.X_static - mean) / std

        if normalize_target and "target_mean" in self.scaler_stats:
            self.y = (
                self.y - self.scaler_stats["target_mean"]
            ) / self.scaler_stats["target_std"]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # LSTM expects (seq_length, n_features) with batch_first=True
        temporal = torch.from_numpy(self.X_temporal[idx])
        static = torch.from_numpy(self.X_static[idx])
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        return temporal, static, target


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class EnergyLSTMHybrid(nn.Module):
    """Hybrid LSTM for energy prediction from temporal + static features.

    Architecture:
        Temporal branch: LSTM(hidden_size, num_layers) -> h_n[-1]
        Static branch:   MLP [Linear -> ReLU -> Dropout]xN -> Linear -> ReLU
        Fusion:          concat -> MLP head [Linear -> GELU -> Dropout]xN -> Linear(1)
    """

    def __init__(
        self,
        n_temporal: int,
        n_static: int,
        params: LSTMParams,
    ):
        super().__init__()
        self.n_temporal = n_temporal
        self.n_static = n_static
        self.hidden_size = params.hidden_size
        self.num_layers = params.num_layers
        self.bidirectional = params.bidirectional

        # --- Temporal branch: LSTM ---
        self.lstm = nn.LSTM(
            input_size=n_temporal,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            batch_first=True,
            dropout=params.dropout_lstm if params.num_layers > 1 else 0.0,
            bidirectional=params.bidirectional,
        )

        # Output dim of LSTM branch
        lstm_out_dim = params.hidden_size * (2 if params.bidirectional else 1)

        # --- Static branch: MLP ---
        static_layers = []
        in_dim = n_static
        for hdim in params.static_hidden_dims:
            static_layers.append(nn.Linear(in_dim, hdim))
            static_layers.append(nn.ReLU())
            static_layers.append(nn.Dropout(params.dropout_static))
            in_dim = hdim
        static_layers.append(nn.Linear(in_dim, params.static_embedding_dim))
        static_layers.append(nn.ReLU())
        self.static_mlp = nn.Sequential(*static_layers)

        # --- Fusion head (GELU activation to match seq_experiment) ---
        fusion_in = lstm_out_dim + params.static_embedding_dim
        head_layers = []
        in_dim = fusion_in
        for hdim in params.head_dims:
            head_layers.append(nn.Linear(in_dim, hdim))
            head_layers.append(nn.GELU())
            head_layers.append(nn.Dropout(params.dropout_head))
            in_dim = hdim
        head_layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, temporal: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal: (B, seq_length, n_temporal)
            static:   (B, n_static)
        Returns:
            prediction: (B,)
        """
        # Temporal branch
        _, (h_n, _) = self.lstm(temporal)
        # h_n: (num_layers * num_directions, B, hidden_size)
        if self.bidirectional:
            # Concat forward and backward last-layer hidden states
            temporal_emb = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            temporal_emb = h_n[-1]  # (B, hidden_size)

        # Static branch
        static_emb = self.static_mlp(static)  # (B, static_embedding_dim)

        # Fusion
        fused = torch.cat([temporal_emb, static_emb], dim=-1)
        out = self.head(fused)  # (B, 1)
        return out.squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_model(
    params: LSTMParams,
    n_temporal: int,
    n_static: int,
    device: str = "auto",
) -> Tuple[EnergyLSTMHybrid, torch.device]:
    """Create EnergyLSTMHybrid and move to appropriate device."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = EnergyLSTMHybrid(n_temporal, n_static, params).to(device)
    return model, device


def create_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    temporal_cols: list[str],
    static_cols: list[str],
    data_cfg: LSTMDataConfig,
) -> Tuple[EnergyLSTMDataset, EnergyLSTMDataset, dict]:
    """Create windowed train/test datasets. Scaler stats computed from train."""
    train_ds = EnergyLSTMDataset(
        df_train,
        temporal_cols,
        static_cols,
        seq_length=data_cfg.seq_length,
        stride=data_cfg.stride,
        scaler_stats=None,
        normalize_features=data_cfg.normalize_features,
        normalize_target=data_cfg.normalize_target,
    )

    # Reuse training scaler stats for test
    test_ds = EnergyLSTMDataset(
        df_test,
        temporal_cols,
        static_cols,
        seq_length=data_cfg.seq_length,
        stride=data_cfg.stride,
        scaler_stats=train_ds.scaler_stats,
        normalize_features=data_cfg.normalize_features,
        normalize_target=data_cfg.normalize_target,
    )

    return train_ds, test_ds, train_ds.scaler_stats


def _denormalize(values: np.ndarray, scaler_stats: dict) -> np.ndarray:
    """Denormalize target values using scaler stats."""
    if "target_mean" in scaler_stats:
        return values * scaler_stats["target_std"] + scaler_stats["target_mean"]
    return values


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, R² from arrays (assumed already denormalized)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_model(
    model: EnergyLSTMHybrid,
    train_dataset: EnergyLSTMDataset,
    test_dataset: EnergyLSTMDataset,
    params: LSTMParams,
    data_cfg: LSTMDataConfig,
    device: torch.device,
    run_dir: Optional[Path] = None,
    tb_cfg: Optional[TensorBoardConfig] = None,
    resume_from: Optional[str | Path] = None,
) -> EnergyLSTMHybrid:
    """Train with AdamW, LR scheduler, gradient clipping, early stopping, and TensorBoard logging.

    Resume: pass ``resume_from`` path to a checkpoint containing
    optimizer_state_dict, scheduler_state_dict, epoch, and best_val_loss.
    """
    ckpt_dir = (run_dir / "checkpoints") if run_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )

    scaler_stats = train_dataset.scaler_stats

    # TensorBoard writer
    tb_dir = run_dir / "tensorboard" if run_dir else None
    writer = SummaryWriter(log_dir=str(tb_dir)) if tb_dir else None

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )

    # LR scheduler
    if params.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params.epochs
        )
    elif params.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.scheduler_step_size,
            gamma=params.scheduler_gamma,
        )
    else:
        scheduler = None

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from {resume_from} (epoch {start_epoch - 1}, best_val_loss={best_val_loss:.6f})")

    if tb_cfg is None:
        tb_cfg = TensorBoardConfig()

    # Log hyperparameters as text at training start
    if writer and tb_cfg.log_hparams_text and start_epoch == 1:
        hparam_text = (
            f"| Param | Value |\n|---|---|\n"
            f"| learning_rate | {params.learning_rate} |\n"
            f"| weight_decay | {params.weight_decay} |\n"
            f"| epochs | {params.epochs} |\n"
            f"| hidden_size | {params.hidden_size} |\n"
            f"| num_layers | {params.num_layers} |\n"
            f"| dropout_lstm | {params.dropout_lstm} |\n"
            f"| bidirectional | {params.bidirectional} |\n"
            f"| static_embedding_dim | {params.static_embedding_dim} |\n"
            f"| head_dims | {params.head_dims} |\n"
            f"| dropout_head | {params.dropout_head} |\n"
            f"| scheduler | {params.scheduler} |\n"
            f"| seq_length | {data_cfg.seq_length} |\n"
            f"| batch_size | {data_cfg.batch_size} |\n"
            f"| stride | {data_cfg.stride} |\n"
        )
        writer.add_text("hyperparameters", hparam_text, 0)

    # Initialize CPU monitoring baseline
    psutil.cpu_percent(interval=None)

    patience_counter = 0
    best_state = None

    for epoch in range(start_epoch, params.epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for temporal_batch, static_batch, y_batch in train_loader:
            temporal_batch = temporal_batch.to(device)
            static_batch = static_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(temporal_batch, static_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate (collect predictions for real-unit metrics) ---
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        with torch.no_grad():
            for temporal_batch, static_batch, y_batch in val_loader:
                temporal_batch = temporal_batch.to(device)
                static_batch = static_batch.to(device)
                y_batch = y_batch.to(device)

                pred = model(temporal_batch, static_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]

        # Compute val metrics in real (denormalized) units
        y_pred_raw = np.concatenate(val_preds)
        y_true_raw = np.concatenate(val_targets)
        y_pred_real = _denormalize(y_pred_raw, scaler_stats)
        y_true_real = _denormalize(y_true_raw, scaler_stats)
        val_metrics = _compute_metrics(y_true_real, y_pred_real)

        # Log to TensorBoard
        if writer:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalar("lr", current_lr, epoch)
            writer.add_scalar("metrics/val_rmse", val_metrics["rmse"], epoch)
            writer.add_scalar("metrics/val_mae", val_metrics["mae"], epoch)
            writer.add_scalar("metrics/val_r2", val_metrics["r2"], epoch)

            # System metrics (CPU, GPU, VRAM) — config-driven
            log_system_metrics_to_tb(writer, tb_cfg, epoch)

            # Weight and gradient histograms
            if tb_cfg.log_histograms and (epoch % tb_cfg.histogram_every_n_epochs == 0 or epoch == 1):
                for name, param in model.named_parameters():
                    writer.add_histogram(f"weights/{name}", param.data, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"gradients/{name}", param.grad, epoch)

        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_dir:
                save_model(
                    model, train_dataset.scaler_stats, ckpt_dir / "model_best.pt",
                    optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, best_val_loss=best_val_loss,
                )
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{params.epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"R²={val_metrics['r2']:.4f}  "
                f"lr={current_lr:.2e}  patience={patience_counter}/{params.early_stopping_patience}"
            )

        if patience_counter >= params.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    if writer:
        writer.close()

    return model


def evaluate_model(
    model: EnergyLSTMHybrid,
    test_dataset: EnergyLSTMDataset,
    data_cfg: LSTMDataConfig,
    device: torch.device,
    scaler_stats: Optional[dict] = None,
    run_dir: Optional[Path] = None,
    params: Optional[LSTMParams] = None,
) -> dict:
    """Evaluate on test set. Logs figures and hparams to TensorBoard."""
    loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
    )

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for temporal_batch, static_batch, y_batch in loader:
            temporal_batch = temporal_batch.to(device)
            static_batch = static_batch.to(device)
            preds = model(temporal_batch, static_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Denormalize
    if scaler_stats and "target_mean" in scaler_stats:
        y_pred = _denormalize(y_pred, scaler_stats)
        y_true = _denormalize(y_true, scaler_stats)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = float("nan")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "n_test": len(y_true),
    }

    # Log figures + hparams to TensorBoard
    if run_dir:
        tb_dir = run_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tb_dir))

        # --- Predicted vs Actual scatter ---
        fig, ax = plt.subplots(figsize=(8, 8))
        sample_idx = np.random.default_rng(42).choice(
            len(y_true), size=min(5000, len(y_true)), replace=False
        )
        ax.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, s=4)
        lims = [
            min(y_true[sample_idx].min(), y_pred[sample_idx].min()),
            max(y_true[sample_idx].max(), y_pred[sample_idx].max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual (energy/sqft)")
        ax.set_ylabel("Predicted (energy/sqft)")
        ax.set_title(f"Predicted vs Actual  (R²={r2:.4f})")
        fig.tight_layout()
        writer.add_figure("figures/pred_vs_actual", fig)
        plt.close(fig)

        # --- Residual distribution ---
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(residuals, bins=100, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (actual - predicted)")
        ax.set_ylabel("Count")
        ax.set_title(f"Residual Distribution  (mean={residuals.mean():.6f}, std={residuals.std():.6f})")
        fig.tight_layout()
        writer.add_figure("figures/residual_distribution", fig)
        plt.close(fig)

        # --- HParams: log hyperparameters tied to final metrics ---
        if params is not None:
            hparam_dict = {
                "lr": params.learning_rate,
                "weight_decay": params.weight_decay,
                "epochs": params.epochs,
                "hidden_size": params.hidden_size,
                "num_layers": params.num_layers,
                "dropout_lstm": params.dropout_lstm,
                "bidirectional": params.bidirectional,
                "static_embedding_dim": params.static_embedding_dim,
                "static_hidden_dims": str(params.static_hidden_dims),
                "head_dims": str(params.head_dims),
                "dropout_head": params.dropout_head,
                "scheduler": params.scheduler,
                "seq_length": data_cfg.seq_length,
                "batch_size": data_cfg.batch_size,
                "stride": data_cfg.stride,
            }
            metric_dict = {
                "hparam/rmse": rmse,
                "hparam/mae": mae,
                "hparam/r2": r2,
            }
            writer.add_hparams(hparam_dict, metric_dict)

        writer.close()

    return metrics


def save_model(
    model: EnergyLSTMHybrid,
    scaler_stats: dict,
    path: str | Path,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    best_val_loss: float = float("inf"),
) -> None:
    """Save model state_dict, scaler stats, and optionally optimizer/scheduler/epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "scaler_stats": scaler_stats,
        "n_temporal_features": model.n_temporal,
        "n_static_features": model.n_static,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_model(
    path: str | Path,
    params: LSTMParams,
    n_temporal: int,
    n_static: int,
    device: str = "auto",
) -> Tuple[EnergyLSTMHybrid, dict, torch.device]:
    """Load model from checkpoint. Returns (model, scaler_stats, device)."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EnergyLSTMHybrid(n_temporal, n_static, params).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["scaler_stats"], device


def get_predictions(
    model: EnergyLSTMHybrid,
    df: pd.DataFrame,
    temporal_cols: list[str],
    static_cols: list[str],
    data_cfg: LSTMDataConfig,
    device: torch.device,
    scaler_stats: dict,
) -> pd.DataFrame:
    """Add 'predicted' and 'residual' columns to DataFrame.

    Each prediction maps to the last timestep of its window.
    Rows without enough preceding context get NaN.
    """
    ds = EnergyLSTMDataset(
        df,
        temporal_cols,
        static_cols,
        seq_length=data_cfg.seq_length,
        stride=1,
        scaler_stats=scaler_stats,
        normalize_features=data_cfg.normalize_features,
        normalize_target=data_cfg.normalize_target,
    )

    loader = DataLoader(ds, batch_size=data_cfg.batch_size, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for temporal_batch, static_batch, _ in loader:
            temporal_batch = temporal_batch.to(device)
            static_batch = static_batch.to(device)
            preds = model(temporal_batch, static_batch).cpu().numpy()
            all_preds.append(preds)

    preds = np.concatenate(all_preds)

    # Denormalize predictions
    if "target_mean" in scaler_stats:
        preds = preds * scaler_stats["target_std"] + scaler_stats["target_mean"]

    # Map predictions back to DataFrame rows via (simscode, readingtime) keys
    pred_df = pd.DataFrame(
        {
            "simscode": [k[0] for k in ds.index_keys],
            "readingtime": [k[1] for k in ds.index_keys],
            "predicted": preds,
        }
    )
    # Deduplicate — keep last prediction for each (simscode, readingtime)
    pred_df = pred_df.drop_duplicates(subset=["simscode", "readingtime"], keep="last")

    result = df.copy()
    result = result.merge(pred_df, on=["simscode", "readingtime"], how="left")
    result["residual"] = result["energy_per_sqft"] - result["predicted"]

    return result
