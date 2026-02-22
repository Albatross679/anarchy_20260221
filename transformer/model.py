"""
Transformer encoder model for energy consumption prediction from temporal feature windows.

Functions:
    create_model     -- instantiate EnergyTransformer from config
    create_datasets  -- convert DataFrames into windowed PyTorch Datasets
    train_model      -- full training loop with validation and early stopping
    evaluate_model   -- compute RMSE, MAE, R², MAPE on test set
    save_model       -- persist state_dict + scaler stats
    load_model       -- load from checkpoint
    get_predictions  -- add predicted and residual columns to DataFrame
"""

import math
from dataclasses import asdict
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

from transformer.config import TransformerDataConfig, TransformerParams, TensorBoardConfig, log_system_metrics_to_tb


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EnergySequenceDataset(Dataset):
    """Sliding-window dataset over per-building time series.

    Groups by building (simscode), sorts by time, creates windows of
    ``seq_length`` timesteps.  Target = energy_per_sqft at the last timestep.

    Returns (seq_length, n_features) shape — Transformers expect (batch, seq_len, features).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_length: int = 24,
        stride: int = 1,
        scaler_stats: Optional[dict] = None,
        normalize_features: bool = True,
        normalize_target: bool = True,
    ):
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.normalize_features = normalize_features
        self.normalize_target = normalize_target

        # Build per-building sorted arrays
        windows_X: list[np.ndarray] = []
        windows_y: list[np.ndarray] = []
        # Track (simscode, readingtime) for each window's last timestep
        self.index_keys: list[tuple] = []

        for code, grp in df.groupby("simscode"):
            grp = grp.sort_values("readingtime")
            features = grp[feature_cols].values.astype(np.float32)
            targets = grp["energy_per_sqft"].values.astype(np.float32)
            times = grp["readingtime"].values

            n = len(grp)
            for start in range(0, n - seq_length + 1, stride):
                end = start + seq_length
                windows_X.append(features[start:end])
                windows_y.append(targets[end - 1])
                self.index_keys.append((code, times[end - 1]))

        self.X = np.stack(windows_X)  # (N, seq_length, n_features)
        self.y = np.array(windows_y)  # (N,)

        # Compute or apply scaler stats
        if scaler_stats is None:
            self.scaler_stats = {}
            if normalize_features:
                self.scaler_stats["feature_mean"] = self.X.mean(axis=(0, 1)).tolist()
                self.scaler_stats["feature_std"] = (
                    self.X.std(axis=(0, 1)) + 1e-8
                ).tolist()
            if normalize_target:
                self.scaler_stats["target_mean"] = float(self.y.mean())
                self.scaler_stats["target_std"] = float(self.y.std() + 1e-8)
        else:
            self.scaler_stats = scaler_stats

        # Apply normalization in-place
        if normalize_features and "feature_mean" in self.scaler_stats:
            mean = np.array(self.scaler_stats["feature_mean"], dtype=np.float32)
            std = np.array(self.scaler_stats["feature_std"], dtype=np.float32)
            self.X = (self.X - mean) / std

        if normalize_target and "target_mean" in self.scaler_stats:
            self.y = (
                self.y - self.scaler_stats["target_mean"]
            ) / self.scaler_stats["target_std"]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transformer expects (seq_length, n_features) — no transpose needed
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to input embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class EnergyTransformer(nn.Module):
    """Transformer encoder for energy prediction from temporal feature windows.

    Architecture:
        Linear(n_features, d_model) -> PositionalEncoding
        -> TransformerEncoder (n_layers x TransformerEncoderLayer)
        -> Mean pooling over sequence
        -> FC head: Linear -> Activation -> Dropout -> Linear(1)
    """

    def __init__(self, n_features: int, seq_length: int, params: TransformerParams):
        super().__init__()
        self.n_features = n_features
        self.seq_length = seq_length

        act_fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }[params.activation]

        # Input projection
        self.input_proj = nn.Linear(n_features, params.d_model)

        # Positional encoding
        self.use_positional_encoding = params.use_positional_encoding
        if params.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                params.d_model, max_len=seq_length, dropout=params.dropout
            )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params.d_model,
            nhead=params.n_heads,
            dim_feedforward=params.d_ff,
            dropout=params.dropout,
            activation=params.activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=params.n_layers
        )

        # FC head
        fc_layers = []
        fc_in = params.d_model
        for fc_dim in params.fc_dims:
            fc_layers.append(nn.Linear(fc_in, fc_dim))
            fc_layers.append(act_fn())
            fc_layers.append(nn.Dropout(params.dropout_fc))
            fc_in = fc_dim

        fc_layers.append(nn.Linear(fc_in, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_length, n_features)
        x = self.input_proj(x)                  # (B, seq_length, d_model)
        if self.use_positional_encoding:
            x = self.pos_encoder(x)             # (B, seq_length, d_model)
        x = self.transformer_encoder(x)         # (B, seq_length, d_model)
        x = x.mean(dim=1)                       # (B, d_model) — mean pooling
        x = self.fc(x)                          # (B, 1)
        return x.squeeze(-1)                    # (B,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_model(
    params: TransformerParams,
    n_features: int,
    seq_length: int,
    device: str = "auto",
) -> Tuple[EnergyTransformer, torch.device]:
    """Create EnergyTransformer and move to appropriate device."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = EnergyTransformer(n_features, seq_length, params).to(device)
    return model, device


def create_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    data_cfg: TransformerDataConfig,
) -> Tuple[EnergySequenceDataset, EnergySequenceDataset, dict]:
    """Create windowed train/test datasets. Scaler stats computed from train."""
    train_ds = EnergySequenceDataset(
        df_train,
        feature_cols,
        seq_length=data_cfg.seq_length,
        stride=data_cfg.stride,
        scaler_stats=None,
        normalize_features=data_cfg.normalize_features,
        normalize_target=data_cfg.normalize_target,
    )

    # Reuse training scaler stats for test
    test_ds = EnergySequenceDataset(
        df_test,
        feature_cols,
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
    model: EnergyTransformer,
    train_dataset: EnergySequenceDataset,
    test_dataset: EnergySequenceDataset,
    params: TransformerParams,
    data_cfg: TransformerDataConfig,
    device: torch.device,
    run_dir: Optional[Path] = None,
    tb_cfg: Optional[TensorBoardConfig] = None,
    resume_from: Optional[str | Path] = None,
) -> EnergyTransformer:
    """Train with AdamW, LR scheduler, early stopping, and TensorBoard logging.

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
            f"| d_model | {params.d_model} |\n"
            f"| n_heads | {params.n_heads} |\n"
            f"| n_layers | {params.n_layers} |\n"
            f"| d_ff | {params.d_ff} |\n"
            f"| fc_dims | {params.fc_dims} |\n"
            f"| dropout | {params.dropout} |\n"
            f"| dropout_fc | {params.dropout_fc} |\n"
            f"| activation | {params.activation} |\n"
            f"| scheduler | {params.scheduler} |\n"
            f"| positional_encoding | {params.use_positional_encoding} |\n"
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
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate (collect predictions for real-unit metrics) ---
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
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
    model: EnergyTransformer,
    test_dataset: EnergySequenceDataset,
    data_cfg: TransformerDataConfig,
    device: torch.device,
    scaler_stats: Optional[dict] = None,
    run_dir: Optional[Path] = None,
    params: Optional[TransformerParams] = None,
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
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
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
                "d_model": params.d_model,
                "n_heads": params.n_heads,
                "n_layers": params.n_layers,
                "d_ff": params.d_ff,
                "fc_dims": str(params.fc_dims),
                "dropout": params.dropout,
                "dropout_fc": params.dropout_fc,
                "activation": params.activation,
                "scheduler": params.scheduler,
                "seq_length": data_cfg.seq_length,
                "batch_size": data_cfg.batch_size,
                "stride": data_cfg.stride,
                "positional_encoding": params.use_positional_encoding,
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
    model: EnergyTransformer,
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
        "n_features": model.n_features,
        "seq_length": model.seq_length,
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
    params: TransformerParams,
    n_features: int,
    seq_length: int,
    device: str = "auto",
) -> Tuple[EnergyTransformer, dict, torch.device]:
    """Load model from checkpoint. Returns (model, scaler_stats, device)."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EnergyTransformer(n_features, seq_length, params).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["scaler_stats"], device


def get_predictions(
    model: EnergyTransformer,
    df: pd.DataFrame,
    feature_cols: list[str],
    data_cfg: TransformerDataConfig,
    device: torch.device,
    scaler_stats: dict,
) -> pd.DataFrame:
    """Add 'predicted' and 'residual' columns to DataFrame.

    Each prediction maps to the last timestep of its window.
    Rows without enough preceding context get NaN.
    """
    # Build dataset without normalization on target (we denormalize preds manually)
    ds = EnergySequenceDataset(
        df,
        feature_cols,
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
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
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
