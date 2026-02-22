"""
1D CNN model for energy consumption prediction from temporal feature windows.

Functions:
    create_model     -- instantiate EnergyCNN from config
    create_datasets  -- convert DataFrames into windowed PyTorch Datasets
    train_model      -- full training loop with validation and early stopping
    evaluate_model   -- compute RMSE, MAE, R², MAPE on test set
    save_model       -- persist state_dict + scaler stats
    load_model       -- load from checkpoint
    get_predictions  -- add predicted and residual columns to DataFrame
"""

import time as _time
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

from cnn.config import CNNDataConfig, CNNParams, TensorBoardConfig, log_system_metrics_to_tb


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EnergySequenceDataset(Dataset):
    """Sliding-window dataset over per-building time series.

    Groups by building (simscode), sorts by time, creates windows of
    ``seq_length`` timesteps.  Target = energy_per_sqft at the last timestep.
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
        # Conv1d expects (channels, length) -> transpose to (n_features, seq_length)
        x = torch.from_numpy(self.X[idx].T)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class EnergyCNN(nn.Module):
    """1D CNN for energy prediction from temporal feature windows.

    Architecture:
        N x [Conv1d -> (BatchNorm) -> Activation -> MaxPool1d -> Dropout]
        -> AdaptiveAvgPool1d(1) -> Flatten
        -> M x [Linear -> Activation -> Dropout]
        -> Linear(1)
    """

    def __init__(self, n_features: int, seq_length: int, params: CNNParams):
        super().__init__()
        self.n_features = n_features
        self.seq_length = seq_length

        # Validate conv_channels and kernel_sizes have matching lengths
        if len(params.conv_channels) != len(params.kernel_sizes):
            raise ValueError(
                f"conv_channels ({len(params.conv_channels)}) and "
                f"kernel_sizes ({len(params.kernel_sizes)}) must have the same length"
            )

        # Validate seq_length is large enough to survive MaxPool layers
        n_pools = len(params.conv_channels)
        min_seq = 2 ** n_pools
        if seq_length < min_seq:
            raise ValueError(
                f"seq_length={seq_length} is too short for {n_pools} MaxPool1d(2) layers; "
                f"minimum is {min_seq}"
            )

        act_fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }[params.activation]

        # Conv blocks
        layers = []
        in_ch = n_features
        for out_ch, ks in zip(params.conv_channels, params.kernel_sizes):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2))
            if params.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(act_fn())
            layers.append(nn.MaxPool1d(kernel_size=params.pool_size))
            layers.append(nn.Dropout(params.dropout_conv))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # FC head
        fc_layers = []
        fc_in = params.conv_channels[-1]
        for fc_dim in params.fc_dims:
            fc_layers.append(nn.Linear(fc_in, fc_dim))
            fc_layers.append(act_fn())
            fc_layers.append(nn.Dropout(params.dropout_fc))
            fc_in = fc_dim

        fc_layers.append(nn.Linear(fc_in, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features, seq_length)
        x = self.conv(x)
        x = self.pool(x)       # (B, C, 1)
        x = x.squeeze(-1)      # (B, C)
        x = self.fc(x)         # (B, 1)
        return x.squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_model(
    params: CNNParams,
    n_features: int,
    seq_length: int,
    device: str = "auto",
) -> Tuple[EnergyCNN, torch.device]:
    """Create EnergyCNN and move to appropriate device."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = EnergyCNN(n_features, seq_length, params).to(device)
    return model, device


def create_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    data_cfg: CNNDataConfig,
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
    model: EnergyCNN,
    train_dataset: EnergySequenceDataset,
    test_dataset: EnergySequenceDataset,
    params: CNNParams,
    data_cfg: CNNDataConfig,
    device: torch.device,
    run_dir: Optional[Path] = None,
    save_every: int = 5,
    tb_cfg: Optional[TensorBoardConfig] = None,
    resume_from: Optional[str | Path] = None,
) -> EnergyCNN:
    """Train with AdamW, LR scheduler, early stopping, and TensorBoard logging.

    Checkpoint saving (requires run_dir):
    - Every ``save_every`` epochs  -> checkpoints/epoch_{N}.pt
    - Best validation loss         -> checkpoints/model_best.pt
    - Final model (after training) -> checkpoints/model_final.pt

    Resume: pass ``resume_from`` path to a checkpoint containing
    optimizer_state_dict, scheduler_state_dict, epoch, and best_val_loss.
    """
    scaler_stats_for_ckpt = train_dataset.scaler_stats
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
            f"| conv_channels | {params.conv_channels} |\n"
            f"| kernel_sizes | {params.kernel_sizes} |\n"
            f"| fc_dims | {params.fc_dims} |\n"
            f"| dropout_conv | {params.dropout_conv} |\n"
            f"| dropout_fc | {params.dropout_fc} |\n"
            f"| pool_size | {params.pool_size} |\n"
            f"| batch_norm | {params.use_batch_norm} |\n"
            f"| activation | {params.activation} |\n"
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
    train_start = _time.time()

    for epoch in range(start_epoch, params.epochs + 1):
        epoch_start = _time.time()

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

        # Timing
        epoch_time = _time.time() - epoch_start
        wall_clock = _time.time() - train_start

        # Log to TensorBoard
        if writer:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalar("lr", current_lr, epoch)
            writer.add_scalar("metrics/val_rmse", val_metrics["rmse"], epoch)
            writer.add_scalar("metrics/val_mae", val_metrics["mae"], epoch)
            writer.add_scalar("metrics/val_r2", val_metrics["r2"], epoch)
            writer.add_scalar("time/epoch_seconds", epoch_time, epoch)
            writer.add_scalar("time/wall_clock_seconds", wall_clock, epoch)

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
            # Save best checkpoint to disk
            if ckpt_dir:
                save_model(
                    model, scaler_stats_for_ckpt, ckpt_dir / "model_best.pt",
                    optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, best_val_loss=best_val_loss,
                )
        else:
            patience_counter += 1

        # Periodic checkpoint every N epochs
        if ckpt_dir and epoch % save_every == 0:
            save_model(
                model, scaler_stats_for_ckpt, ckpt_dir / f"epoch_{epoch}.pt",
                optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_val_loss=best_val_loss,
            )

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

    # Save final model (last epoch state, before restoring best)
    if ckpt_dir:
        save_model(
            model, scaler_stats_for_ckpt, ckpt_dir / "model_final.pt",
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_val_loss=best_val_loss,
        )

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    if writer:
        writer.close()

    return model


def evaluate_model(
    model: EnergyCNN,
    test_dataset: EnergySequenceDataset,
    data_cfg: CNNDataConfig,
    device: torch.device,
    scaler_stats: Optional[dict] = None,
    run_dir: Optional[Path] = None,
    params: Optional[CNNParams] = None,
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
                "conv_channels": str(params.conv_channels),
                "kernel_sizes": str(params.kernel_sizes),
                "fc_dims": str(params.fc_dims),
                "dropout_conv": params.dropout_conv,
                "dropout_fc": params.dropout_fc,
                "pool_size": params.pool_size,
                "batch_norm": params.use_batch_norm,
                "activation": params.activation,
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
    model: EnergyCNN,
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
    params: CNNParams,
    n_features: int,
    seq_length: int,
    device: str = "auto",
) -> Tuple[EnergyCNN, dict, torch.device]:
    """Load model from checkpoint. Returns (model, scaler_stats, device)."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EnergyCNN(n_features, seq_length, params).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["scaler_stats"], device


def get_predictions(
    model: EnergyCNN,
    df: pd.DataFrame,
    feature_cols: list[str],
    data_cfg: CNNDataConfig,
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
