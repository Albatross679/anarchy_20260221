"""
Temporal Fusion Transformer for energy consumption prediction.

Architecture:
    static   (B, n_static)            ->  StaticCovariateEncoder  ->  4 context vectors
    temporal (B, seq_len, n_temporal)  ->  TemporalVSN -> LSTM -> Enrichment -> Attention -> output

Functions:
    create_model     -- instantiate TemporalFusionTransformer from config
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from tft.config import TFTDataConfig, TFTParams


# ---------------------------------------------------------------------------
# Architecture modules
# ---------------------------------------------------------------------------


class GatedLinearUnit(nn.Module):
    """GLU: splits input in half, applies sigmoid gate to one half."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """GRN: ELU -> Linear -> GLU -> LayerNorm + skip connection.

    Optionally accepts a context vector for static enrichment.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
        activation: str = "elu",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context_proj = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GatedLinearUnit(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        # Skip connection projection if dimensions differ
        self.skip_proj = nn.Linear(input_size, output_size) if input_size != output_size else None

        self.activation = F.elu if activation == "elu" else F.relu

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.skip_proj(x) if self.skip_proj else x

        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            # Broadcast context across time dimension if needed
            if context.dim() < hidden.dim():
                context = context.unsqueeze(1).expand_as(hidden)
            hidden = hidden + self.context_proj(context)
        hidden = self.activation(hidden)

        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.glu(hidden)

        return self.layer_norm(hidden + residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: per-variable projections -> softmax weights -> weighted sum.

    Produces interpretable feature importance weights.
    """

    def __init__(
        self,
        input_size: int,
        num_variables: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.hidden_size = hidden_size

        # Per-variable GRNs to transform each variable independently
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_variables)
        ])

        # Softmax weight GRN (operates on flattened input)
        self.weight_grn = GatedResidualNetwork(
            num_variables * input_size, hidden_size, num_variables,
            dropout=dropout, context_size=context_size,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, [T,] num_variables, input_size) or list of per-variable tensors
            context: optional (B, context_size) static context
        Returns:
            selected: (B, [T,] hidden_size) — weighted combination
            weights:  (B, [T,] num_variables) — variable importance weights
        """
        # x shape: (B, [T,] num_variables * input_size) for flattened, or (B, [T,] num_variables, input_size)
        if x.dim() == 2:
            # (B, num_variables * input_size)
            flat = x
            var_inputs = x.reshape(-1, self.num_variables, x.size(-1) // self.num_variables)
        elif x.dim() == 3:
            # Could be (B, T, num_variables * input_size) or (B, num_variables, input_size)
            flat = x
            per_var_size = x.size(-1) // self.num_variables
            var_inputs = x.reshape(*x.shape[:-1], self.num_variables, per_var_size)
        else:
            flat = x.reshape(*x.shape[:-2], -1)
            var_inputs = x

        # Compute variable importance weights
        weights = self.softmax(self.weight_grn(flat, context))  # (B, [T,] num_variables)

        # Transform each variable
        var_outputs = []
        per_var_size = var_inputs.size(-1)
        for i in range(self.num_variables):
            var_i = var_inputs[..., i, :]  # (B, [T,] per_var_size)
            var_outputs.append(self.variable_grns[i](var_i))  # (B, [T,] hidden_size)

        var_outputs = torch.stack(var_outputs, dim=-2)  # (B, [T,] num_variables, hidden_size)

        # Weighted sum
        weights_expanded = weights.unsqueeze(-1)  # (B, [T,] num_variables, 1)
        selected = (var_outputs * weights_expanded).sum(dim=-2)  # (B, [T,] hidden_size)

        return selected, weights


class StaticCovariateEncoder(nn.Module):
    """Encodes static features into 4 context vectors via VSN + 4 GRNs.

    Produces:
        cs — context for temporal variable selection
        ce — context for static enrichment
        ch — initial hidden state for LSTM
        cc — initial cell state for LSTM
    """

    def __init__(self, n_static: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.vsn = VariableSelectionNetwork(
            input_size=1, num_variables=n_static, hidden_size=hidden_size, dropout=dropout
        )
        self.grn_cs = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.grn_ce = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.grn_ch = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.grn_cc = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

    def forward(self, static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            static: (B, n_static)
        Returns:
            cs, ce, ch, cc: each (B, hidden_size)
        """
        # Reshape to (B, n_static, 1) for per-variable processing
        static_expanded = static.unsqueeze(-1)  # (B, n_static, 1)
        # Flatten for VSN: (B, n_static * 1)
        static_flat = static_expanded.reshape(static.size(0), -1)
        selected, _ = self.vsn(static_flat)  # (B, hidden_size)

        cs = self.grn_cs(selected)
        ce = self.grn_ce(selected)
        ch = self.grn_ch(selected)
        cc = self.grn_cc(selected)

        return cs, ce, ch, cc


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with shared value weights for interpretability.

    Uses separate Q, K projections per head but shares V across heads,
    making the attention weights directly interpretable.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, self.head_dim)  # Shared across heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** 0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k, v: (B, T, hidden_size)
        Returns:
            output: (B, T, hidden_size)
            attn_weights: (B, num_heads, T, T)
        """
        B, T, _ = q.shape

        # Q, K: per-head projections
        Q = self.q_proj(q).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        K = self.k_proj(k).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # V: shared across heads
        V = self.v_proj(v)  # (B, T, head_dim)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to shared values: each head uses same V
        # attn_weights: (B, H, T, T), V: (B, T, D) -> expand V for each head
        V_expanded = V.unsqueeze(1).expand(B, self.num_heads, T, self.head_dim)  # (B, H, T, D)
        head_outputs = torch.matmul(attn_weights, V_expanded)  # (B, H, T, D)

        # Concatenate heads
        output = head_outputs.transpose(1, 2).reshape(B, T, -1)  # (B, T, hidden_size)
        output = self.out_proj(output)

        return output, attn_weights


# ---------------------------------------------------------------------------
# Main TFT Model
# ---------------------------------------------------------------------------


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for energy prediction from temporal + static features.

    Forward pass:
        1. Static encoder -> 4 context vectors (cs, ce, ch, cc)
        2. Temporal VSN (with cs context) -> selected features
        3. LSTM (initialized with ch, cc) -> temporal encoding
        4. Static enrichment GRN (with ce context)
        5. Multi-head attention + gated skip connection
        6. Position-wise GRN
        7. Last timestep -> Linear -> scalar prediction
    """

    def __init__(
        self,
        n_temporal: int,
        n_static: int,
        params: TFTParams,
    ):
        super().__init__()
        self.n_temporal = n_temporal
        self.n_static = n_static
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        # 1. Static covariate encoder
        self.static_encoder = StaticCovariateEncoder(
            n_static, params.hidden_size, params.dropout
        )

        # 2. Temporal input projection (project each timestep to hidden_size)
        self.temporal_proj = nn.Linear(n_temporal, params.hidden_size)

        # 3. Temporal variable selection (operates on projected features)
        # After projection, we treat the hidden_size as a single "variable" for simplicity
        # In full TFT, each raw variable would be projected separately
        self.temporal_grn = GatedResidualNetwork(
            params.hidden_size, params.hidden_size, params.hidden_size,
            dropout=params.dropout, context_size=params.hidden_size,
        )

        # 4. LSTM encoder
        self.lstm = nn.LSTM(
            input_size=params.hidden_size,
            hidden_size=params.hidden_size,
            num_layers=params.num_lstm_layers,
            batch_first=True,
            dropout=params.dropout if params.num_lstm_layers > 1 else 0.0,
        )

        # 5. Post-LSTM gate + skip
        self.post_lstm_glu = GatedLinearUnit(params.hidden_size, params.hidden_size)
        self.post_lstm_norm = nn.LayerNorm(params.hidden_size)

        # 6. Static enrichment GRN
        self.enrichment_grn = GatedResidualNetwork(
            params.hidden_size, params.hidden_size, params.hidden_size,
            dropout=params.dropout, context_size=params.hidden_size,
        )

        # 7. Multi-head attention
        self.attention = InterpretableMultiHeadAttention(
            params.hidden_size, params.num_heads, params.dropout
        )
        self.post_attn_glu = GatedLinearUnit(params.hidden_size, params.hidden_size)
        self.post_attn_norm = nn.LayerNorm(params.hidden_size)

        # 8. Position-wise feed-forward
        self.ff_grn = GatedResidualNetwork(
            params.hidden_size, params.hidden_size, params.hidden_size,
            dropout=params.dropout,
        )

        # 9. Output projection
        self.output_proj = nn.Linear(params.hidden_size, 1)

    def forward(self, temporal: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal: (B, seq_length, n_temporal)
            static:   (B, n_static)
        Returns:
            prediction: (B,)
        """
        # 1. Static encoding -> 4 context vectors
        cs, ce, ch, cc = self.static_encoder(static)

        # 2. Project temporal features and apply temporal GRN with static context
        temporal_proj = self.temporal_proj(temporal)  # (B, T, hidden_size)
        temporal_selected = self.temporal_grn(temporal_proj, cs)  # (B, T, hidden_size)

        # 3. LSTM with static-initialized hidden/cell states
        # Expand for num_layers: (num_layers, B, hidden_size)
        h0 = ch.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        c0 = cc.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        lstm_out, _ = self.lstm(temporal_selected, (h0, c0))  # (B, T, hidden_size)

        # Post-LSTM gate + skip connection
        gated = self.post_lstm_glu(lstm_out)  # (B, T, hidden_size)
        temporal_encoding = self.post_lstm_norm(gated + temporal_selected)  # (B, T, hidden_size)

        # 4. Static enrichment
        enriched = self.enrichment_grn(temporal_encoding, ce)  # (B, T, hidden_size)

        # 5. Multi-head attention + gated skip
        attn_out, _ = self.attention(enriched, enriched, enriched)  # (B, T, hidden_size)
        attn_gated = self.post_attn_glu(attn_out)  # (B, T, hidden_size)
        attn_out = self.post_attn_norm(attn_gated + enriched)  # (B, T, hidden_size)

        # 6. Position-wise feed-forward
        ff_out = self.ff_grn(attn_out)  # (B, T, hidden_size)

        # 7. Take last timestep and project to scalar
        last = ff_out[:, -1, :]  # (B, hidden_size)
        out = self.output_proj(last)  # (B, 1)
        return out.squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EnergyTFTDataset(Dataset):
    """Sliding-window dataset over per-building time series.

    Groups by building (simscode), sorts by time, creates windows of
    ``seq_length`` timesteps.  Target = energy_per_sqft at the last timestep.

    Returns 3-tuples:
        temporal: (seq_length, n_temporal) -- weather + time features per timestep
        static:   (n_static,)             -- building features (constant per building)
        target:   scalar                  -- energy_per_sqft at last timestep
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
        temporal = torch.from_numpy(self.X_temporal[idx])
        static = torch.from_numpy(self.X_static[idx])
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        return temporal, static, target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_model(
    params: TFTParams,
    n_temporal: int,
    n_static: int,
    device: str = "auto",
) -> Tuple[TemporalFusionTransformer, torch.device]:
    """Create TemporalFusionTransformer and move to appropriate device."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = TemporalFusionTransformer(n_temporal, n_static, params).to(device)
    return model, device


def create_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    temporal_cols: list[str],
    static_cols: list[str],
    data_cfg: TFTDataConfig,
) -> Tuple[EnergyTFTDataset, EnergyTFTDataset, dict]:
    """Create windowed train/test datasets. Scaler stats computed from train."""
    train_ds = EnergyTFTDataset(
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
    test_ds = EnergyTFTDataset(
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
    model: TemporalFusionTransformer,
    train_dataset: EnergyTFTDataset,
    test_dataset: EnergyTFTDataset,
    params: TFTParams,
    data_cfg: TFTDataConfig,
    device: torch.device,
    run_dir: Optional[Path] = None,
) -> TemporalFusionTransformer:
    """Train with AdamW, LR scheduler, gradient clipping, early stopping, and TensorBoard logging."""
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

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, params.epochs + 1):
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

            # Weight and gradient histograms every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
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
    model: TemporalFusionTransformer,
    test_dataset: EnergyTFTDataset,
    data_cfg: TFTDataConfig,
    device: torch.device,
    scaler_stats: Optional[dict] = None,
    run_dir: Optional[Path] = None,
    params: Optional[TFTParams] = None,
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
                "num_heads": params.num_heads,
                "num_lstm_layers": params.num_lstm_layers,
                "dropout": params.dropout,
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
    model: TemporalFusionTransformer,
    scaler_stats: dict,
    path: str | Path,
) -> None:
    """Save model state_dict and scaler stats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_stats": scaler_stats,
            "n_temporal_features": model.n_temporal,
            "n_static_features": model.n_static,
        },
        path,
    )


def load_model(
    path: str | Path,
    params: TFTParams,
    n_temporal: int,
    n_static: int,
    device: str = "auto",
) -> Tuple[TemporalFusionTransformer, dict, torch.device]:
    """Load model from checkpoint. Returns (model, scaler_stats, device)."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TemporalFusionTransformer(n_temporal, n_static, params).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["scaler_stats"], device


def get_predictions(
    model: TemporalFusionTransformer,
    df: pd.DataFrame,
    temporal_cols: list[str],
    static_cols: list[str],
    data_cfg: TFTDataConfig,
    device: torch.device,
    scaler_stats: dict,
) -> pd.DataFrame:
    """Add 'predicted' and 'residual' columns to DataFrame.

    Each prediction maps to the last timestep of its window.
    Rows without enough preceding context get NaN.
    """
    ds = EnergyTFTDataset(
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
    # Deduplicate -- keep last prediction for each (simscode, readingtime)
    pred_df = pred_df.drop_duplicates(subset=["simscode", "readingtime"], keep="last")

    result = df.copy()
    result = result.merge(pred_df, on=["simscode", "readingtime"], how="left")
    result["residual"] = result["energy_per_sqft"] - result["predicted"]

    return result
