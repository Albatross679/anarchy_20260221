# CNN Energy Consumption Model — Input / Output Specification

## Overview

1D Convolutional Neural Network that predicts per-building energy consumption (energy per square foot) from sliding windows of temporal, weather, and building features. Residuals (actual - predicted) feed into the downstream scoring and investment ranking pipeline.

## Input

### Raw Data Sources

| Source | File(s) | Join Key |
|--------|---------|----------|
| Smart meter data | `data/meter-data-sept-2025.csv`, `data/meter-data-oct-2025.csv` | `simsCode` -> `buildingNumber` |
| Building metadata | `data/building_metadata.csv` | `buildingNumber` |
| Weather data | `data/weather-sept-oct-2025.csv` | `date` -> `readingTime` |

### Feature Columns (14 total)

**Weather features (8):**
`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `direct_radiation`, `wind_speed_10m`, `cloud_cover`, `apparent_temperature`, `precipitation`

**Building features (3):**
`grossarea`, `floorsaboveground`, `building_age`

**Temporal features (3):**
`hour_of_day`, `day_of_week`, `is_weekend`

### Model Input Tensor

| Property | Value |
|----------|-------|
| Shape | `(batch_size, 14, 24)` — `(B, n_features, seq_length)` |
| Format | `torch.float32` |
| Normalization | Z-score normalized per feature (mean/std computed from training set) |

Each sample is a **sliding window** of 24 consecutive hourly timesteps across all 14 features, constructed per-building and sorted by time. Windows are generated with a configurable stride (default 1).

### Target

| Property | Value |
|----------|-------|
| Field | `energy_per_sqft` at the last timestep of the window |
| Normalization | Z-score normalized (mean/std from training set) |
| Utility | `ELECTRICITY` (default; configurable via `--utility`) |

### Train/Test Split

Default: **temporal split** — train on September 2025, test on October 2025 (split date: `2025-10-01`). Optional random 80/20 split via `--no-temporal-split`.

## Architecture

```
Input (B, 14, 24)
  -> Conv1d(14, 32, k=3, pad=1) -> BatchNorm -> ReLU -> MaxPool1d(2) -> Dropout(0.2)
  -> Conv1d(32, 64, k=3, pad=1) -> BatchNorm -> ReLU -> MaxPool1d(2) -> Dropout(0.2)
  -> Conv1d(64, 128, k=3, pad=1) -> BatchNorm -> ReLU -> MaxPool1d(2) -> Dropout(0.2)
  -> AdaptiveAvgPool1d(1) -> Flatten
  -> Linear(128, 64) -> ReLU -> Dropout(0.3)
  -> Linear(64, 1)
Output (B,)
```

### Training Configuration

| Parameter | Default |
|-----------|---------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| LR scheduler | Cosine annealing |
| Epochs | 50 |
| Batch size | 256 |
| Early stopping | Patience 10 (on val loss) |
| Loss function | MSE |

## Output

### 1. Predictions DataFrame (`predictions.parquet`)

Columns added to the original feature matrix:

| Column | Type | Description |
|--------|------|-------------|
| `predicted` | float | Model's predicted `energy_per_sqft` (denormalized to original units) |
| `residual` | float | `energy_per_sqft - predicted` (positive = over-consuming) |

All other original columns (meter data, weather, building metadata, time features) are preserved. Rows without enough preceding context for a full window get `NaN` in `predicted` and `residual`.

### 2. Evaluation Metrics (logged to TensorBoard + console)

| Metric | Description |
|--------|-------------|
| RMSE | Root mean squared error (denormalized units) |
| MAE | Mean absolute error (denormalized units) |
| R² | Coefficient of determination |
| MAPE | Mean absolute percentage error (excludes zero targets) |

### 3. TensorBoard Logs (`tensorboard/`)

- **Scalars:** `loss/train`, `loss/val`, `lr`, `metrics/val_rmse`, `metrics/val_mae`, `metrics/val_r2`, `time/epoch_seconds`
- **Histograms:** Weight and gradient distributions every 5 epochs
- **Figures:** Predicted vs actual scatter, residual distribution
- **HParams:** Full hyperparameter table linked to final metrics

### 4. Checkpoints (`checkpoints/`)

| File | Description |
|------|-------------|
| `model_best.pt` | Best validation loss weights |
| `model_final.pt` | Final epoch weights |
| `epoch_N.pt` | Periodic snapshots (every 5 epochs) |

Each checkpoint contains: `model_state_dict`, `scaler_stats`, `n_features`, `seq_length`.

### 5. Config (`config.json`)

Full experiment configuration serialized as JSON for reproducibility.

## CLI Usage

```bash
python cnn/train.py                                # defaults
python cnn/train.py --name exp1 --seq-length 48    # custom window
python cnn/train.py --utility STEAM --epochs 100   # different utility
python cnn/train.py --lr 0.0005 --batch-size 512   # tuning
python cnn/train.py --no-temporal-split            # random split
```

## Downstream Usage

The `predictions.parquet` output feeds into `src/scoring.py`, which aggregates per-building residuals into investment priority scores. Buildings with consistently positive residuals (actual > expected) are flagged as over-consumers and candidates for energy efficiency investment.
