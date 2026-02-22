# XGBoost Energy Consumption Model — Input / Output Specification

## Overview

Gradient-boosted tree ensemble (XGBRegressor) that predicts per-building energy consumption (energy per square foot) from tabular weather, building, temporal, and engineered lag/rolling features. Residuals (actual - predicted) feed into the downstream scoring and investment ranking pipeline.

## Input

### Raw Data Sources

| Source | File(s) | Join Key |
|--------|---------|----------|
| Smart meter data | `data/meter-data-sept-2025.csv`, `data/meter-data-oct-2025.csv` | `simsCode` -> `buildingNumber` |
| Building metadata | `data/building_metadata.csv` | `buildingNumber` |
| Weather data | `data/weather-sept-oct-2025.csv` | `date` -> `readingTime` |

### Base Feature Columns (15 total)

**Weather features (8):**
`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `direct_radiation`, `wind_speed_10m`, `cloud_cover`, `apparent_temperature`, `precipitation`

**Building features (3):**
`grossarea`, `floorsaboveground`, `building_age`

**Temporal features (4):**
`hour_of_day`, `minute_of_hour`, `day_of_week`, `is_weekend`

### Engineered Features

Feature engineering adds lag, rolling, and interaction columns per building (grouped by `simscode`, sorted by `readingtime`). Rows with NaN from lag/rolling computation are dropped.

**Lag features (4) — energy_per_sqft shifted by N 15-min intervals:**

| Column | Lag (hours) | Intervals |
|--------|-------------|-----------|
| `energy_lag_4` | 1h | 4 |
| `energy_lag_24` | 6h | 24 |
| `energy_lag_96` | 24h | 96 |
| `energy_lag_672` | 168h (1 week) | 672 |

**Rolling statistics (4) — per-building rolling mean and std of energy_per_sqft:**

| Column | Window (hours) | Intervals |
|--------|----------------|-----------|
| `rolling_mean_96` | 24h | 96 |
| `rolling_std_96` | 24h | 96 |
| `rolling_mean_672` | 168h (1 week) | 672 |
| `rolling_std_672` | 168h (1 week) | 672 |

**Interaction features (2):**
`temp_x_area` (temperature_2m × grossarea), `humidity_x_area` (relative_humidity_2m × grossarea)

### Total Feature Count

15 base + 4 lag + 4 rolling + 2 interaction = **25 features**

### Model Input

| Property | Value |
|----------|-------|
| Format | `pandas.DataFrame` (tabular, one row per 15-min interval per building) |
| Dtype | `float64` |
| Normalization | None — XGBoost is tree-based and scale-invariant |

### Target

| Property | Value |
|----------|-------|
| Field | `energy_per_sqft` |
| Normalization | None (raw units) |
| Utility | `ELECTRICITY` (default; configurable via `--utility`) |

### Train/Test Split

Default: **temporal split** — train on September 2025, test on October 2025 (split date: `2025-10-01`). Optional random 80/20 split via `--no-temporal-split`.

## Model

### Algorithm

XGBRegressor (gradient-boosted decision trees) with histogram-based splitting.

### Hyperparameters

| Parameter | Default |
|-----------|---------|
| n_estimators | 1000 |
| max_depth | 7 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 5 |
| reg_alpha (L1) | 0.1 |
| reg_lambda (L2) | 1.0 |
| tree_method | `hist` |
| eval_metric | `rmse` |
| early_stopping_rounds | 50 |

## Output

### 1. Predictions DataFrame (`predictions.parquet`)

Columns added to the full feature-engineered DataFrame:

| Column | Type | Description |
|--------|------|-------------|
| `predicted` | float | Model's predicted `energy_per_sqft` |
| `residual` | float | `energy_per_sqft - predicted` (positive = over-consuming) |

All original columns (meter data, weather, building metadata, time features, engineered features) are preserved.

### 2. Evaluation Metrics (`metrics.json`)

| Metric | Description |
|--------|-------------|
| RMSE | Root mean squared error |
| MAE | Mean absolute error |
| R² | Coefficient of determination |
| MAPE | Mean absolute percentage error (excludes zero targets) |
| n_trees_used | Number of boosting rounds (may be < n_estimators if early stopped) |
| top_features | Top 10 features by importance (gain) |

### 3. Diagnostic Plots (`plots/`)

| File | Description |
|------|-------------|
| `feature_importance.png` | Top 20 features by gain-based importance |
| `pred_vs_actual.png` | Scatter plot of predicted vs actual with R² annotation |
| `residual_dist.png` | Histogram of residuals with mean/std annotation |
| `shap_summary.png` | SHAP beeswarm plot showing feature impact on predictions |
| `shap_importance.png` | SHAP bar plot of mean absolute SHAP values |

### 4. TensorBoard Logs (`tensorboard/`)

- **Scalars:** `validation_0/rmse` (train), `validation_1/rmse` (val), `loss/train`, `loss/val`, `metrics/val_rmse`, `metrics/val_mae`, `metrics/val_r2`, `time/wall_clock_seconds`, `time/round_seconds`
- **System:** CPU, GPU utilization, VRAM, temperature, power (every 10 rounds)
- **Text:** Hyperparameter table at step 0

### 5. Checkpoint (`checkpoints/model_best.json`)

XGBoost native JSON format. Load with:

```python
from xgb.model import load_model
model = load_model("output/<run>/checkpoints/model_best.json")
```

### 6. Config (`config.json`)

Full experiment configuration serialized as JSON for reproducibility.

## CLI Usage

```bash
python xgb/train.py                                    # defaults
python xgb/train.py --name exp1 --n-estimators 500     # overrides
python xgb/train.py --utility STEAM --max-depth 8      # different utility
python xgb/train.py --lr 0.01 --no-temporal-split      # tuning
python xgb/train.py --no-early-stop                    # disable early stopping
```

## Downstream Usage

The `predictions.parquet` output feeds into `src/scoring.py`, which aggregates per-building residuals into investment priority scores. Buildings with consistently positive residuals (actual > expected) are flagged as over-consumers and candidates for energy efficiency investment.
