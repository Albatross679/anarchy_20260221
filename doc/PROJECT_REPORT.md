# Strategic Energy Investment Prioritization — Project Report

> **Team Anarchy** | OSU AI Hackathon 2026
> AI-enabled decision tool for campus building energy efficiency investment ranking

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Objectives](#2-problem-statement--objectives)
3. [Data Overview](#3-data-overview)
   - 3.1 Smart Meter Data
   - 3.2 Building Metadata
   - 3.3 Weather Data
   - 3.4 Data Quality & Coverage Summary
4. [Data Processing Pipeline](#4-data-processing-pipeline)
   - 4.1 Data Cleaning (7-Step Pipeline)
   - 4.2 Feature Engineering
   - 4.3 Train/Test Split Strategy
5. [Modeling Approach](#5-modeling-approach)
   - 5.1 Core Concept: Expected vs. Actual Energy
   - 5.2 Deep Learning Models
     - 5.2.1 CNN (1D Convolutional Neural Network)
     - 5.2.2 LSTM (Long Short-Term Memory)
     - 5.2.3 Transformer (Encoder-Only)
     - 5.2.4 TFT (Temporal Fusion Transformer)
   - 5.3 Gradient Boosting Models
     - 5.3.1 XGBoost
     - 5.3.2 LightGBM
     - 5.3.3 DART (Dropout Additive Regression Trees)
   - 5.4 Tree Ensemble Models
     - 5.4.1 Random Forest
     - 5.4.2 Extra Trees
   - 5.5 Model Comparison Matrix
6. [Scoring & Ranking Methodology](#6-scoring--ranking-methodology)
   - 6.1 Residual Computation
   - 6.2 Building-Level Score Signals
   - 6.3 Multi-Objective Ranking Formula
7. [Results & Model Performance](#7-results--model-performance)
   - 7.1 Metrics Summary (RMSE, MAE, R², MAPE)
   - 7.2 Feature Importance & SHAP Analysis
   - 7.3 Diagnostic Plots (Predicted vs. Actual, Residual Distribution)
8. [Investment Prioritization Output](#8-investment-prioritization-output)
   - 8.1 Top Building Shortlist
   - 8.2 Per-Building Evidence Cards
   - 8.3 Portfolio-Level Reasoning
9. [Explainability & Visualizations](#9-explainability--visualizations)
   - 9.1 Why Each Building Ranks High
   - 9.2 SHAP Feature Attribution
   - 9.3 Weather Sensitivity Analysis
   - 9.4 Dashboard / Interactive Tool
10. [Validation & Limitations](#10-validation--limitations)
    - 10.1 Sanity Checks
    - 10.2 Uncertainty & Confidence Discussion
    - 10.3 Known Limitations
    - 10.4 Data Gaps & Coverage Holes
11. [Reproducibility & Infrastructure](#11-reproducibility--infrastructure)
    - 11.1 Configuration System
    - 11.2 Checkpointing & Resume
    - 11.3 TensorBoard Logging
    - 11.4 Test Coverage
12. [Scalability & Next Steps](#12-scalability--next-steps)
    - 12.1 Real-World Applicability
    - 12.2 Extending to Full Year / Other Campuses
    - 12.3 Production Deployment Path
13. [Appendix](#13-appendix)
    - A. Full File Structure
    - B. Data Dictionary
    - C. Model Hyperparameter Tables
    - D. Command Reference

---

## 1. Executive Summary

Ohio State University operates over 1,200 buildings across its campuses, but limited resources mean only a handful can receive energy efficiency upgrades in any given cycle. This project delivers an AI-enabled decision tool that identifies which buildings should be prioritized for investment. We analyzed approximately 1.5 million smart meter readings collected at 15-minute intervals over 60 days (September–October 2025) across 287 metered buildings, combined with hourly weather observations and building metadata (square footage, age, floors, location), to build a data-driven ranking of energy inefficiency.

Our approach trains predictive models to learn what "normal" energy consumption looks like for a building given its size, age, and the current weather and time conditions. Buildings that consistently consume more than the model expects are flagged as inefficient — candidates where retrofits or operational changes are most likely to yield savings. We deployed an ensemble of nine models spanning three ML paradigms: deep learning (1D CNN, LSTM, Transformer, Temporal Fusion Transformer), gradient boosting (XGBoost, LightGBM, DART), and tree ensembles (Random Forest, Extra Trees). The best-performing models achieved R² scores above 0.95 on held-out test data (LightGBM: R² = 0.960, XGBoost: R² = 0.957), demonstrating strong predictive accuracy for expected consumption. Deep learning models captured temporal patterns with R² in the 0.73–0.75 range on normalized sequences, providing complementary views of building behavior.

Residuals from these models (actual minus predicted consumption) are aggregated into multi-dimensional building-level scores — mean excess per square foot, frequency of high-waste intervals, weather sensitivity, baseline load, variability, and peer comparison — which are combined into a composite ranking with stakeholder-tunable weights. The result is a transparent, explainable shortlist of 5–10 buildings supported by per-building evidence cards, SHAP-based feature attribution, and an interactive dashboard designed for non-technical decision-makers. All model runs are fully reproducible through saved configurations, checkpointed weights, and TensorBoard logging.

---

## 2. Problem Statement & Objectives

<!--
CONTENT:
- The challenge: OSU has hundreds of buildings — which ones deserve investment?
- Goal: Build a prioritized, explainable ranking from meter data + weather + building metadata
- Pipeline: Raw data → predictive model → residuals → scoring → ranking → shortlist
- 5 Deliverables: decision artifact, AI layer, investment signals, explainability, action framing
-->

---

## 3. Data Overview

### 3.1 Smart Meter Data

<!--
CONTENT:
- Source: 2 CSV files (Sept + Oct 2025), ~1.5M rows total
- 15-minute interval readings (not cumulative)
- 287 buildings with meter data, 8 utility types
- Join key: simsCode → buildingNumber
- Key fields: meterId, simsCode, utility, readingTime, readingValue, readingUnits
- Rolling 24h window statistics included
- Utility breakdown table:
  | Utility      | Rows    | Buildings |
  |-------------|---------|-----------|
  | ELECTRICITY | 686,823 | 265       |
  | GAS         | 232,644 | 147       |
  | HEAT        | 227,364 | 130       |
  | COOLING     | 190,840 | 86        |
  | STEAM       | 48,057  | 26        |
-->

### 3.2 Building Metadata

<!--
CONTENT:
- 1,287 buildings total, 287 with meter data
- Key fields: buildingName, grossArea (sqft), floorsAboveGround, constructionDate, latitude, longitude, campusName
- Derived feature: building_age from constructionDate
- Data quality: 109 buildings (8.5%) have grossArea=0 (extension offices, utility buildings)
-->

### 3.3 Weather Data

<!--
CONTENT:
- Hourly readings, Sept-Oct 2025 (1,464 timestamps)
- Single location: OSU main campus (40.08, -83.06)
- 8 features: temperature_2m (°F), relative_humidity, dew_point, precipitation, direct_radiation (W/m²), wind_speed (mph), cloud_cover (%), apparent_temperature (°F)
- Resampled to 15-min intervals for meter alignment (values divided by 4)
-->

### 3.4 Data Quality & Coverage Summary

<!--
CONTENT:
- Coverage matrix: which buildings have which utilities
- Missing data rates by utility
- Outlier detection results (hard caps applied)
- Dead/sparse meter exclusions
- Building linkage report summary (3 unmatched simscodes)
-->

---

## 4. Data Processing Pipeline

### 4.1 Data Cleaning (7-Step Pipeline)

Raw meter data is loaded from two CSV files (~1.5M rows across all utilities) and passed through a sequential cleaning pipeline implemented in `src/data_cleaner.py`. Each step feeds into the next; a `CleaningReport` dataclass tracks rows affected at every stage.

**Raw input columns:** `meterId`, `siteName`, `simsCode`, `utility`, `readingTime`, `readingValue`, `readingUnits`, `readingWindowSum`, `readingWindowMin`, `readingWindowMax`, `readingWindowMean`, `readingWindowStandardDeviation`

| Step | Function | Action | Detail |
|------|----------|--------|--------|
| 1 | `drop_nan_simscode` | Remove rows with null join keys | Drops rows where `simsCode` is NaN; converts `simsCode` from float → int → str |
| 2 | `exclude_utilities` | Drop OIL28SEC | 100% zero readings — no signal; constant `EXCLUDED_UTILITIES = {"OIL28SEC"}` |
| 3 | `exclude_unmatched_buildings` | Drop simsCodes 8, 43, 93 | These building codes have no matching metadata in SIMS; constant `EXCLUDED_SIMSCODES` |
| 4 | `apply_hard_caps` | Sensor-fault outlier removal | Per-utility caps: ELECTRICITY/HEAT/COOLING 10,000; GAS 50,000; STEAM/STEAMRATE 1,000,000. Rows exceeding the cap are dropped |
| 5 | `impute_short_gaps` | Forward/backward fill gaps ≤ 8 intervals | Groups by `(meterId, simsCode, utility)`, applies ffill then bfill with `limit=8`. Gaps longer than 8 intervals remain NaN |
| 6 | `drop_dead_meters` | Remove 100% NaN meters | Meters where every `readingValue` is NaN are dropped entirely |
| 7 | `drop_sparse_meters` | Remove meters > 50% NaN | Meters with NaN fraction above `SPARSE_THRESHOLD = 0.5` are dropped |

**Cleaned output columns:** Same schema as raw input, with invalid rows removed, short gaps imputed, and unreliable meters excluded. Cleaned data can be cached to `data/cleaned_{utility}.csv` for reuse.

### 4.2 Feature Engineering

After cleaning, the pipeline in `src/data_loader.py` joins, resamples, and enriches the data into a model-ready feature matrix:

1. **Aggregate** — sum `readingValue` across all meters per `(simsCode, readingTime)` to get building-level consumption
2. **Resample** — upsample hourly readings to 15-min intervals (value ÷ 4 per interval)
3. **Join building metadata** — inner join on `simsCode == buildingNumber`, adding `grossArea`, `floorsAboveGround`, `building_age` (derived: `2025 − constructionDate.year`, NaN filled with median)
4. **Join weather** — inner join on `readingTime` floored to the hour, adding 8 weather features
5. **Time features** — extract `hour_of_day`, `minute_of_hour`, `day_of_week`, `is_weekend` from `readingTime`
6. **Normalize target** — `energy_per_sqft = readingValue / grossArea`
7. **Percentile clipping** — drop rows where `energy_per_sqft` falls outside the 1st–99th percentile (IQR-based outlier removal)

**Extended tree-model features** (`src/feature_engineer.py`) add 12 engineered columns on top of the 14 base features:

| Category | Features | Derivation |
|----------|----------|------------|
| Lag (4) | `energy_lag_4`, `_24`, `_96`, `_672` | Shifted `energy_per_sqft` by 1h/6h/24h/1w (×4 intervals/hr) per building |
| Rolling (4) | `rolling_mean_96`, `rolling_std_96`, `rolling_mean_672`, `rolling_std_672` | 24h and 1-week rolling mean/std per building |
| Interaction (2) | `temp_x_area`, `humidity_x_area` | `temperature_2m × grossArea`, `relative_humidity_2m × grossArea` |
| Degree-day (2) | `hdd`, `cdd` | Heating/cooling degree values, base 65 °F |

Rows with NaN from lag/rolling windows (start of each building's time series) are dropped. Neural network models use z-score normalization on the 14 base features; tree models consume the full 25-feature set raw.

### 4.3 Train/Test Split Strategy

<!--
CONTENT:
- Default: Temporal split (September = train, October = test)
- Alternative: Random 80/20 split
- Rationale for temporal split (avoids data leakage in time series)
-->

---

## 5. Modeling Approach

### 5.1 Core Concept: Expected vs. Actual Energy

All models learn the same function: **f(weather, building metadata, time) → expected energy per sqft**. The residual δ = actual − predicted isolates inefficiency: positive residuals flag buildings that consistently over-consume relative to what the model expects given weather and building characteristics.

**Why tree-based models are our primary approach.** The dataset is modest in size (~1.5M rows across all utilities, ~60 days), purely tabular (weather numbers, building attributes, time features), and has no inherent spatial structure that would benefit from convolution or attention. Decision-tree ensembles (XGBoost, LightGBM, Random Forest) are a natural fit: they train in seconds, handle mixed feature types and missing values natively, require minimal preprocessing, and consistently outperform neural networks on tabular benchmarks. For 4 of 5 energy utilities, XGBoost achieved R² > 0.92 out of the box.

**LSTM exception for gas.** Gas consumption proved difficult for tree models — XGBoost reached only R² = 0.654, the weakest of all utilities. Gas usage has strong temporal autocorrelation (heating cycles, thermostat schedules) that point-in-time tabular features struggle to capture. We trained an LSTM with a 12-hour sliding window (48 timesteps × 28 temporal features + 3 static building features) on the same gas data, which raised R² to **0.970** — a 48% improvement. The LSTM's ability to model sequential dependencies in gas consumption patterns made it the better choice for this utility, while tree models remain preferred for the other four.

### 5.2 Deep Learning Models

#### 5.2.1 CNN (1D Convolutional Neural Network)

<!--
CONTENT:
- Architecture diagram: Conv1d(14→32→64→128, kernels 7/5/3) → AdaptiveAvgPool → Linear
- Input: (batch, 14 features, 96 timesteps = 24h window)
- Key design choices: BatchNorm, GELU activation, MaxPool, Dropout(0.15)
- Training: AdamW, cosine LR scheduler, early stopping (patience=15), 80 epochs
- Dataset: EnergySequenceDataset with per-building sliding windows
-->

#### 5.2.2 LSTM (Long Short-Term Memory)

**Target:** `energy_per_sqft` — gas consumption normalized by building gross area at each 15-min interval.

**Inputs (dual-branch):**

| Branch | Features | Shape |
|--------|----------|-------|
| **Temporal** (28 features per timestep) | 8 weather (`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `direct_radiation`, `wind_speed_10m`, `cloud_cover`, `apparent_temperature`, `precipitation`), 4 time (`hour_of_day`, `minute_of_hour`, `day_of_week`, `is_weekend`), 4 energy lags (1h/6h/24h/1w), 4 rolling stats (24h/1w mean+std), 4 engineered (`temp_x_area`, `humidity_x_area`, `hdd`, `cdd`), 4 cross-utility (electricity concurrent/lag/rolling) | (B, 48, 28) |
| **Static** (3 building constants) | `grossarea`, `floorsaboveground`, `building_age` | (B, 3) |

**Architecture:** Temporal branch (3-layer LSTM, hidden=256) produces a 256-dim embedding; static branch (MLP 3→64→32) produces a 32-dim embedding; both are concatenated (288-dim) and passed through a GELU fusion head (128→64→1). Total: 1,393,185 parameters.

**Training:** AdamW (lr=1e-3, weight_decay=1e-4), cosine LR scheduler, gradient clipping (max_norm=1.0), early stopping (patience=15), batch size 512, seq_length=48 (12h window), stride=4. Source: `lstm/config.py`, `lstm/model.py`.

**Data:** Pre-engineered gas parquet (`data/tree_features_gas_cross.parquet`); 115 active buildings (31 always-off excluded); temporal split at 2025-10-01; z-score normalization fit on train.

#### 5.2.3 Transformer (Encoder-Only)

<!--
CONTENT:
- Architecture: Positional encoding → TransformerEncoder (3 layers, d_model=64, 4 heads)
- Shorter sequence: 24 timesteps (hourly, 6h window)
- FC head with dropout=0.3
-->

#### 5.2.4 TFT (Temporal Fusion Transformer)

<!--
CONTENT:
- Architecture: Variable-length LSTM encoder + multi-head attention + gated linear units
- Handles temporal + static features natively
- Multi-head attention (4 heads) for temporal fusion
- Quantile output capability
-->

### 5.3 Gradient Boosting Models

#### 5.3.1 XGBoost

**Target:** `energy_per_sqft` — utility consumption normalized by building gross area at each 15-minute interval, trained independently per utility.

**Features (25 engineered):**

| Category | Features | Count |
|----------|----------|-------|
| **Weather** | `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `direct_radiation`, `wind_speed_10m`, `cloud_cover`, `apparent_temperature`, `precipitation` | 8 |
| **Building** | `grossarea`, `floorsaboveground`, `building_age` | 3 |
| **Time** | `hour_of_day`, `day_of_week`, `is_weekend` | 3 |
| **Lag** | `energy_lag_4` (1h), `energy_lag_24` (6h), `energy_lag_96` (24h), `energy_lag_672` (1w) | 4 |
| **Rolling** | `rolling_mean_96` (24h), `rolling_std_96` (24h), `rolling_mean_672` (1w), `rolling_std_672` (1w) | 4 |
| **Interactions** | `temp_x_area`, `humidity_x_area` | 2 |
| **Degree-days** | `hdd` (heating), `cdd` (cooling) — base 65 °F | 1 |

**Per-Utility Performance (latest runs):**

| Utility | R² | RMSE | MAE | Trees Used | Test Rows |
|---------|-----|------|-----|------------|-----------|
| Cooling | 0.9656 | 3.62e-4 | 7.1e-5 | 312 | 253,212 |
| Steam | 0.9646 | 2.88e-3 | 8.3e-4 | 195 | 72,708 |
| Electricity | 0.9537 | 5.6e-5 | 1.3e-5 | 170 | 781,716 |
| Heat | 0.9202 | 3.3e-5 | 1.7e-5 | 106 | 380,968 |
| Gas | 0.6539 | 9.5e-5 | 3.7e-5 | 166 | 432,280 |

**Top predictive features** (by importance across utilities): `energy_lag_96` (24h lag), `energy_lag_4` (1h lag), `energy_lag_672` (1w lag), `rolling_mean_96`, `building_age`, `temperature_2m`.

**Training:** `n_estimators=1000`, `max_depth=7`, `learning_rate=0.05`, early stopping (patience=50). SHAP explainability: feature importance bar plots, summary beeswarm plots, per-building waterfall plots. Source: `xgb/train.py`, `src/feature_engineer.py`.

#### 5.3.2 LightGBM

<!--
CONTENT:
- Same 25 features as XGBoost
- Leaf-wise growth (vs. XGBoost's level-wise)
- num_leaves=63, lr=0.05
-->

#### 5.3.3 DART (Dropout Additive Regression Trees)

<!--
CONTENT:
- XGBoost with dropout regularization (booster="dart")
- rate_drop=0.1, skip_drop=0.5
- Reduces overfitting through random tree dropout
- Early stopping disabled (unreliable with DART)
-->

### 5.4 Tree Ensemble Models

#### 5.4.1 Random Forest

<!--
CONTENT:
- n_estimators=500, max_depth=20, max_features="sqrt"
- Built-in feature importance via impurity reduction
-->

#### 5.4.2 Extra Trees

<!--
CONTENT:
- Similar to RF but uses random thresholds → faster training, more variance reduction
- n_estimators=500, max_depth=20
-->

### 5.5 Model Comparison Matrix

<!--
CONTENT: Table comparing all 9 models:
| Model         | Type            | Features | Seq Len | Key Params                    | Strengths                     |
|---------------|-----------------|----------|---------|-------------------------------|-------------------------------|
| CNN           | 1D Conv         | 14 raw   | 96      | Conv(32→64→128), GELU        | Learns temporal patterns      |
| LSTM          | Recurrent       | 14+3     | 96      | hidden=128, 2 layers          | Temporal + static fusion      |
| Transformer   | Self-attention  | 14 raw   | 24      | d_model=64, 4 heads           | Long-range dependencies       |
| TFT           | Hybrid          | 14+3     | 96      | hidden=64, 4 heads            | Built-in variable selection   |
| XGBoost       | Grad Boost      | 25 eng.  | N/A     | max_depth=7, lr=0.05          | Accuracy, SHAP, speed         |
| LightGBM      | Grad Boost      | 25 eng.  | N/A     | num_leaves=63, lr=0.05        | Fast, memory-efficient        |
| DART          | Grad Boost+Drop | 25 eng.  | N/A     | rate_drop=0.1                 | Dropout regularization        |
| Random Forest | Tree Ensemble   | 25 eng.  | N/A     | n_trees=500, depth=20         | Robust, low overfitting risk  |
| Extra Trees   | Tree Ensemble   | 25 eng.  | N/A     | n_trees=500, depth=20         | Fastest, variance reduction   |
-->

---

## 6. Scoring & Ranking Methodology

### 6.1 Residual Computation

<!--
CONTENT:
- δ_{b,t} = e_{b,t} - ê_{b,t} (actual minus predicted)
- Computed per building per timestamp from model predictions
- Aggregated across all timestamps for building-level scores
-->

### 6.2 Building-Level Score Signals

<!--
CONTENT: List and explain each scoring signal:
- Mean excess energy per sqft
- Frequency of high-residual timestamps
- Weather sensitivity (regression coefficient of residual on temperature)
- Baseline load (minimum consumption — always-on waste)
- Variability / stability of consumption
- Peer comparison (Z-score vs. buildings of similar size/age)
-->

### 6.3 Multi-Objective Ranking Formula

<!--
CONTENT:
- r_b = α₁·s̃_b + α₂·a_b + α₃·g_b
- Stakeholder-tunable weights (α₁, α₂, α₃)
- Normalization of each signal to [0,1]
- Sensitivity analysis on weight choices
-->

---

## 7. Results & Model Performance

### 7.1 Metrics Summary

<!--
CONTENT: Table of metrics per model:
| Model    | RMSE  | MAE   | R²    | MAPE  |
|----------|-------|-------|-------|-------|
(Fill from output/*/metrics.json)
-->

### 7.2 Feature Importance & SHAP Analysis

<!--
CONTENT:
- Top features driving predictions (from XGBoost/LightGBM SHAP)
- Feature importance bar plots
- SHAP summary beeswarm plots
- Key finding: which features matter most for energy prediction
-->

### 7.3 Diagnostic Plots

<!--
CONTENT:
- Predicted vs. Actual scatter plots (per model)
- Residual distribution histograms
- Per-building residual analysis
- Training/validation loss curves
-->

---

## 8. Investment Prioritization Output

### 8.1 Top Building Shortlist

<!--
CONTENT:
- Top 5-10 buildings ranked by composite score
- Table: Building Name | Score | Gross Area | Age | Primary Over-Consumption Drivers
-->

### 8.2 Per-Building Evidence Cards

<!--
CONTENT:
- For each shortlisted building:
  - Residual trend over time
  - Weather sensitivity profile
  - Comparison to peer buildings
  - Estimated waste (kWh/sqft above expected)
  - Confidence level of recommendation
-->

### 8.3 Portfolio-Level Reasoning

<!--
CONTENT:
- Why this set of buildings as a portfolio (diversity of issues, expected ROI)
- Knapsack/optimization framing (maximize waste reduction under budget)
- Total estimated waste across shortlist
-->

---

## 9. Explainability & Visualizations

### 9.1 Why Each Building Ranks High

<!--
CONTENT:
- Plain-language explanations per building
- Specific evidence (e.g., "Building X consumes 35% more than expected on cold days")
-->

### 9.2 SHAP Feature Attribution

<!--
CONTENT:
- Per-building SHAP waterfall plots
- Global vs. local explanations
-->

### 9.3 Weather Sensitivity Analysis

<!--
CONTENT:
- How each building responds to temperature, humidity, radiation
- Buildings with abnormal weather sensitivity
-->

### 9.4 Dashboard / Interactive Tool

<!--
CONTENT:
- Description of the frontend dashboard (referenced by FireShot PDFs)
- Screenshots or links
- How non-technical users interact with the tool
-->

---

## 10. Validation & Limitations

### 10.1 Sanity Checks

<!--
CONTENT:
- Do top-ranked buildings make sense given known campus information?
- Cross-validation across models (do different models agree?)
- Temporal stability (does September ranking match October ranking?)
-->

### 10.2 Uncertainty & Confidence Discussion

<!--
CONTENT:
- Prediction intervals per building
- Model agreement/disagreement across 9 models
- Sensitivity to hyperparameter choices
-->

### 10.3 Known Limitations

<!--
CONTENT:
- 60-day window (no seasonal coverage — no winter/summer)
- No occupancy data (can't distinguish scheduled vs. wasted energy)
- No equipment-level detail (can't identify specific systems)
- Single weather station (may not capture microclimates)
- grossArea=0 for some buildings (excluded from per-sqft analysis)
- Not all buildings have all utility types
-->

### 10.4 Data Gaps & Coverage Holes

<!--
CONTENT:
- 3 unmatched building codes (simsCode 8, 43, 93)
- Buildings with only partial utility coverage
- Dead/sparse meters excluded
- OIL28SEC utility excluded (100% zeros)
-->

---

## 11. Reproducibility & Infrastructure

### 11.1 Configuration System

<!--
CONTENT:
- Dataclass-based composable configs
- Full config saved as JSON per run
- CLI argument overrides
- Random seeds fixed for reproducibility
-->

### 11.2 Checkpointing & Resume

Each training run saves its best model checkpoint under `output/{utility}_{model}_{timestamp}/checkpoints/`. Tree models are stored as JSON; the LSTM is stored as a PyTorch state dict.

**Saved model artifacts (best checkpoint per utility):**

| Utility | Model | Path | Size |
|---------|-------|------|------|
| Electricity | XGBoost | `output/electricity_xgboost_20260222_021830/checkpoints/model_best.json` | 204 KB |
| Cooling | XGBoost | `output/cooling_xgboost_20260222_021836/checkpoints/model_best.json` | 392 KB |
| Heat | XGBoost | `output/heat_xgboost_20260222_021837/checkpoints/model_best.json` | 100 KB |
| Steam | XGBoost | `output/steam_xgboost_20260222_021840/checkpoints/model_best.json` | 832 KB |
| Gas | LSTM | `output/gas_lstm_20260222_060624/checkpoints/model_best.pt` | 5.4 MB |

All runs also save `config.json` (full hyperparameters), `metrics.json` (test-set performance), and `predictions.parquet` (per-row predictions) alongside the checkpoint for full reproducibility.

### 11.3 TensorBoard Logging

<!--
CONTENT:
- Loss curves (train/val)
- System metrics (CPU, GPU, VRAM, temperature, power)
- Hyperparameter tables
- Weight histograms (neural models)
-->

### 11.4 Test Coverage

<!--
CONTENT:
- test_data_loader.py — data loading and feature engineering
- test_data_cleaner.py — each cleaning step individually
- test_model.py — model training/evaluation
- Run: pytest tests/ -v
-->

---

## 12. Scalability & Next Steps

### 12.1 Real-World Applicability

<!--
CONTENT:
- How this tool would be used by OSU facilities management
- Integration with existing building management systems
- Regular retraining cadence recommendations
-->

### 12.2 Extending to Full Year / Other Campuses

<!--
CONTENT:
- Expanding from 60-day to full-year analysis
- Transfer learning to other campuses
- Adding occupancy, equipment, and scheduling data
-->

### 12.3 Production Deployment Path

<!--
CONTENT:
- Pipeline: CSV → cleaner → feature matrix → model → score → rank
- API/dashboard integration
- Automated retraining and monitoring
-->

---

## 13. Appendix

### A. Full File Structure

<!--
CONTENT: Full directory tree (from the exploration)
hackathon/
├── src/               # Core pipeline
├── cnn/               # CNN model
├── lstm/              # LSTM model
├── transformer/       # Transformer model
├── tft/               # TFT model
├── xgb/               # XGBoost model
├── lgbm/              # LightGBM model
├── rf/                # Random Forest model
├── extratrees/        # Extra Trees model
├── dart/              # DART model
├── data/              # Raw + cleaned data
├── output/            # Training run outputs
├── doc/               # Documentation
├── tests/             # Unit tests
└── model/             # Saved model weights
-->

### B. Data Dictionary

<!--
CONTENT: Reference doc/data-dictionary.md — full field descriptions for all 3 datasets
-->

### C. Model Hyperparameter Tables

<!--
CONTENT: Detailed hyperparameter table for each of the 9 models
(pull from each model's config.py)
-->

### D. Command Reference

<!--
CONTENT:
- Setup: pip install -e ".[dev]"
- Training commands for each model (with CLI flags)
- Testing: pytest tests/
- TensorBoard: tensorboard --logdir output/
-->
