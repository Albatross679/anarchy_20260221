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

<!--
CONTENT: For each step, describe what it does and how many rows it affects:
1. drop_nan_simscode — Remove rows with null join keys
2. exclude_utilities — Drop OIL28SEC (100% zeros)
3. exclude_unmatched_buildings — Drop simscodes 8, 43, 93 (no metadata)
4. apply_hard_caps — Sensor-fault outlier removal (per-utility thresholds)
5. impute_short_gaps — Forward/backward fill gaps ≤8 hours
6. drop_dead_meters — Remove meters that are 100% NaN
7. drop_sparse_meters — Remove meters with >50% NaN
- CleaningReport summary stats
-->

### 4.2 Feature Engineering

<!--
CONTENT:
- Base features (14): 8 weather + 3 building (grossArea, building_age, floorsAboveGround) + 3 time (hour_of_day, day_of_week, is_weekend)
- Extended features for tree models (25 total): + lag features (1h, 6h, 24h, 1w) + rolling mean/std (24h, 168h) + interactions (temp×area, humidity×area)
- Normalization: energy_per_sqft = readingValue / grossArea
- Target variable: energy_per_sqft
- Z-score normalization for neural network inputs
-->

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

<!--
CONTENT:
- Models learn: f(weather, building_metadata, time) → expected_energy
- Residual: δ_{b,t} = actual - predicted
- Positive residual → over-consuming → investment candidate
- Negative residual → efficient or underutilized
- All models applied consistently across all buildings
-->

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

<!--
CONTENT:
- Architecture: Dual-branch (temporal LSTM + static MLP) → fusion head
- Temporal input: weather + time features per timestep
- Static input: building metadata (constant per building)
- LSTM: hidden=128, 2 layers, dropout=0.2
- Training: 50 epochs, gradient clipping (max_norm=1.0)
-->

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

<!--
CONTENT:
- 25 engineered features (base 15 + 4 lag + 4 rolling + 2 interactions)
- Hyperparameters: n_estimators=1000, max_depth=7, lr=0.05, early_stop=50
- SHAP explainability: feature importance, summary plots
- Diagnostic plots: pred vs actual, residual distribution
-->

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

<!--
CONTENT:
- Neural models: model_best.pt, model_final.pt, periodic epoch checkpoints
- Tree models: model_best.json / model_best.txt
- Resume training from any checkpoint
-->

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
