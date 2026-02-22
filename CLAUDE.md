# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OSU AI Hackathon — **Team Anarchy**. Strategic Energy Investment Prioritization: an AI-enabled decision tool that analyzes ~60 days of campus building energy meter data, weather, and building metadata to rank buildings for energy efficiency investment.

**Pipeline**: Raw data -> predictive model -> residuals (actual - expected) -> building-level scores -> ranking/optimization -> investment shortlist (top 5-10 buildings)

## Deliverables

1. **Decision-support artifact** — interactive dashboard or ranking tool for non-technical decision-makers
2. **AI/model reasoning layer** — regression/ML model applied consistently across all buildings
3. **Investment signals** — multiple data-derived signals with explainable weighting
4. **Explainability** — per-building evidence, uncertainty/confidence, stated limitations
5. **Action framing** — next steps, scalability reflection

## Commands

### Setup
```bash
pip install -e ".[dev]"  # uses default conda env (Lightning Studio)
```

### Run tests
```bash
pytest tests/
pytest tests/test_scoring.py -v          # single test file
pytest tests/test_scoring.py::test_name  # single test
```

### Pre-compute tree features
```bash
python src/prepare_tree_features.py                          # all utilities
python src/prepare_tree_features.py --utilities ELECTRICITY  # single utility
```
Outputs `data/tree_features_{utility}.parquet` + `data/tree_features_manifest.json`.

Then train any tree model with `--precomputed` to skip the data pipeline:
```bash
python xgb/train.py --precomputed --utility ELECTRICITY
```

## Architecture

```
src/
├── data_loader.py           # Load and preprocess energy, weather, building data
├── feature_engineer.py      # Shared tree-model feature engineering (lags, rolling, interactions)
├── prepare_tree_features.py # CLI: pre-compute tree features to parquet (run once, train many)
├── model.py                 # Predictive model f_θ(w_t, m_b, t) for expected consumption
├── scoring.py               # Compute residuals δ_{b,t} and building-level scores s_b
├── anomaly.py               # Anomaly detection (z-score, isolation forest)
├── optimizer.py             # 0-1 knapsack / ranking for investment selection
├── explainer.py             # Explainability and visualization of rankings
└── utils/                   # Shared helpers
```

- `data/` — energy meter CSVs, weather data, building metadata
- `output/` — generated rankings, reports, plots (named `{utility}_{model}_{timestamp}`, e.g. `electricity_xgboost_20260222_001425`)
- `model/` — saved model weights
- `doc/` — info packet, challenge documentation, and model specs
  - `doc/CNN_MODEL.md` — CNN model input/output specification
  - `doc/XGB_MODEL.md` — XGBoost model input/output specification

## Data Schema

### Smart Meter Data (CSV, 15-min intervals, ~60 days from 2025)
- Join key: `simsCode` (links to building metadata `buildingNumber`)
- Key fields: `meterId`, `siteName`, `simsCode`, `utility`, `readingTime`, `readingValue`, `readingUnits`
- Rolling 24h window stats: `readingWindowSum`, `readingWindowMin`, `readingWindowMax`, `readingWindowMean`, `readingWindowStandardDeviation`
- `readingValue` is per-interval (not cumulative) — sum intervals for daily/monthly totals
- Utilities: ELECTRICITY, ELECTRICAL_POWER, GAS, HEAT, STEAM, STEAMRATE, COOLING, COOLING_POWER, OIL28SEC
- **Energy vs Power**: ELECTRICITY/GAS/HEAT/STEAM/COOLING are energy over time; ELECTRICAL_POWER/COOLING_POWER/STEAMRATE are instantaneous rates
- Not all buildings have all utilities; analyze utilities separately unless explicitly converting

### SIMS Building Metadata (CSV)
- Join key: `buildingNumber` -> meter `simsCode`
- Key fields: `buildingName`, `grossArea` (sqft), `floorsAboveGround`, `floorsBelowGround`, `constructionDate`, `latitude`, `longitude`, `campusName`

### Weather Data (CSV, hourly, 2025)
- Join key: `date` -> meter `readingTime`
- All from OSU main campus location (40.08, -83.06)
- Key fields: `temperature_2m` (°F), `dew_point_2m`, `relative_humidity_2m`, `precipitation`, `direct_radiation` (W/m²), `wind_speed_10m` (mph), `cloud_cover` (%), `apparent_temperature` (°F)

## Evaluation Rubric

- Data Processing & Scale: 25% — all buildings, automated workflows
- Analytical Rigor & AI Usage: 25% — ML models, expected vs observed, weather/metadata usage
- Explainability & Reasoning: 20% — why buildings rank high, visualizations, stated limitations
- Investment Prioritization Logic: 15% — consistent criteria, portfolio-level reasoning
- Validation & Reflection: 15% — sanity checks, uncertainty discussion, real-world applicability

## Key Analytical Notes

- Analyze **all buildings**, not a subset
- Normalize energy by `grossArea` (sqft) for fair comparison
- Weather is input, energy is output — model expected energy response to weather
- No ground truth ranking — methodology and clarity matter most
- Residual: `δ_{b,t} = e_{b,t} - ê_{b,t}`
- Anomaly flagging: z-score, isolation forest, or autoencoder
- Multi-objective ranking: `r_b = α₁·s̃_b + α₂·a_b + α₃·g_b` (stakeholder-tunable weights)
