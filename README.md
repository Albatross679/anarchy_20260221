# Strategic Energy Investment Prioritization

**Team Anarchy** — OSU AI Hackathon 2026

AI-enabled decision tool that analyzes ~60 days of campus building energy meter data, weather, and building metadata from Ohio State University to rank buildings for energy efficiency investment. The system identifies buildings that consistently consume more energy than expected — given their size, age, and weather conditions — and produces an explainable, stakeholder-tunable ranking of the top candidates for retrofit investment.

## Links

- [Presentation Video](https://drive.google.com/drive/folders/16ZrS_h_ANLD-5eg7Bl71KPT2ZSlrplOm)
- [Interactive Map Dashboard](http://65.109.75.3:3000/)
- [Application Repository](https://github.com/morimori00/anarchy_20260221_application)

## Pipeline

```
Raw meter data ─> Predictive model f_θ(weather, building, time)
               ─> Expected consumption ê_{b,t}
               ─> Residuals δ_{b,t} = actual - expected
               ─> Building-level scores
               ─> Multi-objective ranking
               ─> Investment shortlist (top 5-10 buildings)
```

## Approach

1. **Data Preparation** — ~1.5M readings across 1,022 meters, 287 buildings, 8 utility types. Meter data joined with building metadata and hourly weather. Features include weather conditions, building characteristics (sqft, floors, age), and time signals (hour, day of week, weekend).

2. **Expected Consumption Model** — Train regression models to predict normal energy use: `ê = f_θ(weather, building, time)`. The model learns typical consumption patterns across the full building portfolio, establishing a baseline of normal behavior.

3. **Inefficiency Scoring** — Compute residuals `δ_{b,t} = e_{b,t} - ê_{b,t}`. Buildings the model consistently under-predicts are consuming more than what's typical for similar buildings under similar conditions.

4. **Anomaly Detection** — Flag buildings with statistically significant excess consumption using z-scores and isolation forests.

5. **Investment Ranking** — Multi-objective score `r_b = α₁·s̃_b + α₂·a_b + α₃·g_b` with stakeholder-tunable weights. Portfolio-level optimization under budget constraints.

6. **Explainability** — Per-building evidence cards, SHAP feature attributions, uncertainty quantification, and an interactive map dashboard for non-technical decision-makers.

## Models

We trained 14 model variants spanning four families to ensure robust predictions across utility types:

| Family | Models |
|---|---|
| Gradient Boosting | XGBoost, LightGBM, DART, CatBoost, NGBoost |
| Tree Ensembles | Random Forest, Extra Trees, Quantile Regression Forest |
| Deep Learning | CNN (1D Conv), LSTM, Transformer (Encoder-Only), Temporal Fusion Transformer |
| Gas-specific | XGBoost-Gas, Neural Net-Gas |

## Data

| Dataset | Records | Interval | Join Key |
|---|---|---|---|
| Smart meter readings | ~1.5M | 15-min | `simsCode` → `buildingNumber` |
| Building metadata (SIMS) | 287 buildings | static | `buildingNumber` |
| Weather (Open-Meteo) | ~60 days | hourly | `date` → `readingTime` |

Utilities covered: Electricity, Gas, Heat, Steam, Cooling, and corresponding power/rate types.

## Project Structure

```
src/                          # Core pipeline
├── data_loader.py            # Load and preprocess energy, weather, building data
├── data_cleaner.py           # 7-step data cleaning pipeline
├── feature_engineer.py       # Shared feature engineering (lags, rolling, interactions)
├── prepare_tree_features.py  # Pre-compute tree features to parquet
├── scoring.py                # Residuals and building-level scores
├── explainer.py              # SHAP analysis and visualizations
├── uncertainty.py            # Uncertainty quantification
├── evidence_cards.py         # Per-building evidence card generation
└── config.py                 # Shared configuration

xgb/  cb/  lgbm/  dart/       # Gradient boosting model implementations
rf/  extratrees/  qrf/        # Tree ensemble model implementations
cnn/  lstm/  transformer/     # Deep learning model implementations
tft/                          # Temporal Fusion Transformer
xgb_gas/  nn_gas/             # Gas-specific model variants

data/                         # Energy meter CSVs, weather, building metadata
output/                       # Generated rankings, reports, plots
doc/                          # Methodology, model specs, project report
slides/                       # Presentation materials
```

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

Pre-compute features (run once):
```bash
python src/prepare_tree_features.py
```

Train a model:
```bash
python xgb/train.py --precomputed --utility ELECTRICITY
```

Run tests:
```bash
pytest tests/
```

## Documentation

- [`doc/PROJECT_REPORT.md`](doc/PROJECT_REPORT.md) — Full project report
- [`doc/methodology.md`](doc/methodology.md) — Detailed methodology
- [`doc/XGB_MODEL.md`](doc/XGB_MODEL.md) — XGBoost model specification
- [`doc/CNN_MODEL.md`](doc/CNN_MODEL.md) — CNN model specification
- [`doc/data-dictionary.md`](doc/data-dictionary.md) — Data dictionary
