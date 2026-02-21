# Methodology

## Core Idea

Buildings that consistently consume more energy than expected — given their size, age, and weather conditions — are the best candidates for energy efficiency investment. We build a model that learns what "normal" consumption looks like, then flag buildings that deviate from that norm.

## Pipeline

```
Raw data ─> predictive model ─> residuals (actual - expected) ─> building scores ─> ranking ─> investment shortlist
```

### Step 1: Data Preparation

**Inputs:**
- Meter data: ~1.5M readings across 1,022 meters, 287 buildings, 8 utility types
- Building metadata: sqft, floors, construction year, location
- Weather: hourly temperature, humidity, wind, radiation, precipitation

**Processing:**
- Sum meters of the same utility per building to get building-level consumption
- Align hourly weather to 15-min meter intervals (truncate readingtime to hour)
- Join building metadata via `buildingnumber` = `simscode`
- Engineer time features: hour of day, day of week, weekend flag

**Scope:** Start with ELECTRICITY (most widely metered utility), then extend to others.

### Step 2: Train the Expected Consumption Model

We train a regression model to predict energy consumption:

```
expected_energy = f(weather, building_features, time_features)
```

**Features (X):**
- Weather: `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `wind_speed_10m`, `cloud_cover`, `apparent_temperature`, `direct_radiation`, `precipitation`
- Building: `grossarea` (sqft), `floorsaboveground`, `floorsbelowground`, building age (derived from `constructiondate`)
- Time: hour of day, day of week, weekend indicator

**Label (y):**
- `readingvalue` — actual energy consumed in the 15-min interval

**Why this works without separate labels:** The model learns the average relationship between conditions and consumption across all buildings. It doesn't need a "correct answer" — it establishes a baseline of normal behavior by learning from the entire portfolio. Buildings that the model consistently under-predicts are consuming more than what's typical for buildings like them under those conditions.

**Model candidates:**
- Gradient-boosted trees (XGBoost/LightGBM) — handles mixed feature types, fast, interpretable feature importances
- Linear regression with interaction terms — simpler, fully transparent
- Random forest — robust, less prone to overfitting

### Step 3: Compute Residuals

For each building b at each time step t:

```
δ_{b,t} = actual_{b,t} - expected_{b,t}
```

- **Positive residual**: building used more than expected (potential inefficiency)
- **Negative residual**: building used less than expected (efficient or underutilized)

### Step 4: Score Each Building

Aggregate residuals into building-level scores. Multiple signals, each capturing a different dimension of inefficiency:

| Signal | Definition | Why It Matters |
|---|---|---|
| **Mean excess** | `mean(δ_{b,t}) / grossarea` | Overall inefficiency normalized by size |
| **Tail behavior** | Fraction of intervals where `δ_{b,t}` exceeds threshold τ | How often the building is wasteful, not just on average |
| **Weather sensitivity** | Slope of consumption vs temperature | Buildings overreacting to weather may have HVAC issues |
| **Baseline load** | Consumption in mild weather / overnight | High baseline suggests always-on waste (leaks, poor controls) |
| **Variability** | `std(δ_{b,t}) / grossarea` | Unstable consumption may indicate malfunctioning systems |
| **Peer comparison** | Z-score of building's mean consumption vs buildings of similar size/age | How it compares to its closest peers |

### Step 5: Rank and Prioritize

Combine signals into a composite score with stakeholder-tunable weights:

```
rank_b = α₁ · normalized_excess_b + α₂ · anomaly_score_b + α₃ · savings_potential_b
```

- Weights are adjustable — the dashboard should let users explore how rankings shift under different priorities
- Optionally apply knapsack optimization: maximize total savings potential subject to a budget constraint on estimated retrofit costs

### Step 6: Validate and Explain

- **Sanity checks**: Do top-ranked buildings make intuitive sense (old, large, high baseline)?
- **Stability**: Do rankings change significantly if we train on Sept only vs Oct only?
- **Uncertainty**: Report confidence tiers, not just point rankings — a building ranked #3 with high variance in its residuals is less certain than one with consistent excess
- **Limitations**: 60-day window, no occupancy data, no equipment-level detail

## Analogy

Grading on a curve. We don't have an answer key for what each building "should" use. Instead, the model learns the class average — how buildings of a given size, age, and weather exposure typically perform. Buildings far above that curve are the underperformers worth investigating.

## Key Assumptions

1. Weather, building size, age, and time of day/week are the primary drivers of "expected" consumption
2. The model's prediction errors are informative — persistent over-consumption signals real inefficiency, not just model noise
3. Utilities should be analyzed separately (electricity ≠ steam ≠ cooling)
4. Summing meters of the same utility per building gives a reasonable building-level total
5. 60 days is enough to identify persistent patterns, but not seasonal ones — this is a screening tool, not a final diagnosis
