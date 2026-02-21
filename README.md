# Strategic Energy Investment Prioritization

**Team Anarchy** — OSU AI Hackathon Competition Entry

AI-enabled decision tool that analyzes campus building energy data to prioritize energy efficiency investments. Uses ~60 days of smart meter data, weather information, and building metadata from Ohio State University to identify inefficiencies, detect anomalies, and produce explainable investment rankings for a top 5-10 building shortlist.

## Setup

```bash
pip install -e ".[dev]"
```

## Pipeline

```
Raw data -> Predictive model f_θ -> Expected consumption ê_{b,t}
-> Residuals δ_{b,t} -> Building scores s_b -> Ranking / Optimization -> Investment set F*
```

## Approach

1. **Predictive Model**: Fit `ê_{b,t} = f_θ(w_t, m_b, t)` to estimate normal consumption given weather and building metadata
2. **Inefficiency Scoring**: Compute residuals `δ_{b,t} = e_{b,t} - ê_{b,t}` and aggregate per-building scores
3. **Anomaly Detection**: Flag buildings with statistically significant excess consumption
4. **Investment Optimization**: Rank/optimize building selection under budget constraints
5. **Explainability**: Per-building evidence, uncertainty communication, interactive dashboard

## Data

| Dataset | Format | Interval | Join Key |
|---|---|---|---|
| Smart meter readings | CSV | 15-min | `simsCode` |
| Building metadata (SIMS) | CSV | static | `buildingNumber` |
| Weather (Open-Meteo) | CSV | hourly | `date` |
