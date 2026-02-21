# Strategic Energy Investment Prioritization

AI-enabled decision tool that analyzes campus building energy data to prioritize energy efficiency investments. Uses energy meter data, weather information, and building metadata to identify inefficiencies, detect anomalies, and produce explainable investment rankings.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```
Raw data -> Predictive model f_θ -> Expected consumption ê_{b,t}
-> Residuals δ_{b,t} -> Building scores s_b -> Ranking / Knapsack -> Investment set F*
```

## Approach

1. **Predictive Model**: Fit `ê_{b,t} = f_θ(w_t, m_b, t)` to estimate normal consumption given weather and building metadata
2. **Inefficiency Scoring**: Compute residuals `δ_{b,t} = e_{b,t} - ê_{b,t}` and aggregate per-building scores
3. **Anomaly Detection**: Flag buildings with statistically significant excess consumption
4. **Investment Optimization**: Solve 0-1 knapsack to maximize savings under budget constraint
5. **Multi-Objective Ranking**: Weighted combination of inefficiency, anomaly, and equity scores
