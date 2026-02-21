# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hackathon project: Strategic Energy Investment Prioritization. Builds an AI-enabled decision tool that analyzes campus building energy meter data, weather, and building metadata to rank buildings for energy efficiency investment.

**Pipeline**: Raw data -> predictive model -> residuals (actual - expected) -> building-level scores -> ranking/knapsack optimization -> investment set

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

## Architecture

```
src/
├── data_loader.py    # Load and preprocess energy, weather, building data
├── model.py          # Predictive model f_θ(w_t, m_b, t) for expected consumption
├── scoring.py        # Compute residuals δ_{b,t} and building-level scores s_b
├── anomaly.py        # Anomaly detection (z-score, isolation forest)
├── optimizer.py      # 0-1 knapsack / ranking for investment selection
├── explainer.py      # Explainability and visualization of rankings
└── utils/            # Shared helpers
```

- `data/` — energy meter CSVs, weather data, building metadata
- `output/` — generated rankings, reports, plots
- `model/` — saved model weights

## Key Mathematical Details

- Inefficiency residual: `δ_{b,t} = e_{b,t} - ê_{b,t}`
- Building score options: mean excess, tail behavior (above threshold τ), normalized by floor area
- Anomaly flagging: `(s_b - μ_s) / σ_s > z_α`
- Investment optimization: maximize `Σ Δ_b * x_b` subject to `Σ c_b * x_b ≤ B` (knapsack)
- Multi-objective ranking: `r_b = α₁·s̃_b + α₂·a_b + α₃·g_b` with stakeholder-tunable weights
