# Known Issues

## Data Quality

- **GAS unit mismatch** — ~248k GAS records report kWh instead of kg
- **Extreme outliers** — 1,194 records with physically implausible values (10^8–10^11), mostly in STEAM and GAS
- **Non-random missing data** — STEAM and HEAT have >10% missing readings; mean imputation inflates STEAM totals by ~14%
- **Building-meter linkage gap** — 24% of meter sites can't be reliably linked to building metadata; 1,004 buildings have no meter data

## Empty / Unusable Utilities

- **ELECTRICAL_POWER** — zero meter records; training fails with empty dataset warnings
- **STEAMRATE, OIL28SEC, COOLING_POWER** — limited or no data availability

## Model Metrics

- **MAPE is meaningless** — normalized energy values are tiny (0.0001–0.001), producing MAPE of 1,000–15,000% even when R² is decent
- **Inconsistent metric labels** — some models report `R2`, others `R²`

## CNN / LSTM

- **Very high MAPE** — CNN MAPE ~15,000%, LSTM ~5,000%, despite R² of 0.65–0.73
- **Several interrupted runs** — multiple CNN training runs terminated early or produced no predictions

## GAS Two-Stage Model

- **Lower R² than single-stage** — two-stage gas model gets R²=0.61 vs single-stage R²=0.63
- **Sparse cross-utility features** — 12 columns with 64–90% NaN; dropped or zero-filled, hurting performance

## QRF (Quantile Random Forest)

- **NumPy RNG deprecation warning** — `np.random.seed` will stop working in future NumPy versions
- **SHAP computation slow** — PermutationExplainer takes ~12 minutes per run

## Data Loading

- **Three fallback paths** — pre-cleaned CSV → on-the-fly cleaning → raw data; not clearly documented which is preferred
- **Precomputed features pipeline** — new `--precomputed` flag works but has minimal error handling on manifest parsing

## Output / Naming

- **Run naming changed mid-project** — old: `energy_{model}_{timestamp}`, new: `{utility}_{model}_{timestamp}`; breaks continuity of output tracking
- **Truncated console logs** — ~12 runs have <100 lines in console.log, unclear if data was lost

## Weather Data

- **Single weather station** — all data from one campus location (40.08, -83.06); no adjustment for microclimates or distant buildings

## Git / Repo

- **Large binary files tracked** — model checkpoints (.pt, .json) committed directly instead of using Git LFS
