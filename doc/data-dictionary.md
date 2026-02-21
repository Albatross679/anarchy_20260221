# Data Dictionary

## Overview

Four CSV files covering ~60 days of campus energy data (Sept–Oct 2025) for Ohio State University buildings.

| File | Rows | Size | Description |
|---|---|---|---|
| `meter-data-sept-2025.csv` | 735,840 | 227 MB | 15-min interval meter readings, September 2025 |
| `meter-data-oct-2025.csv` | 760,368 | 235 MB | 15-min interval meter readings, October 2025 |
| `building_metadata.csv` | 1,287 | 246 KB | Building-level metadata (area, floors, location, vintage) |
| `weather-sept-oct-2025.csv` | 1,464 | 393 KB | Hourly weather observations, Sept 1 – Oct 31 2025 |

### Key Counts

- **1,287** buildings in metadata
- **287** buildings with meter data (linked via `simscode`/`buildingnumber`)
- **1,022** unique meters across **8** utility types
- **~1.5M** total meter readings

---

## Joins

```
building_metadata.buildingnumber ──direct──> meter_data.simscode
weather_data.date ──truncate to hour──> meter_data.readingtime
```

- Building join is exact: `buildingnumber` = `simscode`
- Weather join requires aligning hourly weather timestamps to 15-min meter intervals (e.g., truncate `readingtime` to the hour, or forward-fill)

---

## meter-data-sept-2025.csv / meter-data-oct-2025.csv

Each row is one 15-minute interval reading from a single meter. Readings are **per-interval** (not cumulative) — sum them for daily/monthly totals.

**Date range:** 2025-09-01 to 2025-11-01 (combined)

| Column | Type | Description | Example |
|---|---|---|---|
| `meterid` | int | Unique meter identifier | 292883 |
| `siteid` | int | Site identifier | 44071 |
| `sitename` | string | Human-readable site name | "Independence Hall" |
| `simscode` | int | Building code (join key to `buildingnumber`) | 338 |
| `utility` | string | Utility type (see table below) | "STEAM" |
| `readingtime` | datetime | Timestamp of the reading (ISO 8601) | 2025-09-01T05:00:00 |
| `readingvalue` | float | Consumption for this 15-min interval | 11.764 |
| `readingunits` | string | Unit of measurement | "kg" |
| `readingunitsdisplay` | string | Display-friendly unit name | "Kilograms" |
| `readingwindowstart` | datetime | Start of the rolling 24h summary window | 2025-09-01T05:00:00 |
| `readingwindowend` | datetime | End of the rolling 24h summary window | 2025-09-02T04:45:00 |
| `expectedwindowreadings` | int | Expected readings in window (96 = 24h) | 96 |
| `totalwindowreadings` | int | Actual readings received in window | 96 |
| `missingwindowreadings` | int | Missing readings in window | 0 |
| `filteredwindowreadings` | int | Readings excluded by quality filters | 0 |
| `readingwindowsum` | float | Sum of all readings in 24h window | 1103.79 |
| `readingwindowmean` | float | Mean reading in 24h window | 11.498 |
| `readingwindowstandarddeviation` | float | Std dev of readings in 24h window | 1.182 |
| `readingwindowmin` | float | Min reading in 24h window | 9.039 |
| `readingwindowmintime` | datetime | Timestamp of min reading | 2025-09-01T15:30:00 |
| `readingwindowmax` | float | Max reading in 24h window | 17.424 |
| `readingwindowmaxtime` | datetime | Timestamp of max reading | 2025-09-01T11:00:00 |
| `year` | int | Year extracted from readingtime | 2025 |
| `month` | int | Month extracted from readingtime | 09 |
| `day` | int | Day extracted from readingtime | 01 |

### Utility Types

| Utility | Category | Description | Units |
|---|---|---|---|
| **ELECTRICITY** | Energy | Electrical energy consumption | kWh |
| **GAS** | Energy | Natural gas consumption | varies |
| **HEAT** | Energy | Thermal energy delivered for heating | varies |
| **STEAM** | Energy | Thermal energy delivered as steam | kg |
| **COOLING** | Energy | Thermal energy delivered for cooling | ton-hours |
| **COOLING_POWER** | Power | Instantaneous cooling demand | tons |
| **STEAMRATE** | Power | Instantaneous steam flow rate | varies |
| **OIL28SEC** | Energy | Fuel oil consumption | varies |

**Energy** utilities (ELECTRICITY, GAS, HEAT, STEAM, COOLING) represent total consumption over the 15-min interval.
**Power** utilities (COOLING_POWER, STEAMRATE) represent instantaneous rate/demand.
Analyze utilities separately — do not sum across different utility types without explicit unit conversion.

---

## building_metadata.csv

One row per building. Not all buildings have meter data — only 287 of 1,287 appear in the meter files.

| Column | Type | Description | Completeness | Example |
|---|---|---|---|---|
| `buildingnumber` | string | Building ID (join key to `simscode`) | 1287/1287 | "356" |
| `buildingname` | string | Building name | 1287/1287 | "Twelfth Ave, 395 W (0356)" |
| `campusname` | string | Campus name | 1287/1287 | "Medical Center" |
| `address` | string | Street address | 1287/1287 | "395 W 12th Ave" |
| `city` | string | City | 1287/1287 | "Columbus" |
| `state` | string | State | 1287/1287 | "OH" |
| `postalcode` | string | ZIP code | 1287/1287 | "43210-1267" |
| `county` | string | County | 1287/1287 | "Franklin" |
| `formalname` | string | Formal building name | partial | — |
| `alsoknownas` | string | Alternate name | sparse | — |
| `grossarea` | float | Gross square footage | 1287/1287 | 90747 |
| `floorsaboveground` | float | Floors above ground | 1287/1287 | 9.0 |
| `floorsbelowground` | float | Floors below ground | 1287/1287 | 0.0 |
| `constructiondate` | date | Construction date (YYYY-MM-DD) | 982/1287 | "2007-07-01" |
| `latitude` | float | Latitude | 1249/1287 | 39.9959 |
| `longitude` | float | Longitude | 1249/1287 | -83.0179 |

---

## weather-sept-oct-2025.csv

Hourly weather observations from a single station at OSU main campus (40.08, -83.06). All rows share the same lat/lon.

**Date range:** 2025-09-01 00:00 to 2025-10-31 23:00 (1,464 hours)

| Column | Type | Units | Description | Example |
|---|---|---|---|---|
| `date` | datetime | ISO 8601 | Observation timestamp | 2025-09-01T00:00:00 |
| `latitude` | float | degrees | Station latitude (constant) | 40.0795 |
| `longitude` | float | degrees | Station longitude (constant) | -83.0732 |
| `temperature_2m` | float | °F | Air temperature at 2m | 68.2 |
| `shortwave_radiation` | int | W/m² | Shortwave radiation | 60 |
| `direct_radiation` | float | W/m² | Direct solar radiation | 35.0 |
| `diffuse_radiation` | float | W/m² | Diffuse solar radiation | 25 |
| `direct_normal_irradiance` | float | W/m² | Direct normal irradiance | 257.6 |
| `relative_humidity_2m` | float | % | Relative humidity at 2m | 58.1 |
| `dew_point_2m` | float | °F | Dew point at 2m | 52.9 |
| `precipitation` | float | mm | Total precipitation during the hour | 0.0 |
| `wind_speed_10m` | float | mph | Wind speed at 10m | 8.08 |
| `wind_speed_100m` | float | mph | Wind speed at 100m | 14.2 |
| `wind_direction_100m` | float | degrees | Wind direction at 100m | 26.2 |
| `wind_direction_10m` | float | degrees | Wind direction at 10m | 21.9 |
| `cloud_cover` | int | % | Cloud cover fraction | 0 |
| `apparent_temperature` | float | °F | Feels-like temperature | 65.7 |
| `partition_0` | string | — | Data partition label (ignorable) | "2025-01-01_2025-11-05" |
