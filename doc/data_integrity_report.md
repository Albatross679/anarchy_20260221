# Data Integrity & Quality Audit: Campus Energy Systems (Sept-Oct 2025)

This report provides a comprehensive evaluation of the data quality across the building metadata, meter readings, and weather datasets. Our analysis has identified critical inconsistencies that must be addressed before any reliable research or modeling can proceed.

---

## 1. The Outlier & Unit Mismatch Crisis

The most immediate concern is the presence of "physically impossible" data points and systematic logging errors, particularly in the STEAM and GAS sectors.

### Systematic Unit Mismatch in GAS

We discovered a widespread configuration error where **GAS utility meters are reporting in kWh instead of the expected kg**. This affects nearly 248,000 records, suggesting a default setting in the data ingestion pipeline was never overridden for gas assets.

| # | Utility     | Expected Unit | Observed Primary Unit | Status         |
|---|-------------|---------------|-----------------------|----------------|
| 0 | GAS         | kg            | kWh (Mismatch)        | Critical Error |
| 1 | STEAM       | kg            | kg                    | Correct        |
| 2 | ELECTRICITY | kWh           | kWh                   | Correct        |
| 3 | HEAT        | kWh           | kWh                   | Correct        |
| 4 | COOLING     | kWh           | kWh                   | Correct        |

### Extreme Outliers (>10^8)

A total of 1,194 records contain astronomically high values (10^8 to 10^11), which are physically implausible.

#### Distribution of Extreme Outliers (>1e8) by Utility Type

> *[Bar chart showing the distribution of extreme outliers by utility type. STEAM has the highest count (~800), followed by GAS (~150), COOLING (~50), ELECTRICITY/ELEC (~25), and HEAT (~10).]*

**Key Observation:** These anomalies are not random. They are heavily concentrated in **older buildings** and specific sites (e.g., Site 44006). The chart below shows that buildings constructed around 1918 and 1970 are "hotspots" for sensor errors.

#### Outlier Density by Construction Year

> *[Bar chart showing outlier density (outliers per building) by construction year. Buildings from ~1918 have the highest density (~250), followed by ~1970 (~225), with smaller peaks around ~1983, ~1995, and ~2022.]*

---

## 2. The Metadata Gap: Broken Linkage

A significant hurdle for building-level analysis is the lack of a shared primary key between the `building_metadata` and `meter-data` tables.

- **The Issue:** The `buildingNumber` in metadata does not match the `siteId` in meter tables.
- **The Workaround:** We employed fuzzy string matching on names, but this is an imperfect bridge.
- **The Result:** Approximately 24% of sites remain "unlinked" or ambiguous. These sites have energy data but cannot be reliably linked to building attributes like gross area or construction date.

### Site Mapping Success Rate

> *[Pie chart showing site mapping success rates:*
> - *Mapped (Known Building): 75.4% (blue)*
> - *Unmapped (Other): 23.9% (orange)*
> - *Auxiliary/Unknown: 0.65% (green)]*

---

## 3. Statistical Bias: The Risk of Naive Imputation

Missing meter readings are not distributed randomly, which poses a major risk for aggregate reporting and carbon footprint estimation.

- **Non-Randomness:** Missingness is utility-specific. **STEAM** and **HEAT** have missingness rates exceeding 10%, while GAS is nearly complete.
- **The Bias:** If we fill these gaps with simple averages (mean imputation), the total reported consumption for **STEAM jumps by 14%**. This indicates that the missing periods likely occurred during high-usage times or at high-usage sites.

### Impact of Mean Imputation on Reported Totals

> *[Bar chart showing the percentage increase in total consumption (%) when using mean imputation, by utility type. STEAM has the highest increase (~14%), followed by HEAT (~8%), ELECTRICITY (~4%), COOLING (~3%), and GAS (~1%).]*

---

## Interactive Exploration: Outlier Impact Simulator

Use the tool below to see how different "Hard Caps" (thresholds) would affect your total consumption metrics. This helps in deciding where to set the filters for data cleaning.

> **Outlier Impact Simulator**
>
> Adjust the threshold to see how many records would be flagged as outliers and the resulting reduction in "phantom" energy usage. The logic assumes that any value above the cap is a sensor error.
>
> | SELECT UTILITY | HARD CAP (VALUE) |
> |----------------|------------------|
> | STEAM          | 1000000          |
>
> **[Run Simulation]**

---

## Conclusion & Recommended Actions

To ensure the data is "research-ready," the following pre-processing steps are mandatory:

1. **Immediate Unit Correction:** Convert GAS readings from kWh to kg using a verified conversion factor.
2. **Targeted Sensor Audit:** Prioritize physical inspection of meters at **Site 44006** and **Site 44056**, which drive the majority of STEAM anomalies.
3. **Establish Master Mapping:** Create a permanent, manually verified lookup table between `siteId` and `buildingNumber` to resolve the 24% mapping gap.
4. **Advanced Imputation:** Do not use simple means for STEAM or HEAT. Implement regression-based imputation (using weather and building size as predictors) to avoid the 14% bias identified in this report.
