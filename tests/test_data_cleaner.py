"""Unit tests for data_cleaner functions using synthetic DataFrames."""

import pandas as pd
import numpy as np
import pytest

from src.data_cleaner import (
    UTILITY_HARD_CAPS,
    EXCLUDED_UTILITIES,
    EXCLUDED_SIMSCODES,
    CleaningReport,
    drop_nan_simscode,
    exclude_utilities,
    exclude_unmatched_buildings,
    apply_hard_caps,
    impute_short_gaps,
    clean_meter_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meter_df(**overrides):
    """Create a minimal meter DataFrame for testing."""
    defaults = {
        "meterid": ["M1"],
        "simscode": ["100"],
        "sitename": ["Test Building"],
        "utility": ["ELECTRICITY"],
        "readingtime": pd.to_datetime(["2025-09-01 00:00"]),
        "readingvalue": [50.0],
        "readingunits": ["kWh"],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


def _make_building_df(**overrides):
    """Create a minimal building metadata DataFrame for testing."""
    defaults = {
        "buildingnumber": ["100"],
        "buildingname": ["Test Building"],
        "grossarea": [10000.0],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


# ---------------------------------------------------------------------------
# TestDropNanSimscode
# ---------------------------------------------------------------------------

class TestDropNanSimscode:
    def test_nan_rows_removed(self):
        """Rows with NaN simscode should be dropped."""
        df = _make_meter_df(
            simscode=[100, np.nan, 200],
            meterid=["M1", "M2", "M3"],
            sitename=["A", "B", "C"],
            utility=["ELECTRICITY"] * 3,
            readingtime=pd.to_datetime(["2025-09-01"] * 3),
            readingvalue=[10.0, 20.0, 30.0],
            readingunits=["kWh"] * 3,
        )
        report = CleaningReport()
        result = drop_nan_simscode(df, report)

        assert len(result) == 2
        assert report.nan_simscode_dropped == 1

    def test_valid_rows_kept(self):
        """All rows kept when no NaN simscodes."""
        df = _make_meter_df(simscode=[100])
        report = CleaningReport()
        result = drop_nan_simscode(df, report)

        assert len(result) == 1
        assert report.nan_simscode_dropped == 0

    def test_simscode_cast_to_str(self):
        """simscode should be cast to str after dropping NaN."""
        df = _make_meter_df(simscode=[338.0])
        report = CleaningReport()
        result = drop_nan_simscode(df, report)

        assert result["simscode"].iloc[0] == "338"
        assert result["simscode"].dtype == object


# ---------------------------------------------------------------------------
# TestExcludeUtilities
# ---------------------------------------------------------------------------

class TestExcludeUtilities:
    def test_oil28sec_dropped(self):
        """OIL28SEC rows should be removed."""
        df = _make_meter_df(
            utility=["ELECTRICITY", "OIL28SEC"],
            meterid=["M1", "M2"],
            simscode=["100", "200"],
            sitename=["A", "B"],
            readingtime=pd.to_datetime(["2025-09-01"] * 2),
            readingvalue=[10.0, 0.0],
            readingunits=["kWh", "gal"],
        )
        report = CleaningReport()
        result = exclude_utilities(df, report)

        assert len(result) == 1
        assert report.excluded_utility_dropped == 1
        assert result["utility"].iloc[0] == "ELECTRICITY"

    def test_other_utilities_kept(self):
        """Non-excluded utilities should be retained."""
        df = _make_meter_df(utility=["GAS"])
        report = CleaningReport()
        result = exclude_utilities(df, report)

        assert len(result) == 1
        assert report.excluded_utility_dropped == 0


# ---------------------------------------------------------------------------
# TestExcludeUnmatchedBuildings
# ---------------------------------------------------------------------------

class TestExcludeUnmatchedBuildings:
    def test_excluded_simscodes_dropped(self):
        """simscodes 8, 43, 93 should be dropped."""
        df = _make_meter_df(
            simscode=["8", "43", "93", "100"],
            meterid=["M1", "M2", "M3", "M4"],
            sitename=["A", "B", "C", "D"],
            utility=["ELECTRICITY"] * 4,
            readingtime=pd.to_datetime(["2025-09-01"] * 4),
            readingvalue=[10.0, 20.0, 30.0, 40.0],
            readingunits=["kWh"] * 4,
        )
        report = CleaningReport()
        result = exclude_unmatched_buildings(df, report)

        assert len(result) == 1
        assert result["simscode"].iloc[0] == "100"
        assert report.excluded_buildings_dropped == 3

    def test_valid_simscodes_kept(self):
        """simscodes not in the exclusion set should be retained."""
        df = _make_meter_df(simscode=["100"])
        report = CleaningReport()
        result = exclude_unmatched_buildings(df, report)

        assert len(result) == 1
        assert report.excluded_buildings_dropped == 0


# ---------------------------------------------------------------------------
# TestApplyHardCaps
# ---------------------------------------------------------------------------

class TestApplyHardCaps:
    def test_extreme_values_removed(self):
        """Values above the hard cap should be dropped."""
        df = _make_meter_df(
            utility=["ELECTRICITY", "ELECTRICITY"],
            readingvalue=[5000.0, 1e9],
            meterid=["M1", "M2"],
            simscode=["100", "100"],
            sitename=["A", "A"],
            readingunits=["kWh", "kWh"],
            readingtime=pd.to_datetime(["2025-09-01", "2025-09-01"]),
        )
        report = CleaningReport()
        result = apply_hard_caps(df, report)

        assert len(result) == 1
        assert result["readingvalue"].iloc[0] == 5000.0
        assert report.outliers_removed["ELECTRICITY"] == 1
        assert report.total_outliers_removed == 1

    def test_valid_retained(self):
        """Values within the cap should be kept."""
        df = _make_meter_df(readingvalue=[500.0])
        report = CleaningReport()
        result = apply_hard_caps(df, report)

        assert len(result) == 1
        assert report.total_outliers_removed == 0

    def test_report_counts_by_utility(self):
        """Report should break down removals by utility."""
        df = pd.DataFrame({
            "meterid": ["M1", "M2", "M3"],
            "simscode": ["100", "200", "300"],
            "sitename": ["A", "B", "C"],
            "utility": ["ELECTRICITY", "GAS", "GAS"],
            "readingtime": pd.to_datetime(["2025-09-01"] * 3),
            "readingvalue": [1e9, 1e9, 100.0],
            "readingunits": ["kWh", "kg", "kg"],
        })
        report = CleaningReport()
        result = apply_hard_caps(df, report)

        assert len(result) == 1
        assert report.outliers_removed.get("ELECTRICITY") == 1
        assert report.outliers_removed.get("GAS") == 1

    def test_custom_caps(self):
        """Custom cap dict should override defaults."""
        df = _make_meter_df(readingvalue=[500.0])
        report = CleaningReport()
        result = apply_hard_caps(df, report, caps={"ELECTRICITY": 100})

        assert len(result) == 0
        assert report.total_outliers_removed == 1

    def test_unknown_utility_kept(self):
        """Utilities not in the caps dict should not be dropped."""
        df = _make_meter_df(utility=["UNKNOWN_UTIL"], readingvalue=[1e12])
        report = CleaningReport()
        result = apply_hard_caps(df, report)

        assert len(result) == 1
        assert report.total_outliers_removed == 0


# ---------------------------------------------------------------------------
# TestImputeShortGaps
# ---------------------------------------------------------------------------

class TestImputeShortGaps:
    def test_short_gap_filled(self):
        """Gaps <= gap_limit should be filled."""
        times = pd.date_range("2025-09-01", periods=5, freq="15min")
        df = pd.DataFrame({
            "meterid": ["M1"] * 5,
            "simscode": ["100"] * 5,
            "utility": ["ELECTRICITY"] * 5,
            "readingtime": times,
            "readingvalue": [10.0, np.nan, np.nan, np.nan, 20.0],
        })
        report = CleaningReport()
        result = impute_short_gaps(df, report, gap_limit=4)

        assert result["readingvalue"].isna().sum() == 0
        assert report.intervals_filled == 3

    def test_long_gap_not_filled(self):
        """Gaps > gap_limit should remain NaN."""
        times = pd.date_range("2025-09-01", periods=12, freq="15min")
        values = [10.0] + [np.nan] * 10 + [20.0]
        df = pd.DataFrame({
            "meterid": ["M1"] * 12,
            "simscode": ["100"] * 12,
            "utility": ["ELECTRICITY"] * 12,
            "readingtime": times,
            "readingvalue": values,
        })
        report = CleaningReport()
        result = impute_short_gaps(df, report, gap_limit=4)

        # With gap_limit=4, ffill fills 4 from the left, bfill fills 4 from the right
        # Middle values (2 of them) stay NaN
        assert result["readingvalue"].isna().sum() > 0
        assert report.gaps_remaining > 0

    def test_no_fill_across_meters(self):
        """Values should not fill across different meter groups."""
        times = pd.date_range("2025-09-01", periods=2, freq="15min")
        df = pd.DataFrame({
            "meterid": ["M1", "M2"],
            "simscode": ["100", "200"],
            "utility": ["ELECTRICITY", "ELECTRICITY"],
            "readingtime": times,
            "readingvalue": [10.0, np.nan],
        })
        report = CleaningReport()
        result = impute_short_gaps(df, report, gap_limit=4)

        # M2 has no prior value in its group, so it should remain NaN
        m2_row = result[result["meterid"] == "M2"]
        assert m2_row["readingvalue"].isna().all()

    def test_noop_when_complete(self):
        """No changes when there are no NaN values."""
        df = _make_meter_df(readingvalue=[10.0])
        report = CleaningReport()
        result = impute_short_gaps(df, report)

        assert len(result) == 1
        assert report.intervals_filled == 0


# ---------------------------------------------------------------------------
# TestCleanMeterData (integration)
# ---------------------------------------------------------------------------

class TestCleanMeterData:
    def test_full_pipeline(self):
        """Integration test: all cleaning steps run in sequence."""
        meter_df = pd.DataFrame({
            "meterid": ["M1", "M2", "M3", "M4", "M5"],
            "simscode": [100, 100, 200, np.nan, 8],
            "sitename": ["Bldg A", "Bldg A", "Bldg B", "Unknown", "No Meta"],
            "utility": ["GAS", "ELECTRICITY", "ELECTRICITY", "ELECTRICITY", "ELECTRICITY"],
            "readingtime": pd.to_datetime(["2025-09-01 00:00"] * 5),
            "readingvalue": [29.3, 1e10, 50.0, 100.0, 75.0],
            "readingunits": ["kWh", "kWh", "kWh", "kWh", "kWh"],
        })

        building_df = _make_building_df(
            buildingnumber=["100", "200"],
            buildingname=["Bldg A", "Bldg B"],
            grossarea=[10000.0, 5000.0],
        )

        cleaned, report = clean_meter_data(meter_df, building_df)

        # NaN simscode row should be dropped
        assert report.nan_simscode_dropped == 1
        # Excluded building (simscode 8) should be dropped
        assert report.excluded_buildings_dropped == 1
        # Extreme outlier (1e10 ELECTRICITY) should be removed
        assert report.total_outliers_removed >= 1
        # Fewer rows than input
        assert len(cleaned) < len(meter_df)
        assert report.rows_before == 5
        assert report.rows_after == len(cleaned)

    def test_report_populated(self):
        """Report should have correct rows_before and rows_after."""
        df = _make_meter_df(simscode=[100])
        building_df = _make_building_df()
        cleaned, report = clean_meter_data(df, building_df)

        assert report.rows_before == 1
        assert report.rows_after == 1
