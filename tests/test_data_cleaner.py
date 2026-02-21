"""Unit tests for data_cleaner functions using synthetic DataFrames."""

import pandas as pd
import numpy as np
import pytest

from src.data_cleaner import (
    GAS_KWH_TO_KG,
    UTILITY_HARD_CAPS,
    CleaningReport,
    fix_gas_units,
    apply_hard_caps,
    build_site_lookup,
    apply_metadata_linkage,
    impute_missing_intervals,
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
# TestFixGasUnits
# ---------------------------------------------------------------------------

class TestFixGasUnits:
    def test_converts_gas_kwh_to_kg(self):
        """GAS rows with kWh units should be converted to kg."""
        df = _make_meter_df(
            utility=["GAS"],
            readingvalue=[29.3],  # 29.3 kWh = 1 kg
            readingunits=["kWh"],
        )
        report = CleaningReport()
        result = fix_gas_units(df, report)

        assert report.gas_rows_converted == 1
        assert abs(result["readingvalue"].iloc[0] - 1.0) < 1e-6
        assert result["readingunits"].iloc[0] == "kg"

    def test_steam_unchanged(self):
        """Non-GAS utilities should not be modified."""
        df = _make_meter_df(
            utility=["STEAM"],
            readingvalue=[100.0],
            readingunits=["kWh"],
        )
        report = CleaningReport()
        result = fix_gas_units(df, report)

        assert report.gas_rows_converted == 0
        assert result["readingvalue"].iloc[0] == 100.0

    def test_no_double_convert(self):
        """GAS rows already in kg should not be converted again."""
        df = _make_meter_df(
            utility=["GAS"],
            readingvalue=[1.0],
            readingunits=["kg"],
        )
        report = CleaningReport()
        result = fix_gas_units(df, report)

        assert report.gas_rows_converted == 0
        assert result["readingvalue"].iloc[0] == 1.0

    def test_window_stats_scaled(self):
        """Window stat columns should be scaled by the same factor."""
        df = _make_meter_df(
            utility=["GAS"],
            readingvalue=[29.3],
            readingunits=["kWh"],
        )
        df["readingwindowsum"] = [293.0]
        df["readingwindowmean"] = [29.3]

        report = CleaningReport()
        result = fix_gas_units(df, report)

        assert abs(result["readingwindowsum"].iloc[0] - 10.0) < 1e-4
        assert abs(result["readingwindowmean"].iloc[0] - 1.0) < 1e-6

    def test_input_not_mutated(self):
        """Original DataFrame should not be modified."""
        df = _make_meter_df(
            utility=["GAS"],
            readingvalue=[29.3],
            readingunits=["kWh"],
        )
        original_value = df["readingvalue"].iloc[0]
        report = CleaningReport()
        fix_gas_units(df, report)

        assert df["readingvalue"].iloc[0] == original_value


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
# TestBuildSiteLookup
# ---------------------------------------------------------------------------

class TestBuildSiteLookup:
    def test_direct_match_found(self):
        """simscode matching buildingnumber should be a direct match."""
        meter_df = _make_meter_df(simscode=["100"], sitename=["Foo Hall"])
        building_df = _make_building_df(buildingnumber=["100"], buildingname=["Foo Hall"])

        lookup = build_site_lookup(meter_df, building_df)
        assert len(lookup) == 1
        assert lookup["match_type"].iloc[0] == "direct"
        assert lookup["buildingnumber"].iloc[0] == "100"

    def test_unmatched_recorded(self):
        """simscode with no matching buildingnumber should be unmatched."""
        meter_df = _make_meter_df(simscode=["999"], sitename=["Nowhere"])
        building_df = _make_building_df(
            buildingnumber=["100"], buildingname=["Completely Different"]
        )

        lookup = build_site_lookup(meter_df, building_df, threshold=95.0)
        assert len(lookup) == 1
        assert lookup["match_type"].iloc[0] == "unmatched"

    def test_fuzzy_match(self):
        """Similar names should produce a fuzzy match above threshold."""
        meter_df = _make_meter_df(simscode=["999"], sitename=["Science Hall West"])
        building_df = _make_building_df(
            buildingnumber=["200"], buildingname=["Science Hall West Wing"]
        )

        lookup = build_site_lookup(meter_df, building_df, threshold=50.0)
        assert len(lookup) == 1
        assert lookup["match_type"].iloc[0] == "fuzzy"
        assert lookup["buildingnumber"].iloc[0] == "200"


# ---------------------------------------------------------------------------
# TestApplyMetadataLinkage
# ---------------------------------------------------------------------------

class TestApplyMetadataLinkage:
    def test_resolved_column_added(self):
        """Should add resolved_buildingnumber from lookup."""
        meter_df = _make_meter_df(simscode=["100"])
        building_df = _make_building_df()
        lookup_df = pd.DataFrame({
            "simscode": ["100"],
            "sitename": ["Test"],
            "buildingnumber": ["100"],
            "buildingname": ["Test Building"],
            "match_type": ["direct"],
            "fuzzy_score": [100.0],
        })
        report = CleaningReport()
        result = apply_metadata_linkage(meter_df, building_df, lookup_df, report)

        assert "resolved_buildingnumber" in result.columns
        assert result["resolved_buildingnumber"].iloc[0] == "100"
        assert report.matched_direct == 1

    def test_unmatched_rows_have_nan(self):
        """Unmatched simscodes should have NaN in resolved_buildingnumber."""
        meter_df = _make_meter_df(simscode=["999"])
        building_df = _make_building_df()
        lookup_df = pd.DataFrame({
            "simscode": ["999"],
            "sitename": ["Unknown"],
            "buildingnumber": [None],
            "buildingname": [None],
            "match_type": ["unmatched"],
            "fuzzy_score": [0.0],
        })
        report = CleaningReport()
        result = apply_metadata_linkage(meter_df, building_df, lookup_df, report)

        assert pd.isna(result["resolved_buildingnumber"].iloc[0])
        assert report.unmatched_simscodes == 1

    def test_report_counts(self):
        """Report should count direct, fuzzy, and unmatched separately."""
        meter_df = pd.DataFrame({
            "meterid": ["M1", "M2", "M3"],
            "simscode": ["100", "200", "300"],
            "sitename": ["A", "B", "C"],
            "utility": ["ELECTRICITY"] * 3,
            "readingtime": pd.to_datetime(["2025-09-01"] * 3),
            "readingvalue": [10.0, 20.0, 30.0],
            "readingunits": ["kWh"] * 3,
        })
        building_df = _make_building_df()
        lookup_df = pd.DataFrame({
            "simscode": ["100", "200", "300"],
            "sitename": ["A", "B", "C"],
            "buildingnumber": ["100", "200", None],
            "buildingname": ["A", "B", None],
            "match_type": ["direct", "fuzzy", "unmatched"],
            "fuzzy_score": [100.0, 85.0, 30.0],
        })
        report = CleaningReport()
        apply_metadata_linkage(meter_df, building_df, lookup_df, report)

        assert report.matched_direct == 1
        assert report.matched_fuzzy == 1
        assert report.unmatched_simscodes == 1


# ---------------------------------------------------------------------------
# TestImputeMissingIntervals
# ---------------------------------------------------------------------------

class TestImputeMissingIntervals:
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
        result = impute_missing_intervals(df, report, gap_limit=4)

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
        result = impute_missing_intervals(df, report, gap_limit=4)

        # With gap_limit=4, ffill fills 4 from the left, bfill fills 4 from the right
        # Middle values (2 of them) stay NaN
        assert result["readingvalue"].isna().sum() > 0
        assert report.gaps_too_long > 0

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
        result = impute_missing_intervals(df, report, gap_limit=4)

        # M2 has no prior value in its group, so it should remain NaN
        m2_row = result[result["meterid"] == "M2"]
        assert m2_row["readingvalue"].isna().all()

    def test_noop_when_complete(self):
        """No changes when there are no NaN values."""
        df = _make_meter_df(readingvalue=[10.0])
        report = CleaningReport()
        result = impute_missing_intervals(df, report)

        assert len(result) == 1
        assert report.intervals_filled == 0


# ---------------------------------------------------------------------------
# TestCleanMeterData (integration)
# ---------------------------------------------------------------------------

class TestCleanMeterData:
    def test_full_pipeline(self, tmp_path):
        """Integration test: all cleaning steps run in sequence."""
        # Create meter data with issues
        meter_df = pd.DataFrame({
            "meterid": ["M1", "M2", "M3", "M4"],
            "simscode": ["100", "100", "200", "300"],
            "sitename": ["Test Bldg", "Test Bldg", "Other Bldg", "Unknown"],
            "utility": ["GAS", "ELECTRICITY", "ELECTRICITY", "ELECTRICITY"],
            "readingtime": pd.to_datetime([
                "2025-09-01 00:00", "2025-09-01 00:00",
                "2025-09-01 00:00", "2025-09-01 00:00",
            ]),
            "readingvalue": [29.3, 1e10, 50.0, 100.0],  # GAS in kWh, extreme outlier
            "readingunits": ["kWh", "kWh", "kWh", "kWh"],
        })

        building_df = _make_building_df(
            buildingnumber=["100", "200"],
            buildingname=["Test Bldg", "Other Bldg"],
            grossarea=[10000.0, 5000.0],
        )

        lookup_path = str(tmp_path / "lookup.csv")
        cleaned, report = clean_meter_data(
            meter_df, building_df, lookup_path=lookup_path
        )

        # GAS should be converted
        assert report.gas_rows_converted == 1

        # Extreme outlier (1e10 ELECTRICITY) should be removed
        assert report.total_outliers_removed >= 1

        # Lookup file should be created
        assert (tmp_path / "lookup.csv").exists()

        # resolved_buildingnumber should exist
        assert "resolved_buildingnumber" in cleaned.columns

        # Result should have fewer rows than input (outlier removed)
        assert len(cleaned) < len(meter_df)
