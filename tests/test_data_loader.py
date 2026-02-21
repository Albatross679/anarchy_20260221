"""Unit tests for data_loader functions using synthetic DataFrames."""

import pandas as pd
import numpy as np
import pytest

from src.data_loader import aggregate_building_meters, engineer_time_features


class TestAggregateBuildingMeters:
    def test_sums_multiple_meters(self):
        """Two meters at the same building/time should sum."""
        df = pd.DataFrame({
            "simscode": ["100", "100", "200"],
            "readingtime": pd.to_datetime(["2025-09-01 00:00", "2025-09-01 00:00", "2025-09-01 00:00"]),
            "readingvalue": [10.0, 20.0, 5.0],
        })
        result = aggregate_building_meters(df)
        assert len(result) == 2

        bldg_100 = result[result["simscode"] == "100"]
        assert bldg_100["readingvalue"].iloc[0] == 30.0

        bldg_200 = result[result["simscode"] == "200"]
        assert bldg_200["readingvalue"].iloc[0] == 5.0

    def test_preserves_different_timestamps(self):
        """Same building, different timestamps should stay separate."""
        df = pd.DataFrame({
            "simscode": ["100", "100"],
            "readingtime": pd.to_datetime(["2025-09-01 00:00", "2025-09-01 00:15"]),
            "readingvalue": [10.0, 15.0],
        })
        result = aggregate_building_meters(df)
        assert len(result) == 2

    def test_single_meter_passthrough(self):
        """Single meter per building should pass through unchanged."""
        df = pd.DataFrame({
            "simscode": ["100"],
            "readingtime": pd.to_datetime(["2025-09-01 00:00"]),
            "readingvalue": [42.0],
        })
        result = aggregate_building_meters(df)
        assert len(result) == 1
        assert result["readingvalue"].iloc[0] == 42.0


class TestEngineerTimeFeatures:
    def test_basic_features(self):
        """Check hour, day_of_week, is_weekend are correct."""
        df = pd.DataFrame({
            "readingtime": pd.to_datetime([
                "2025-09-01 14:00",  # Monday
                "2025-09-06 08:00",  # Saturday
                "2025-09-07 23:00",  # Sunday
            ]),
        })
        result = engineer_time_features(df)
        assert list(result["hour_of_day"]) == [14, 8, 23]
        assert list(result["day_of_week"]) == [0, 5, 6]  # Mon=0, Sat=5, Sun=6
        assert list(result["is_weekend"]) == [0, 1, 1]

    def test_does_not_modify_input(self):
        """Should return a copy, not modify the input."""
        df = pd.DataFrame({
            "readingtime": pd.to_datetime(["2025-09-01 14:00"]),
        })
        result = engineer_time_features(df)
        assert "hour_of_day" not in df.columns
        assert "hour_of_day" in result.columns
