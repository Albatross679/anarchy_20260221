"""Unit tests for model functions using tiny synthetic data."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import XGBoostParams
from src.model import create_model, evaluate_model, load_model, save_model, train_model


@pytest.fixture
def synthetic_data():
    """Create a small synthetic dataset: energy ~ temperature + area."""
    rng = np.random.RandomState(42)
    n = 200
    temp = rng.uniform(50, 100, n)
    area = rng.uniform(1000, 50000, n)
    noise = rng.normal(0, 0.01, n)
    # Target: simple linear relationship
    y = 0.001 * temp + 0.00001 * area + noise

    df = pd.DataFrame({"temperature_2m": temp, "grossarea": area})
    feature_cols = ["temperature_2m", "grossarea"]

    # Split: first 150 train, last 50 test
    X_train = df.iloc[:150]
    X_test = df.iloc[150:]
    y_train = pd.Series(y[:150])
    y_test = pd.Series(y[150:])

    return X_train, X_test, y_train, y_test, feature_cols


@pytest.fixture
def default_params():
    return XGBoostParams(
        n_estimators=50,
        max_depth=3,
        early_stopping_rounds=10,
    )


class TestCreateModel:
    def test_returns_model_and_fit_params(self, default_params):
        model, fit_params = create_model(default_params, seed=42)
        assert model is not None
        # XGBoost 3.x: early_stopping_rounds and eval_metric are on the model
        assert model.get_params()["eval_metric"] == "rmse"

    def test_early_stopping_on_model(self, default_params):
        model, _ = create_model(default_params)
        assert model.get_params()["early_stopping_rounds"] == 10

    def test_no_early_stopping(self):
        params = XGBoostParams(early_stopping_rounds=0)
        model, _ = create_model(params)
        assert model.get_params().get("early_stopping_rounds") is None


class TestTrainAndEvaluate:
    def test_train_runs(self, synthetic_data, default_params):
        X_train, X_test, y_train, y_test, _ = synthetic_data
        model, fit_params = create_model(default_params)
        trained = train_model(model, X_train, y_train, X_test, y_test, fit_params)
        assert trained is not None

    def test_evaluate_metrics(self, synthetic_data, default_params):
        X_train, X_test, y_train, y_test, _ = synthetic_data
        model, fit_params = create_model(default_params)
        train_model(model, X_train, y_train, X_test, y_test, fit_params)
        metrics = evaluate_model(model, X_test, y_test)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape_pct" in metrics
        assert "feature_importance" in metrics
        # Model should beat random on this simple problem
        assert metrics["r2"] > 0

    def test_evaluate_feature_importance_keys(self, synthetic_data, default_params):
        X_train, X_test, y_train, y_test, feature_cols = synthetic_data
        model, fit_params = create_model(default_params)
        train_model(model, X_train, y_train, X_test, y_test, fit_params)
        metrics = evaluate_model(model, X_test, y_test)

        for col in feature_cols:
            assert col in metrics["feature_importance"]


class TestSaveLoad:
    def test_round_trip(self, synthetic_data, default_params):
        X_train, X_test, y_train, y_test, _ = synthetic_data
        model, fit_params = create_model(default_params)
        train_model(model, X_train, y_train, X_test, y_test, fit_params)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            save_model(model, path)
            assert path.exists()

            loaded = load_model(path)
            # Predictions should match
            orig_preds = model.predict(X_test)
            loaded_preds = loaded.predict(X_test)
            np.testing.assert_array_almost_equal(orig_preds, loaded_preds)
