"""
모델 학습 및 예측 단위 테스트.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from src.data.schema import TARGET
from src.features.pipeline import build_pipeline, load_pipeline, prepare_dataframe, save_pipeline
from src.models.train import compute_metrics


class TestComputeMetrics:
    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = compute_metrics(y, y)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_returns_expected_keys(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 1.5, 3.5])
        metrics = compute_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {"rmse", "mae", "r2"}

    def test_rmse_non_negative(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 1.0, 4.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0


class TestBuildPipeline:
    def test_fit_predict(self, sample_df: pd.DataFrame) -> None:
        X = prepare_dataframe(sample_df)
        y = sample_df[TARGET]
        pipeline = build_pipeline(DecisionTreeRegressor(random_state=42))
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)
        assert all(isinstance(p, (int, float, np.floating)) for p in preds)

    def test_pipeline_has_two_steps(self) -> None:
        pipeline = build_pipeline(DecisionTreeRegressor())
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps


class TestSaveLoadPipeline:
    def test_round_trip(self, trained_pipeline, sample_df: pd.DataFrame) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_pipeline(trained_pipeline, path)
            loaded = load_pipeline(path)
            X = prepare_dataframe(sample_df)
            original_preds = trained_pipeline.predict(X)
            loaded_preds = loaded.predict(X)
            np.testing.assert_array_almost_equal(original_preds, loaded_preds)
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_pipeline("/tmp/nonexistent_model_12345.pkl")
