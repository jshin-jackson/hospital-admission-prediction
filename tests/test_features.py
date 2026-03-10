"""
전처리 파이프라인 단위 테스트.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.features.pipeline import build_preprocessor, prepare_dataframe


class TestPrepareDataframe:
    def test_returns_correct_columns(self, sample_df: pd.DataFrame) -> None:
        result = prepare_dataframe(sample_df)
        assert list(result.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES

    def test_drops_target_column(self, sample_df: pd.DataFrame) -> None:
        result = prepare_dataframe(sample_df)
        assert TARGET not in result.columns

    def test_accepts_dict_input(self) -> None:
        data = {
            "age": 45,
            "gender": "M",
            "diagnosis": "flu",
            "comorbidity": "none",
            "num_medications": 2,
            "prior_admissions": 1,
            "lab_result_abnormal": "no",
        }
        result = prepare_dataframe(data)
        assert len(result) == 1
        assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES).issubset(result.columns)

    def test_fills_missing_columns_with_defaults(self) -> None:
        partial = pd.DataFrame([{"age": 40, "diagnosis": "flu"}])
        result = prepare_dataframe(partial)
        for col in NUMERIC_FEATURES:
            assert col in result.columns
        for col in CATEGORICAL_FEATURES:
            assert col in result.columns


class TestPreprocessor:
    def test_transforms_without_error(self, sample_df: pd.DataFrame) -> None:
        X = prepare_dataframe(sample_df)
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(X)
        assert transformed.shape[0] == len(sample_df)

    def test_output_is_numeric(self, sample_df: pd.DataFrame) -> None:
        import numpy as np

        X = prepare_dataframe(sample_df)
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(X)
        assert transformed.dtype in (float, "float64", "float32") or transformed.dtype.kind == "f"

    def test_handles_unknown_categories(self, sample_df: pd.DataFrame) -> None:
        X_train = prepare_dataframe(sample_df)
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)

        unknown = pd.DataFrame(
            [
                {
                    "age": 50,
                    "num_medications": 3,
                    "prior_admissions": 1,
                    "gender": "M",
                    "diagnosis": "flu",
                    "comorbidity": "none",
                    "lab_result_abnormal": "no",
                }
            ]
        )
        result = preprocessor.transform(unknown)
        assert result.shape[0] == 1
