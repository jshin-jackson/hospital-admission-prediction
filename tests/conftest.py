"""
pytest 공통 픽스처 정의.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.tree import DecisionTreeRegressor

from src.data.schema import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.features.pipeline import build_pipeline, save_pipeline


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """테스트용 소규모 DataFrame."""
    return pd.DataFrame(
        {
            "age": [25, 45, 60, 30, 55],
            "num_medications": [2, 5, 8, 1, 3],
            "prior_admissions": [0, 1, 3, 0, 2],
            "gender": ["M", "F", "M", "F", "M"],
            "diagnosis": ["cold", "flu", "pneumonia", "asthma", "heart_disease"],
            "comorbidity": ["none", "hypertension", "diabetes", "none", "obesity"],
            "lab_result_abnormal": ["no", "yes", "yes", "no", "yes"],
            TARGET: [2, 4, 9, 3, 8],
        }
    )


@pytest.fixture
def trained_pipeline(sample_df: pd.DataFrame):
    """소규모 데이터로 학습된 Pipeline 객체."""
    from src.features.pipeline import prepare_dataframe

    X = prepare_dataframe(sample_df)
    y = sample_df[TARGET]
    pipeline = build_pipeline(DecisionTreeRegressor(random_state=42))
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def model_file(trained_pipeline) -> str:
    """임시 파일에 저장된 모델 경로."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    save_pipeline(trained_pipeline, path)
    yield path
    os.unlink(path)


@pytest.fixture
def api_client(model_file: str):
    """FastAPI TestClient (모델 로드 포함)."""
    os.environ["MODEL_PATH"] = model_file

    from src.api.main import app

    with TestClient(app) as client:
        yield client

    del os.environ["MODEL_PATH"]
