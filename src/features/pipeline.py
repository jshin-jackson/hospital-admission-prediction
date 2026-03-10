"""
전처리 파이프라인 빌더.

sklearn Pipeline + ColumnTransformer 를 조합해 수치형/범주형 피처를
일관되게 전처리한 뒤, 모델과 함께 단일 Pipeline 객체로 반환.
파이프라인 전체를 pkl로 저장하므로 학습/예측 간 전처리 불일치를 방지.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def build_preprocessor() -> ColumnTransformer:
    """수치형 + 범주형 전처리기를 ColumnTransformer로 구성."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())],
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ],
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    """전처리기 + 추정기를 하나의 Pipeline으로 조합."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def load_pipeline(path: str) -> Pipeline:
    """저장된 Pipeline을 불러온다."""
    return joblib.load(path)


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Pipeline을 pkl 파일로 저장."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(pipeline, path)


def prepare_dataframe(data: dict | pd.DataFrame) -> pd.DataFrame:
    """dict 또는 DataFrame 입력을 파이프라인 입력 형식으로 변환."""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"

    # 타겟 컬럼이 있으면 제거
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
