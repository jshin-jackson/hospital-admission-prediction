"""
다중 모델 학습, GridSearchCV 하이퍼파라미터 튜닝, MLflow 실험 추적.

지원 모델:
    - DecisionTreeRegressor
    - RandomForestRegressor
    - GradientBoostingRegressor
    - XGBRegressor

사용법:
    python -m src.models.train \
        [--data data/synthetic/hospital_data.csv] \
        [--output models/best_model.pkl] \
        [--experiment "Hospital Stay Prediction"]
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.data.schema import TARGET
from src.features.pipeline import build_pipeline, save_pipeline

DEFAULT_DATA = "data/synthetic/hospital_data.csv"
DEFAULT_OUTPUT = "models/best_model.pkl"
DEFAULT_EXPERIMENT = "Hospital Stay Prediction"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


@dataclass
class ModelConfig:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]] = field(default_factory=dict)


MODEL_CONFIGS: list[ModelConfig] = [
    ModelConfig(
        name="DecisionTree",
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid={
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
        },
    ),
    ModelConfig(
        name="RandomForest",
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None],
            "model__min_samples_split": [2, 5],
        },
    ),
    ModelConfig(
        name="GradientBoosting",
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
        },
    ),
    ModelConfig(
        name="XGBoost",
        estimator=XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric="rmse",
        ),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
        },
    ),
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_log(
    config: ModelConfig,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    experiment_name: str,
) -> tuple[Any, dict[str, float]]:
    """단일 모델을 GridSearchCV로 튜닝하고 MLflow에 기록."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    pipeline = build_pipeline(config.estimator)

    cv = GridSearchCV(
        estimator=pipeline,
        param_grid=config.param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    with mlflow.start_run(run_name=config.name):
        cv.fit(X_train, y_train)
        best_pipeline = cv.best_estimator_

        y_pred = best_pipeline.predict(X_test)
        metrics = compute_metrics(y_test.to_numpy(), y_pred)

        mlflow.log_param("model_type", config.name)
        for k, v in cv.best_params_.items():
            clean_key = k.replace("model__", "")
            mlflow.log_param(clean_key, v)

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="pipeline",
            registered_model_name=f"HospitalStay_{config.name}",
        )

        print(
            f"  [{config.name}] RMSE={metrics['rmse']:.3f} "
            f"MAE={metrics['mae']:.3f} R²={metrics['r2']:.3f}"
        )
        print(f"    Best params: {cv.best_params_}")

    return best_pipeline, metrics


def run_training(
    data_path: str = DEFAULT_DATA,
    output_path: str = DEFAULT_OUTPUT,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> tuple[Any, dict[str, float]]:
    """전체 모델 비교 학습을 실행하고 최적 모델을 저장."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"학습 데이터: {len(X_train):,}건 / 테스트 데이터: {len(X_test):,}건\n")

    results: list[tuple[Any, dict[str, float], str]] = []

    for config in MODEL_CONFIGS:
        print(f">>> {config.name} 학습 중...")
        pipeline, metrics = train_and_log(
            config, X_train, X_test, y_train, y_test, experiment_name
        )
        results.append((pipeline, metrics, config.name))

    # RMSE 기준 최적 모델 선택
    best_pipeline, best_metrics, best_name = min(results, key=lambda x: x[1]["rmse"])

    print(f"\n최적 모델: {best_name}")
    print(
        f"  RMSE={best_metrics['rmse']:.3f} "
        f"MAE={best_metrics['mae']:.3f} R²={best_metrics['r2']:.3f}"
    )

    save_pipeline(best_pipeline, output_path)
    print(f"\n모델 저장 완료: {output_path}")

    return best_pipeline, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="병원 입원일수 예측 모델 학습")
    parser.add_argument("--data", default=DEFAULT_DATA, help="학습 데이터 CSV 경로")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="모델 저장 경로")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow 실험명")
    args = parser.parse_args()

    run_training(
        data_path=args.data,
        output_path=args.output,
        experiment_name=args.experiment,
    )


if __name__ == "__main__":
    main()
