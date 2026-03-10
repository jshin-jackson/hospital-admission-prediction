"""
FastAPI 애플리케이션 엔트리포인트.

실행:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

환경변수:
    MODEL_PATH          모델 pkl 경로 (기본: models/best_model.pkl)
    MLFLOW_TRACKING_URI MLflow 서버 URI (기본: http://localhost:5000)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import predict as predict_router
from src.api.schemas import HealthResponse, ModelInfoResponse
from src.data.schema import ALL_FEATURES
from src.features.pipeline import load_pipeline

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """앱 시작 시 모델 로드, 종료 시 정리."""
    if os.path.exists(MODEL_PATH):
        app.state.pipeline = load_pipeline(MODEL_PATH)
        app.state.model_path = MODEL_PATH
        print(f"모델 로드 완료: {MODEL_PATH}")
    else:
        app.state.pipeline = None
        app.state.model_path = None
        print(f"경고: 모델 파일을 찾을 수 없습니다 ({MODEL_PATH})")
    yield
    app.state.pipeline = None


app = FastAPI(
    title="Hospital Admission Prediction API",
    description="환자 정보를 기반으로 예상 입원일수를 예측하는 REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """서비스 헬스체크."""
    return HealthResponse(
        status="ok",
        model_loaded=app.state.pipeline is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["system"])
async def model_info() -> ModelInfoResponse:
    """현재 로드된 모델 정보 조회."""
    if app.state.pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "모델이 로드되지 않았습니다."},
        )

    estimator = app.state.pipeline.named_steps["model"]
    model_type = type(estimator).__name__

    return ModelInfoResponse(
        model_type=model_type,
        model_path=app.state.model_path or "",
        features=ALL_FEATURES,
    )
