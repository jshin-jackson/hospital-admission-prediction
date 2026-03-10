"""
/predict 엔드포인트 라우터.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.api.metrics import PREDICTION_DAYS, PREDICTION_ERRORS, PREDICTION_REQUESTS
from src.api.schemas import PredictRequest, PredictResponse
from src.features.pipeline import prepare_dataframe

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictResponse, summary="입원일수 예측")
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """
    환자 정보를 입력받아 예측 입원일수를 반환합니다.

    - **age**: 환자 나이
    - **gender**: 성별 (M / F)
    - **diagnosis**: 주요 병명
    - **comorbidity**: 기저질환 (기본값: none)
    - **num_medications**: 복용 약물 수 (기본값: 0)
    - **prior_admissions**: 과거 입원 횟수 (기본값: 0)
    - **lab_result_abnormal**: 검사 이상 여부 (기본값: no)
    """
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 로드되지 않았습니다. 먼저 모델을 학습하고 저장하세요.",
        )

    input_dict = body.model_dump()
    df = prepare_dataframe(input_dict)

    try:
        predicted = float(pipeline.predict(df)[0])
    except Exception as exc:
        # 예측 실패 시 에러 카운터 증가
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"예측 실패: {exc}") from exc

    # ── 커스텀 Prometheus 메트릭 기록 ─────────────────────────────────────
    # 병명별 예측 요청 카운터 증가
    PREDICTION_REQUESTS.labels(diagnosis=body.diagnosis).inc()
    # 예측 입원일수를 히스토그램에 기록 (분포 파악용)
    PREDICTION_DAYS.observe(predicted)

    return PredictResponse(
        predicted_days=round(predicted, 2),
        predicted_days_rounded=max(1, round(predicted)),
        input_summary={
            "age": body.age,
            "gender": body.gender,
            "diagnosis": body.diagnosis,
            "comorbidity": body.comorbidity,
            "lab_result_abnormal": body.lab_result_abnormal,
        },
    )
