"""
/predict 엔드포인트 라우터.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

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
        raise HTTPException(status_code=500, detail=f"예측 실패: {exc}") from exc

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
