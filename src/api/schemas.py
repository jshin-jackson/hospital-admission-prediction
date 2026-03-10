"""
FastAPI 요청/응답 Pydantic v2 스키마 정의.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.data.schema import (
    AGE_MAX,
    AGE_MIN,
    COMORBIDITIES,
    DIAGNOSES,
    GENDERS,
    NUM_MEDICATIONS_MAX,
    NUM_MEDICATIONS_MIN,
    PRIOR_ADMISSIONS_MAX,
    PRIOR_ADMISSIONS_MIN,
)

DiagnosisType = Literal[
    "cold", "flu", "asthma", "pneumonia", "diabetes", "hypertension", "heart_disease"
]
GenderType = Literal["M", "F"]
ComorbidityType = Literal["none", "diabetes", "hypertension", "heart_disease", "obesity"]
LabResultType = Literal["yes", "no"]


class PredictRequest(BaseModel):
    """입원일수 예측 요청 스키마."""

    age: int = Field(
        ...,
        ge=AGE_MIN,
        le=AGE_MAX,
        description=f"환자 나이 ({AGE_MIN}~{AGE_MAX})",
        examples=[45],
    )
    gender: GenderType = Field(
        ...,
        description="성별 (M: 남성, F: 여성)",
        examples=["M"],
    )
    diagnosis: DiagnosisType = Field(
        ...,
        description=f"주요 병명 ({', '.join(DIAGNOSES)})",
        examples=["pneumonia"],
    )
    comorbidity: ComorbidityType = Field(
        default="none",
        description=f"기저질환 ({', '.join(COMORBIDITIES)})",
        examples=["hypertension"],
    )
    num_medications: int = Field(
        default=0,
        ge=NUM_MEDICATIONS_MIN,
        le=NUM_MEDICATIONS_MAX,
        description=f"복용 약물 수 ({NUM_MEDICATIONS_MIN}~{NUM_MEDICATIONS_MAX})",
        examples=[3],
    )
    prior_admissions: int = Field(
        default=0,
        ge=PRIOR_ADMISSIONS_MIN,
        le=PRIOR_ADMISSIONS_MAX,
        description=f"과거 입원 횟수 ({PRIOR_ADMISSIONS_MIN}~{PRIOR_ADMISSIONS_MAX})",
        examples=[1],
    )
    lab_result_abnormal: LabResultType = Field(
        default="no",
        description="검사 이상 여부 (yes/no)",
        examples=["yes"],
    )

    @field_validator("diagnosis")
    @classmethod
    def validate_diagnosis(cls, v: str) -> str:
        if v not in DIAGNOSES:
            raise ValueError(f"지원하지 않는 병명: {v}. 가능한 값: {DIAGNOSES}")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in GENDERS:
            raise ValueError(f"지원하지 않는 성별: {v}. 가능한 값: {GENDERS}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 55,
                "gender": "M",
                "diagnosis": "pneumonia",
                "comorbidity": "hypertension",
                "num_medications": 4,
                "prior_admissions": 2,
                "lab_result_abnormal": "yes",
            }
        }
    }


class PredictResponse(BaseModel):
    """입원일수 예측 응답 스키마."""

    predicted_days: float = Field(..., description="예측 입원일수")
    predicted_days_rounded: int = Field(..., description="반올림된 예측 입원일수")
    input_summary: dict = Field(..., description="입력값 요약")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_days": 8.3,
                "predicted_days_rounded": 8,
                "input_summary": {
                    "age": 55,
                    "gender": "M",
                    "diagnosis": "pneumonia",
                },
            }
        }
    }


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    model_path: str
    features: list[str]
