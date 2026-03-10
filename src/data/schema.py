"""
피처 정의 및 데이터 스키마 상수 모음.
합성 데이터 생성, 전처리 파이프라인, 예측 입력 검증에서 공통으로 사용.
"""

# 지원 병명 목록
DIAGNOSES = [
    "cold",
    "flu",
    "asthma",
    "pneumonia",
    "diabetes",
    "hypertension",
    "heart_disease",
]

# 성별 옵션
GENDERS = ["M", "F"]

# 기저질환 옵션 (diagnosis 와 별도로 만성 기저질환)
COMORBIDITIES = ["none", "diabetes", "hypertension", "heart_disease", "obesity"]

# 수치형 피처
NUMERIC_FEATURES = [
    "age",
    "num_medications",
    "prior_admissions",
]

# 범주형 피처
CATEGORICAL_FEATURES = [
    "gender",
    "diagnosis",
    "comorbidity",
    "lab_result_abnormal",
]

# 전체 입력 피처 (순서 고정)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# 타겟 컬럼
TARGET = "days_in_hospital"

# 나이 범위
AGE_MIN = 1
AGE_MAX = 100

# 복용 약물 수 범위
NUM_MEDICATIONS_MIN = 0
NUM_MEDICATIONS_MAX = 15

# 과거 입원 횟수 범위
PRIOR_ADMISSIONS_MIN = 0
PRIOR_ADMISSIONS_MAX = 10

# 병명별 기대 입원일수 (합성 데이터 시드에 사용)
DIAGNOSIS_BASE_DAYS: dict[str, float] = {
    "cold": 2.0,
    "flu": 3.5,
    "asthma": 4.5,
    "pneumonia": 7.0,
    "diabetes": 5.0,
    "hypertension": 4.0,
    "heart_disease": 9.0,
}
