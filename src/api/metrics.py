"""
Prometheus 커스텀 메트릭 정의 모듈.

이 파일에서 정의한 메트릭 객체를 predict.py, main.py에서 임포트해 사용합니다.
모듈 레벨에서 한 번만 생성해야 중복 등록 에러를 방지할 수 있습니다.

메트릭 종류:
    Counter   : 단조 증가하는 값 (요청 수, 에러 수 등)
    Histogram : 값의 분포를 측정 (응답 시간, 예측값 등)
    Gauge     : 올라가거나 내려갈 수 있는 값 (현재 연결 수, 모델 로드 여부 등)
    Info      : 레이블로만 구성된 정적 정보 (모델 종류 등)
"""

from prometheus_client import Counter, Gauge, Histogram

# ─────────────────────────────────────────────
# 예측 관련 메트릭
# ─────────────────────────────────────────────

# 예측 요청 수: 병명(diagnosis) 레이블별로 집계
# 사용 예: PREDICTION_REQUESTS.labels(diagnosis="pneumonia").inc()
PREDICTION_REQUESTS = Counter(
    name="hospital_prediction_requests_total",
    documentation="병명별 예측 요청 총 횟수",
    labelnames=["diagnosis"],
)

# 예측 입원일수 분포: 히스토그램으로 분포를 기록
# buckets: 1일, 3일, 5일, 7일, 10일, 14일, 21일, 30일 구간으로 나눔
# 사용 예: PREDICTION_DAYS.observe(predicted_days)
PREDICTION_DAYS = Histogram(
    name="hospital_prediction_days",
    documentation="예측된 입원일수 분포",
    buckets=[1, 3, 5, 7, 10, 14, 21, 30],
)

# 예측 실패 수 카운터
# 사용 예: PREDICTION_ERRORS.inc()
PREDICTION_ERRORS = Counter(
    name="hospital_prediction_errors_total",
    documentation="예측 실패(에러) 총 횟수",
)

# ─────────────────────────────────────────────
# 모델 관련 메트릭
# ─────────────────────────────────────────────

# 모델 로드 여부: 1=로드됨, 0=미로드
# 사용 예: MODEL_LOADED.set(1)
MODEL_LOADED = Gauge(
    name="hospital_model_loaded",
    documentation="모델 로드 여부 (1=로드됨, 0=미로드)",
)

# 모델 종류 정보: 레이블로 모델 이름을 기록하고 값은 항상 1
# 사용 예: MODEL_INFO.labels(model_type="GradientBoostingRegressor").set(1)
MODEL_INFO = Gauge(
    name="hospital_model_info",
    documentation="현재 로드된 모델 종류 (레이블로 표시)",
    labelnames=["model_type"],
)
