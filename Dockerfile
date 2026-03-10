# ── 1단계: 의존성 설치 ──────────────────────────────────────────────────────
# Python 3.11 slim 이미지를 베이스로 사용 (가벼운 이미지)
FROM python:3.11-slim AS builder

# 작업 디렉터리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 빌드 도구 설치
# libomp-dev: XGBoost가 macOS 외 환경에서 필요로 하는 OpenMP 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 먼저 복사해서 캐시 레이어 활용
COPY requirements.txt .

# pip 업그레이드 후 의존성 설치
# --no-cache-dir: 이미지 크기 줄이기 위해 캐시 사용 안 함
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── 2단계: 실행 이미지 ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# 시스템 패키지: 런타임에도 libomp 필요
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# builder 단계에서 설치된 Python 패키지만 복사 (이미지 크기 최적화)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 소스코드 복사
COPY src/ ./src/
COPY models/ ./models/

# 앱이 사용하는 포트 명시 (실제 바인딩은 run 커맨드에서 수행)
EXPOSE 8000

# 환경변수 기본값 설정
ENV MODEL_PATH=models/best_model.pkl
ENV PYTHONUNBUFFERED=1

# FastAPI 서버 실행
# --host 0.0.0.0: 컨테이너 외부에서 접근 가능하도록 설정
# --workers 1: 단일 워커 (필요시 증가 가능)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
