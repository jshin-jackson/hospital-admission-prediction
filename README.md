# 병원 입원일수 예측 시스템

환자의 나이, 성별, 병명, 기저질환 등 기본 정보를 입력하면 **예상 입원일수를 자동으로 예측**해주는 머신러닝 시스템입니다.

---

## 이 프로젝트는 무엇인가요?

### 쉽게 말하면

> "이 환자는 병원에 며칠이나 있어야 할까?"를 컴퓨터가 예측하는 시스템입니다.

예를 들어, 55세 남성이 폐렴(pneumonia)으로 입원하고 고혈압 기저질환이 있다면 — 이 시스템은 과거 수많은 환자 데이터를 학습해서 **"약 8일 입원할 것 같다"** 고 예측합니다.

### 왜 만들었나요?

병원에서 입원일수를 미리 예측할 수 있다면:
- 병상(침대) 수를 미리 계획할 수 있습니다
- 의료진과 자원을 효율적으로 배치할 수 있습니다
- 환자와 보호자에게 퇴원 일정을 미리 안내할 수 있습니다

### AI/머신러닝 입문자를 위한 설명

이 프로젝트는 **지도학습(Supervised Learning)** 의 한 종류인 **회귀(Regression)** 를 사용합니다.

| 용어 | 쉬운 설명 |
|------|-----------|
| 지도학습 | 정답이 있는 데이터로 모델을 훈련시키는 방법 |
| 회귀 | 숫자를 예측하는 것 (예: 입원일수 = 8.3일) |
| 피처(Feature) | 예측에 사용하는 입력 정보 (나이, 병명 등) |
| 타겟(Target) | 예측하고 싶은 정답값 (입원일수) |
| 모델 학습 | 수많은 과거 데이터에서 패턴을 찾아내는 과정 |
| 모델 저장 | 학습한 패턴을 파일로 저장해 나중에 재사용 |

### 전체 흐름 한눈에 보기

```
1. 데이터 준비
   실제 환자 데이터가 없으므로 AI(CTGAN)로 가상 데이터 10만 건 생성
          ↓
2. 모델 학습
   4가지 알고리즘으로 학습 후 가장 정확한 모델 자동 선택
          ↓
3. 모델 저장
   학습된 모델을 파일(best_model.pkl)로 저장
          ↓
4. 예측 사용
   저장된 모델로 새로운 환자 정보 → 입원일수 예측
   (터미널 또는 API 서버로 사용)
```

### 이 프로젝트에서 배울 수 있는 것

- **합성 데이터 생성**: 실제 데이터 없이도 AI로 학습 데이터 만들기 (CTGAN)
- **데이터 전처리**: 범주형(텍스트) 데이터를 숫자로 변환하는 방법 (One-Hot Encoding)
- **모델 비교**: 여러 알고리즘을 자동으로 비교해 최적 모델 선택 (GridSearchCV)
- **실험 추적**: 학습 결과를 기록하고 비교하는 방법 (MLflow)
- **API 서버**: 모델을 웹 서비스로 배포하는 방법 (FastAPI)
- **프로젝트 구조**: 실제 ML 프로젝트에서 사용하는 폴더 구조와 코드 분리

---

## 이 시스템이 하는 일

```
환자 정보 입력
(나이, 성별, 병명 등)
       ↓
  머신러닝 모델
       ↓
예상 입원일수 출력
```

예측 방법은 두 가지입니다.
- **터미널(CLI)**: 명령어로 바로 예측
- **REST API**: 다른 서비스와 연동 가능한 웹 서버

---

## 시작하기 전에 필요한 것

- Python 3.11
- Homebrew (macOS 패키지 관리자)

---

## 설치 및 실행 순서

### 0단계. 프로젝트 폴더로 이동

```bash
cd hospital-admission-prediction
```

---

### 1단계. 가상환경 활성화 및 패키지 설치

> 가상환경이란? 이 프로젝트에서만 사용하는 독립적인 Python 공간입니다. 다른 프로젝트와 충돌을 방지합니다.

```bash
# 가상환경 활성화
source .venv/bin/activate

# XGBoost 의존 라이브러리 설치 (macOS 한정)
brew install libomp

# 패키지 설치
pip install -r requirements.txt
```

설치가 완료되면 터미널 프롬프트 앞에 `(.venv)` 표시가 나타납니다.

---

### 2단계. 환경 변수 파일 생성

```bash
cp .env.example .env
```

> `.env` 파일에는 MLflow 서버 주소, 모델 파일 경로 등 설정값이 담깁니다.  
> 기본값 그대로 사용해도 됩니다.

---

### 3단계. 학습용 데이터 생성

실제 환자 데이터 대신, AI(CTGAN)로 합성 데이터를 만들어 학습에 사용합니다.

```bash
python -m src.data.generate
```

> 완료까지 약 5~10분 소요됩니다. (CTGAN 모델 학습 포함)  
> 빠르게 테스트하려면 샘플 수를 줄이세요:
> ```bash
> python -m src.data.generate --n-samples 10000
> ```

완료 후 생성되는 파일:
- `data/raw/seed_data.csv` — 70건의 시드 데이터
- `data/synthetic/hospital_data.csv` — 10만 건의 합성 데이터

---

### 4단계. MLflow 서버 실행

MLflow는 모델 학습 과정과 성능 지표를 기록·비교하는 실험 추적 도구입니다.  
**새 터미널 탭을 열어서** 실행하세요.

```bash
# 새 터미널에서
cd hospital-admission-prediction
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

실행 후 브라우저에서 확인: [http://localhost:5000](http://localhost:5000)

---

### 5단계. 모델 학습

원래 터미널로 돌아와서 실행합니다.  
4가지 모델을 자동으로 비교해 가장 좋은 모델을 저장합니다.

```bash
python -m src.models.train
```

학습하는 모델 목록:
| 모델 | 특징 |
|------|------|
| DecisionTree | 단순하고 빠름 |
| RandomForest | 여러 트리를 조합해 정확도 향상 |
| GradientBoosting | 오차를 반복적으로 줄여나감 |
| XGBoost | 대회에서 자주 쓰이는 고성능 모델 |

완료 후:
- `models/best_model.pkl` — 최적 모델 파일 저장
- [http://localhost:5000](http://localhost:5000) 에서 4개 모델 성능 비교 확인 가능

---

### 6단계. 모델 성능 확인 (선택)

```bash
python -m src.models.evaluate
```

아래 3개 그래프가 `reports/figures/` 폴더에 저장됩니다:
- `pred_vs_actual.png` — 예측값 vs 실제값 비교
- `residuals.png` — 오차 분포
- `feature_importance.png` — 어떤 피처가 예측에 중요한지

---

### 7단계. 예측하기

#### 방법 A: 터미널(CLI)로 바로 예측

```bash
python -m cli.predict \
    --age 55 \
    --gender M \
    --diagnosis pneumonia \
    --comorbidity hypertension \
    --num-medications 4 \
    --prior-admissions 2 \
    --lab-result-abnormal yes
```

출력 예시:
```
예측 결과
------------------------------
  나이              : 55세
  성별              : 남성
  병명              : pneumonia
  기저질환          : hypertension
  복용 약물 수      : 4개
  과거 입원 횟수    : 2회
  검사 이상 여부    : yes
------------------------------
  예상 입원일수     : 8.3일 (반올림: 8일)
  사용 모델         : models/best_model.pkl
```

#### 방법 B: API 서버로 예측

**서버 실행** (새 터미널에서):
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**브라우저에서 테스트**: [http://localhost:8000/docs](http://localhost:8000/docs)

Swagger UI가 열리면 `/predict` 항목에서 값을 입력하고 "Execute" 버튼을 눌러 바로 테스트할 수 있습니다.

**curl로 테스트**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": "M",
    "diagnosis": "pneumonia",
    "comorbidity": "hypertension",
    "num_medications": 4,
    "prior_admissions": 2,
    "lab_result_abnormal": "yes"
  }'
```

응답:
```json
{
  "predicted_days": 8.3,
  "predicted_days_rounded": 8,
  "input_summary": {
    "age": 55,
    "gender": "M",
    "diagnosis": "pneumonia",
    "comorbidity": "hypertension",
    "lab_result_abnormal": "yes"
  }
}
```

---

## 입력 가능한 값 정리

| 항목 | 입력 가능한 값 | 예시 |
|------|--------------|------|
| `age` | 1 ~ 100 (숫자) | `45` |
| `gender` | `M` (남성), `F` (여성) | `M` |
| `diagnosis` | `cold` `flu` `asthma` `pneumonia` `diabetes` `hypertension` `heart_disease` | `flu` |
| `comorbidity` | `none` `diabetes` `hypertension` `heart_disease` `obesity` | `none` |
| `num_medications` | 0 ~ 15 (숫자) | `3` |
| `prior_admissions` | 0 ~ 10 (숫자) | `1` |
| `lab_result_abnormal` | `yes`, `no` | `no` |

---

## 테스트 실행

코드가 올바르게 동작하는지 자동으로 검사합니다.

```bash
pytest tests/ -v
```

---

## 전체 실행 순서 요약

```
[터미널 1]                         [터미널 2]           [터미널 3]
source .venv/bin/activate          MLflow 서버 실행     FastAPI 서버 실행
python -m src.data.generate    →   mlflow server    →   uvicorn src.api.main:app
python -m src.models.train
python -m cli.predict ...
```

---

## 프로젝트 구조

```
hospital-admission-prediction/
├── data/
│   ├── raw/                  # CTGAN 학습용 시드 데이터 (자동 생성)
│   └── synthetic/            # 생성된 합성 학습 데이터
├── src/
│   ├── data/
│   │   ├── schema.py         # 피처/병명 상수 정의
│   │   └── generate.py       # 합성 데이터 생성 스크립트
│   ├── features/
│   │   └── pipeline.py       # 데이터 전처리 파이프라인
│   ├── models/
│   │   ├── train.py          # 모델 학습 + MLflow 실험 기록
│   │   └── evaluate.py       # 성능 평가 + 그래프 생성
│   └── api/
│       ├── main.py           # FastAPI 서버
│       ├── schemas.py        # 입력/출력 데이터 형식 정의
│       └── routers/
│           └── predict.py    # /predict 엔드포인트
├── cli/
│   └── predict.py            # 터미널용 예측 스크립트
├── tests/                    # 자동화 테스트 (pytest)
├── models/                   # 저장된 모델 파일 (.pkl)
├── reports/figures/          # 평가 그래프 이미지
├── mlruns/                   # MLflow 실험 기록
├── requirements.txt          # 필요한 패키지 목록
└── .env.example              # 환경 변수 설정 예시
```
