"""
병원 입원일수 예측 - Streamlit Web UI

FastAPI 백엔드(localhost:8000)에 HTTP 요청을 보내 예측 결과를 화면에 표시합니다.
Streamlit은 Python 코드만으로 웹 UI를 만들 수 있는 라이브러리입니다.

실행 방법:
    streamlit run web/streamlit/app.py

사전 조건:
    FastAPI 서버가 먼저 실행되어 있어야 합니다.
    uvicorn src.api.main:app --reload
"""

import json

import requests
import streamlit as st

# ─────────────────────────────────────────────
# 설정 상수
# ─────────────────────────────────────────────

# FastAPI 서버 주소 (환경에 따라 변경 가능)
API_URL = "http://localhost:8000"

# 선택 가능한 병명 목록 (한글 레이블 → 영문 값 매핑)
DIAGNOSIS_OPTIONS = {
    "감기 (cold)": "cold",
    "독감 (flu)": "flu",
    "천식 (asthma)": "asthma",
    "폐렴 (pneumonia)": "pneumonia",
    "당뇨병 (diabetes)": "diabetes",
    "고혈압 (hypertension)": "hypertension",
    "심장질환 (heart_disease)": "heart_disease",
}

# 기저질환 목록 (한글 레이블 → 영문 값 매핑)
COMORBIDITY_OPTIONS = {
    "없음 (none)": "none",
    "당뇨병 (diabetes)": "diabetes",
    "고혈압 (hypertension)": "hypertension",
    "심장질환 (heart_disease)": "heart_disease",
    "비만 (obesity)": "obesity",
}


# ─────────────────────────────────────────────
# 페이지 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="병원 입원일수 예측",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────

def check_server_health() -> dict:
    """
    FastAPI 서버의 헬스체크 엔드포인트를 호출합니다.
    서버가 정상이면 {"status": "ok", "model_loaded": true} 를 반환합니다.
    """
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.json()
    except Exception:
        return {"status": "error", "model_loaded": False}


def get_model_info() -> dict | None:
    """
    현재 로드된 모델 정보를 FastAPI 서버에서 가져옵니다.
    """
    try:
        resp = requests.get(f"{API_URL}/model/info", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def call_predict_api(payload: dict) -> dict | None:
    """
    예측 API(/predict)에 환자 데이터를 POST 요청으로 보내고 결과를 반환합니다.
    """
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            # 422 등 유효성 검사 오류 메시지 추출
            detail = resp.json().get("detail", "알 수 없는 오류")
            st.error(f"예측 실패 (HTTP {resp.status_code}): {detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ FastAPI 서버에 연결할 수 없습니다.\n\n"
            "아래 명령어로 서버를 먼저 실행해주세요:\n"
            "```\nuvicorn src.api.main:app --reload\n```"
        )
        return None
    except Exception as e:
        st.error(f"예상치 못한 오류: {e}")
        return None


# ─────────────────────────────────────────────
# 사이드바: 서버 상태 및 모델 정보
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 시스템 상태")

    # 서버 헬스체크 결과 표시
    health = check_server_health()
    if health.get("status") == "ok":
        st.success("✅ API 서버 정상")
        # 모델 로드 여부 추가 확인
        if health.get("model_loaded"):
            st.success("✅ 모델 로드됨")
        else:
            st.warning("⚠️ 모델 미로드 — 먼저 학습을 실행하세요")
    else:
        st.error("❌ API 서버 연결 실패")
        st.info(
            "아래 명령어로 서버를 실행하세요:\n"
            "```\nuvicorn src.api.main:app --reload\n```"
        )

    st.divider()

    # 모델 상세 정보
    st.header("📋 모델 정보")
    model_info = get_model_info()
    if model_info:
        st.write(f"**모델 종류**: `{model_info['model_type']}`")
        st.write(f"**파일 경로**: `{model_info['model_path']}`")
        with st.expander("사용 피처 목록"):
            for feature in model_info.get("features", []):
                st.write(f"- `{feature}`")
    else:
        st.info("모델 정보를 불러올 수 없습니다.")

    st.divider()
    st.caption(f"API 서버: `{API_URL}`")
    st.caption("[API 문서 열기](" + API_URL + "/docs)")


# ─────────────────────────────────────────────
# 메인 화면
# ─────────────────────────────────────────────
st.title("🏥 병원 입원일수 예측 시스템")
st.markdown(
    "환자 정보를 입력하면 머신러닝 모델이 **예상 입원일수**를 예측합니다."
)
st.divider()

# ─────────────────────────────────────────────
# 입력 폼: 2열 레이아웃
# ─────────────────────────────────────────────
st.subheader("👤 환자 정보 입력")

col_left, col_right = st.columns(2)

with col_left:
    # 나이 입력 (숫자 입력 위젯)
    age = st.number_input(
        "나이",
        min_value=1,
        max_value=100,
        value=45,
        step=1,
        help="환자의 나이 (1~100세)",
    )

    # 성별 선택 (라디오 버튼)
    gender_label = st.radio(
        "성별",
        options=["남성 (M)", "여성 (F)"],
        horizontal=True,
    )
    # 라디오 선택값에서 영문 코드 추출
    gender = "M" if "M" in gender_label else "F"

    # 병명 선택 (드롭다운)
    diagnosis_label = st.selectbox(
        "병명",
        options=list(DIAGNOSIS_OPTIONS.keys()),
        index=3,  # 기본값: 폐렴
        help="주요 진단명을 선택하세요",
    )
    diagnosis = DIAGNOSIS_OPTIONS[diagnosis_label]

    # 검사 이상 여부 (라디오 버튼)
    lab_label = st.radio(
        "검사 이상 여부",
        options=["정상 (no)", "이상 (yes)"],
        horizontal=True,
        help="혈액검사, 영상검사 등 임상검사 이상 여부",
    )
    lab_result_abnormal = "yes" if "yes" in lab_label else "no"

with col_right:
    # 기저질환 선택 (드롭다운)
    comorbidity_label = st.selectbox(
        "기저질환",
        options=list(COMORBIDITY_OPTIONS.keys()),
        index=0,  # 기본값: 없음
        help="만성 기저질환을 선택하세요",
    )
    comorbidity = COMORBIDITY_OPTIONS[comorbidity_label]

    # 복용 약물 수 (슬라이더)
    num_medications = st.slider(
        "복용 약물 수",
        min_value=0,
        max_value=15,
        value=3,
        help="현재 복용 중인 약물 수 (0~15)",
    )

    # 과거 입원 횟수 (슬라이더)
    prior_admissions = st.slider(
        "과거 입원 횟수",
        min_value=0,
        max_value=10,
        value=1,
        help="과거 총 입원 횟수 (0~10)",
    )

st.divider()

# ─────────────────────────────────────────────
# 예측 버튼
# ─────────────────────────────────────────────
if st.button("🔍 입원일수 예측하기", type="primary", use_container_width=True):

    # API에 전달할 데이터 딕셔너리 구성
    payload = {
        "age": age,
        "gender": gender,
        "diagnosis": diagnosis,
        "comorbidity": comorbidity,
        "num_medications": num_medications,
        "prior_admissions": prior_admissions,
        "lab_result_abnormal": lab_result_abnormal,
    }

    # 로딩 스피너를 보여주면서 API 호출
    with st.spinner("모델이 예측 중입니다..."):
        result = call_predict_api(payload)

    # ─────────────────────────────────────────
    # 예측 결과 표시
    # ─────────────────────────────────────────
    if result:
        st.subheader("📊 예측 결과")

        # 핵심 지표를 2개의 메트릭 카드로 표시
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label="예상 입원일수",
                value=f"{result['predicted_days']:.1f}일",
            )
        with col_b:
            st.metric(
                label="반올림 입원일수",
                value=f"{result['predicted_days_rounded']}일",
            )

        # 입력 요약 (접이식 패널)
        with st.expander("📋 입력 요약 보기"):
            st.json(result["input_summary"])
