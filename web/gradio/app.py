"""
병원 입원일수 예측 - Gradio Web UI

FastAPI 백엔드(localhost:8000)에 HTTP 요청을 보내 예측 결과를 화면에 표시합니다.
Gradio는 AI 모델 데모에 특화된 웹 UI 라이브러리입니다.
Hugging Face Spaces에 배포할 때 주로 사용합니다.

실행 방법:
    python web/gradio/app.py

사전 조건:
    FastAPI 서버가 먼저 실행되어 있어야 합니다.
    uvicorn src.api.main:app --reload
"""

import json

import gradio as gr
import requests

# ─────────────────────────────────────────────
# 설정 상수
# ─────────────────────────────────────────────

# FastAPI 서버 주소
API_URL = "http://localhost:8000"

# 지원하는 병명 목록
DIAGNOSES = ["cold", "flu", "asthma", "pneumonia", "diabetes", "hypertension", "heart_disease"]

# 기저질환 목록
COMORBIDITIES = ["none", "diabetes", "hypertension", "heart_disease", "obesity"]


# ─────────────────────────────────────────────
# 서버 상태 확인 함수
# ─────────────────────────────────────────────

def get_server_status() -> str:
    """
    FastAPI 서버 상태를 확인하고 상태 메시지를 반환합니다.
    Gradio UI 상단의 상태 표시줄에 사용됩니다.
    """
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        data = resp.json()
        if data.get("status") == "ok" and data.get("model_loaded"):
            return "✅ 서버 정상 | 모델 로드됨"
        elif data.get("status") == "ok":
            return "⚠️ 서버 정상 | 모델 미로드"
        else:
            return "❌ 서버 오류"
    except Exception:
        return "❌ 서버 연결 실패 — `uvicorn src.api.main:app --reload` 로 서버를 실행하세요"


# ─────────────────────────────────────────────
# 핵심 예측 함수
# ─────────────────────────────────────────────

def predict(
    age: int,
    gender: str,
    diagnosis: str,
    comorbidity: str,
    num_medications: int,
    prior_admissions: int,
    lab_result_abnormal: str,
) -> tuple[str, str, str]:
    """
    환자 정보를 받아 FastAPI /predict 엔드포인트에 요청을 보내고 결과를 반환합니다.

    Args:
        age: 환자 나이
        gender: 성별 (M/F)
        diagnosis: 병명
        comorbidity: 기저질환
        num_medications: 복용 약물 수
        prior_admissions: 과거 입원 횟수
        lab_result_abnormal: 검사 이상 여부 (yes/no)

    Returns:
        (예측 결과 문자열, 입력 요약 JSON 문자열, 서버 상태 문자열)
    """
    # 서버 상태 확인
    status = get_server_status()

    # API 요청 데이터 구성
    payload = {
        "age": int(age),
        "gender": gender,
        "diagnosis": diagnosis,
        "comorbidity": comorbidity,
        "num_medications": int(num_medications),
        "prior_admissions": int(prior_admissions),
        "lab_result_abnormal": lab_result_abnormal,
    }

    try:
        # POST 요청으로 예측 실행
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

        if resp.status_code == 200:
            result = resp.json()
            predicted = result["predicted_days"]
            rounded = result["predicted_days_rounded"]

            # 결과 메시지 포맷팅
            result_text = (
                f"🏥 예상 입원일수: {predicted:.1f}일\n"
                f"📅 반올림 입원일수: {rounded}일\n\n"
                f"입력 병명: {diagnosis} | 나이: {age}세 | 성별: {'남성' if gender == 'M' else '여성'}"
            )

            # 입력 요약을 보기 좋은 JSON 형태로 반환
            summary_text = json.dumps(
                result["input_summary"],
                ensure_ascii=False,
                indent=2,
            )

            return result_text, summary_text, status

        else:
            # HTTP 에러 처리 (422 유효성 검사 실패 등)
            detail = resp.json().get("detail", "알 수 없는 오류")
            return f"❌ 예측 실패 (HTTP {resp.status_code}): {detail}", "", status

    except requests.exceptions.ConnectionError:
        # 서버가 실행되지 않은 경우
        error_msg = (
            "❌ FastAPI 서버에 연결할 수 없습니다.\n\n"
            "아래 명령어로 서버를 먼저 실행해주세요:\n"
            "uvicorn src.api.main:app --reload"
        )
        return error_msg, "", "❌ 서버 연결 실패"

    except Exception as e:
        return f"❌ 예상치 못한 오류: {e}", "", status


# ─────────────────────────────────────────────
# 예시 데이터 (Examples 탭에 표시됨)
# ─────────────────────────────────────────────
EXAMPLES = [
    [55, "M", "pneumonia", "hypertension", 4, 2, "yes"],
    [30, "F", "cold", "none", 1, 0, "no"],
    [70, "M", "heart_disease", "diabetes", 10, 5, "yes"],
    [25, "F", "flu", "none", 2, 0, "no"],
    [60, "M", "asthma", "obesity", 6, 3, "yes"],
]


# ─────────────────────────────────────────────
# Gradio UI 구성
# ─────────────────────────────────────────────

# gr.Blocks: 자유롭게 레이아웃을 구성할 수 있는 컨테이너
# Gradio 6.0부터 theme, css는 launch()에 전달해야 합니다
with gr.Blocks(title="병원 입원일수 예측") as demo:

    # ── 상단 헤더 ──────────────────────────────
    gr.Markdown("# 🏥 병원 입원일수 예측 시스템")
    gr.Markdown(
        "환자 정보를 입력하면 머신러닝 모델이 **예상 입원일수**를 예측합니다.\n\n"
        f"> FastAPI 서버: `{API_URL}` | [API 문서]({API_URL}/docs)"
    )

    # 서버 상태 표시 (읽기 전용 텍스트박스)
    server_status = gr.Textbox(
        label="서버 상태",
        value=get_server_status(),
        interactive=False,
        max_lines=1,
    )

    gr.Markdown("---")

    # ── 입력 폼: 2열 레이아웃 ──────────────────
    with gr.Row():

        # 왼쪽 열: 기본 정보
        with gr.Column():
            gr.Markdown("### 기본 정보")

            age = gr.Slider(
                label="나이",
                minimum=1,
                maximum=100,
                value=45,
                step=1,
                info="환자의 나이 (1~100세)",
            )

            gender = gr.Radio(
                label="성별",
                choices=[("남성", "M"), ("여성", "F")],
                value="M",
            )

            diagnosis = gr.Dropdown(
                label="병명",
                choices=DIAGNOSES,
                value="pneumonia",
                info="주요 진단명을 선택하세요",
            )

            lab_result_abnormal = gr.Radio(
                label="검사 이상 여부",
                choices=[("정상", "no"), ("이상", "yes")],
                value="no",
                info="혈액검사 등 임상검사 이상 여부",
            )

        # 오른쪽 열: 임상 정보
        with gr.Column():
            gr.Markdown("### 임상 정보")

            comorbidity = gr.Dropdown(
                label="기저질환",
                choices=COMORBIDITIES,
                value="none",
                info="만성 기저질환을 선택하세요",
            )

            num_medications = gr.Slider(
                label="복용 약물 수",
                minimum=0,
                maximum=15,
                value=3,
                step=1,
                info="현재 복용 중인 약물 수",
            )

            prior_admissions = gr.Slider(
                label="과거 입원 횟수",
                minimum=0,
                maximum=10,
                value=1,
                step=1,
                info="과거 총 입원 횟수",
            )

    # ── 예측 버튼 ──────────────────────────────
    predict_btn = gr.Button("🔍 입원일수 예측하기", variant="primary", size="lg")

    gr.Markdown("---")

    # ── 결과 출력 영역 ─────────────────────────
    gr.Markdown("### 📊 예측 결과")

    with gr.Row():
        # 예측 결과 텍스트
        result_output = gr.Textbox(
            label="예측 결과",
            interactive=False,
            lines=4,
            placeholder="예측 버튼을 클릭하면 결과가 여기에 표시됩니다.",
        )

        # 입력 요약 JSON
        summary_output = gr.Textbox(
            label="입력 요약 (JSON)",
            interactive=False,
            lines=4,
        )

    # ── 예시 데이터 ────────────────────────────
    gr.Markdown("### 📝 예시 데이터")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[age, gender, diagnosis, comorbidity, num_medications, prior_admissions, lab_result_abnormal],
        label="클릭하면 자동 입력됩니다",
    )

    # ── 이벤트 연결: 버튼 클릭 → predict 함수 호출 ──
    predict_btn.click(
        fn=predict,
        inputs=[age, gender, diagnosis, comorbidity, num_medications, prior_admissions, lab_result_abnormal],
        outputs=[result_output, summary_output, server_status],
    )


# ─────────────────────────────────────────────
# 앱 실행 (직접 실행 시)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_port=7860,                                          # Gradio 기본 포트
        share=False,                                               # True로 바꾸면 공개 URL 생성
        show_error=True,                                           # 에러 메시지를 UI에 표시
        theme=gr.themes.Soft(),                                    # Gradio 6.0: theme은 launch()에 전달
        css=".gradio-container { max-width: 900px !important; }",  # Gradio 6.0: css도 launch()에 전달
    )
