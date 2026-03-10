/**
 * 예측 결과 표시 컴포넌트
 *
 * FastAPI에서 받은 예측 결과를 카드 형태로 보여줍니다.
 * 에러 상태와 성공 상태를 각각 다르게 표시합니다.
 */

import type { PredictResponse } from "../types";
import { COMORBIDITY_LABELS, DIAGNOSIS_LABELS } from "../types";

// ─────────────────────────────────────────────
// Props 타입 정의
// ─────────────────────────────────────────────

interface PredictResultProps {
  /** 예측 결과 (null이면 아직 예측 전) */
  result: PredictResponse | null;
  /** 에러 메시지 (null이면 에러 없음) */
  error: string | null;
}

// ─────────────────────────────────────────────
// 컴포넌트
// ─────────────────────────────────────────────

export function PredictResult({ result, error }: PredictResultProps) {

  // 에러 상태: 빨간 카드로 에러 메시지 표시
  if (error) {
    return (
      <div className="result-card result-card--error">
        <h2 className="result-title">❌ 오류 발생</h2>
        <p className="result-error-msg">{error}</p>
        <p className="result-hint">
          FastAPI 서버가 실행 중인지 확인하세요:
          <br />
          <code>uvicorn src.api.main:app --reload</code>
        </p>
      </div>
    );
  }

  // 초기 상태: 안내 메시지
  if (!result) {
    return (
      <div className="result-card result-card--empty">
        <div className="result-empty-icon">🏥</div>
        <p className="result-empty-msg">
          왼쪽 폼에 환자 정보를 입력하고
          <br />
          <strong>예측하기</strong> 버튼을 클릭하세요.
        </p>
      </div>
    );
  }

  // 성공 상태: 예측 결과 표시
  const { predicted_days, predicted_days_rounded, input_summary } = result;

  return (
    <div className="result-card result-card--success">
      <h2 className="result-title">📊 예측 결과</h2>

      {/* ── 핵심 지표: 예측 입원일수 ── */}
      <div className="result-metrics">
        <div className="metric-box metric-box--primary">
          <span className="metric-label">예상 입원일수</span>
          {/* toFixed(1): 소수 첫째 자리까지 표시 */}
          <span className="metric-value">{predicted_days.toFixed(1)}일</span>
        </div>
        <div className="metric-box metric-box--secondary">
          <span className="metric-label">반올림 입원일수</span>
          <span className="metric-value">{predicted_days_rounded}일</span>
        </div>
      </div>

      {/* ── 입력 요약 ── */}
      <div className="result-summary">
        <h3 className="summary-title">입력 요약</h3>
        <dl className="summary-list">
          <div className="summary-row">
            <dt>나이</dt>
            <dd>{input_summary.age}세</dd>
          </div>
          <div className="summary-row">
            <dt>성별</dt>
            <dd>{input_summary.gender === "M" ? "남성" : "여성"}</dd>
          </div>
          <div className="summary-row">
            <dt>병명</dt>
            <dd>
              {/* 타입 캐스팅: input_summary.diagnosis는 Diagnosis 타입임을 보장 */}
              {DIAGNOSIS_LABELS[input_summary.diagnosis]} ({input_summary.diagnosis})
            </dd>
          </div>
          <div className="summary-row">
            <dt>기저질환</dt>
            <dd>
              {COMORBIDITY_LABELS[input_summary.comorbidity]} ({input_summary.comorbidity})
            </dd>
          </div>
          <div className="summary-row">
            <dt>검사 이상</dt>
            <dd>{input_summary.lab_result_abnormal === "yes" ? "이상 있음" : "정상"}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
