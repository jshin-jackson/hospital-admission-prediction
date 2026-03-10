/**
 * 최상위 App 컴포넌트
 *
 * 전체 레이아웃과 상태(state)를 관리합니다.
 * - 폼 입력값 상태
 * - 예측 결과 상태
 * - 로딩/에러 상태
 * - 서버 상태
 *
 * React의 상태 관리:
 *   useState: 컴포넌트 내 상태 변수를 선언합니다.
 *   useEffect: 컴포넌트가 화면에 표시될 때 자동으로 실행됩니다.
 */

import { useEffect, useState } from "react";
import { ApiError, fetchHealth, fetchModelInfo, postPredict } from "./api/predict";
import { PredictForm } from "./components/PredictForm";
import { PredictResult } from "./components/PredictResult";
import type { ModelInfoResponse, PredictRequest, PredictResponse } from "./types";

// ─────────────────────────────────────────────
// 폼 초기값 정의
// ─────────────────────────────────────────────

/** 처음 화면에 표시될 기본 입력값 */
const DEFAULT_FORM: PredictRequest = {
  age: 45,
  gender: "M",
  diagnosis: "pneumonia",
  comorbidity: "none",
  num_medications: 3,
  prior_admissions: 1,
  lab_result_abnormal: "no",
};

// ─────────────────────────────────────────────
// App 컴포넌트
// ─────────────────────────────────────────────

export default function App() {

  // ── 상태 변수 선언 ──────────────────────────

  /** 폼 입력값 상태: 사용자가 입력을 바꿀 때마다 업데이트 */
  const [formValues, setFormValues] = useState<PredictRequest>(DEFAULT_FORM);

  /** 예측 결과 상태: API 호출 성공 시 저장 */
  const [result, setResult] = useState<PredictResponse | null>(null);

  /** 예측 API 호출 중 여부 */
  const [isLoading, setIsLoading] = useState(false);

  /** 에러 메시지 상태 */
  const [error, setError] = useState<string | null>(null);

  /** 서버 연결 상태 */
  const [serverStatus, setServerStatus] = useState<"ok" | "error" | "checking">("checking");

  /** 현재 로드된 모델 정보 */
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);


  // ── 컴포넌트 마운트 시 서버 상태 확인 ─────────

  /**
   * useEffect: 컴포넌트가 처음 화면에 표시될 때 한 번 실행됩니다.
   * 빈 배열([])을 두 번째 인자로 전달하면 최초 1회만 실행됩니다.
   */
  useEffect(() => {
    const checkServer = async () => {
      try {
        const health = await fetchHealth();
        if (health.status === "ok" && health.model_loaded) {
          setServerStatus("ok");
          // 서버가 정상이면 모델 정보도 가져옵니다
          const info = await fetchModelInfo();
          setModelInfo(info);
        } else {
          setServerStatus("error");
        }
      } catch {
        setServerStatus("error");
      }
    };

    checkServer();
  }, []); // 빈 배열: 처음 렌더링 시에만 실행


  // ── 예측 실행 핸들러 ───────────────────────

  /**
   * 예측 버튼 클릭 시 호출됩니다.
   * 1. 로딩 상태 시작
   * 2. API 호출
   * 3. 결과 또는 에러 상태 업데이트
   * 4. 로딩 상태 종료
   */
  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);     // 이전 에러 초기화
    setResult(null);    // 이전 결과 초기화

    try {
      const response = await postPredict(formValues);
      setResult(response);
    } catch (err) {
      // ApiError: 서버에서 반환한 에러
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("예상치 못한 오류가 발생했습니다.");
      }
    } finally {
      // 성공/실패 상관없이 로딩 종료
      setIsLoading(false);
    }
  };


  // ── 렌더링 ────────────────────────────────

  return (
    <div className="app">

      {/* ── 헤더 ── */}
      <header className="app-header">
        <div className="header-inner">
          <h1 className="app-title">🏥 병원 입원일수 예측 시스템</h1>
          <p className="app-subtitle">
            환자 정보를 입력하면 AI가 예상 입원일수를 예측합니다.
          </p>

          {/* 서버 상태 뱃지 */}
          <div className="status-badges">
            <span className={`badge badge--${serverStatus}`}>
              {serverStatus === "checking" && "⏳ 서버 확인 중..."}
              {serverStatus === "ok" && "✅ 서버 정상"}
              {serverStatus === "error" && "❌ 서버 연결 실패"}
            </span>

            {/* 모델 정보가 로드된 경우 모델 종류 표시 */}
            {modelInfo && (
              <span className="badge badge--info">
                🤖 {modelInfo.model_type}
              </span>
            )}
          </div>
        </div>
      </header>

      {/* ── 메인 콘텐츠: 2열 레이아웃 ── */}
      <main className="app-main">

        {/* 왼쪽: 입력 폼 */}
        <section className="app-section">
          <PredictForm
            values={formValues}
            onChange={setFormValues}
            onSubmit={handlePredict}
            isLoading={isLoading}
          />
        </section>

        {/* 오른쪽: 예측 결과 */}
        <section className="app-section">
          <PredictResult result={result} error={error} />
        </section>

      </main>

      {/* ── 푸터 ── */}
      <footer className="app-footer">
        <p>
          API 서버:{" "}
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">
            localhost:8000/docs
          </a>
          {" | "}
          <a href="http://localhost:5000" target="_blank" rel="noreferrer">
            MLflow UI
          </a>
        </p>
      </footer>
    </div>
  );
}
