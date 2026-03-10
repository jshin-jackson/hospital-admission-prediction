/**
 * 환자 정보 입력 폼 컴포넌트
 *
 * 7개 피처를 입력받아 부모 컴포넌트(App.tsx)로 전달합니다.
 * 각 입력 요소에는 유효성 검사와 도움말 텍스트가 포함되어 있습니다.
 */

import type { ChangeEvent } from "react";
import type {
  Comorbidity,
  Diagnosis,
  Gender,
  LabResult,
  PredictRequest,
} from "../types";
import {
  COMORBIDITIES,
  COMORBIDITY_LABELS,
  DIAGNOSES,
  DIAGNOSIS_LABELS,
} from "../types";

// ─────────────────────────────────────────────
// Props 타입 정의
// ─────────────────────────────────────────────

interface PredictFormProps {
  /** 현재 폼 입력값 */
  values: PredictRequest;
  /** 입력값 변경 시 호출되는 콜백 */
  onChange: (updated: PredictRequest) => void;
  /** 예측 버튼 클릭 시 호출되는 콜백 */
  onSubmit: () => void;
  /** 예측 요청 중 여부 (버튼 비활성화에 사용) */
  isLoading: boolean;
}

// ─────────────────────────────────────────────
// 컴포넌트
// ─────────────────────────────────────────────

export function PredictForm({
  values,
  onChange,
  onSubmit,
  isLoading,
}: PredictFormProps) {
  /**
   * 수치형 필드 변경 핸들러
   * input의 value는 항상 string이므로 숫자로 변환합니다.
   */
  const handleNumber =
    (key: keyof PredictRequest) => (e: ChangeEvent<HTMLInputElement>) => {
      onChange({ ...values, [key]: Number(e.target.value) });
    };

  /**
   * 범주형(select/radio) 필드 변경 핸들러
   * 제네릭을 사용해 다양한 select 필드에 재사용합니다.
   */
  const handleSelect =
    <K extends keyof PredictRequest>(key: K) =>
    (e: ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
      onChange({ ...values, [key]: e.target.value as PredictRequest[K] });
    };

  return (
    <div className="form-card">
      <h2 className="section-title">👤 환자 정보 입력</h2>

      {/* ── 2열 그리드 레이아웃 ── */}
      <div className="form-grid">

        {/* 나이 */}
        <div className="form-group">
          <label className="form-label">
            나이
            <span className="form-hint">1 ~ 100세</span>
          </label>
          <input
            type="number"
            className="form-input"
            min={1}
            max={100}
            value={values.age}
            onChange={handleNumber("age")}
          />
        </div>

        {/* 성별 */}
        <div className="form-group">
          <label className="form-label">성별</label>
          <div className="radio-group">
            {(["M", "F"] as Gender[]).map((g) => (
              <label key={g} className="radio-label">
                <input
                  type="radio"
                  name="gender"
                  value={g}
                  checked={values.gender === g}
                  onChange={handleSelect("gender")}
                />
                {g === "M" ? "남성" : "여성"}
              </label>
            ))}
          </div>
        </div>

        {/* 병명 */}
        <div className="form-group">
          <label className="form-label">
            병명
            <span className="form-hint">주요 진단명</span>
          </label>
          <select
            className="form-select"
            value={values.diagnosis}
            onChange={handleSelect("diagnosis")}
          >
            {DIAGNOSES.map((d) => (
              <option key={d} value={d}>
                {DIAGNOSIS_LABELS[d as Diagnosis]} ({d})
              </option>
            ))}
          </select>
        </div>

        {/* 기저질환 */}
        <div className="form-group">
          <label className="form-label">
            기저질환
            <span className="form-hint">만성 기저질환</span>
          </label>
          <select
            className="form-select"
            value={values.comorbidity}
            onChange={handleSelect("comorbidity")}
          >
            {COMORBIDITIES.map((c) => (
              <option key={c} value={c}>
                {COMORBIDITY_LABELS[c as Comorbidity]} ({c})
              </option>
            ))}
          </select>
        </div>

        {/* 복용 약물 수 */}
        <div className="form-group">
          <label className="form-label">
            복용 약물 수
            <span className="form-hint">0 ~ 15개</span>
          </label>
          <div className="range-wrapper">
            <input
              type="range"
              className="form-range"
              min={0}
              max={15}
              value={values.num_medications}
              onChange={handleNumber("num_medications")}
            />
            <span className="range-value">{values.num_medications}개</span>
          </div>
        </div>

        {/* 과거 입원 횟수 */}
        <div className="form-group">
          <label className="form-label">
            과거 입원 횟수
            <span className="form-hint">0 ~ 10회</span>
          </label>
          <div className="range-wrapper">
            <input
              type="range"
              className="form-range"
              min={0}
              max={10}
              value={values.prior_admissions}
              onChange={handleNumber("prior_admissions")}
            />
            <span className="range-value">{values.prior_admissions}회</span>
          </div>
        </div>

        {/* 검사 이상 여부 */}
        <div className="form-group form-group--full">
          <label className="form-label">
            검사 이상 여부
            <span className="form-hint">혈액검사 등 임상검사 이상 여부</span>
          </label>
          <div className="radio-group">
            {(["no", "yes"] as LabResult[]).map((v) => (
              <label key={v} className="radio-label">
                <input
                  type="radio"
                  name="lab_result_abnormal"
                  value={v}
                  checked={values.lab_result_abnormal === v}
                  onChange={handleSelect("lab_result_abnormal")}
                />
                {v === "no" ? "정상 (no)" : "이상 (yes)"}
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* ── 예측 버튼 ── */}
      <button
        className={`predict-btn ${isLoading ? "predict-btn--loading" : ""}`}
        onClick={onSubmit}
        disabled={isLoading}
      >
        {isLoading ? "⏳ 예측 중..." : "🔍 입원일수 예측하기"}
      </button>
    </div>
  );
}
