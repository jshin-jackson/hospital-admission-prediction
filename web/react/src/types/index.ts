/**
 * 타입 정의 파일
 *
 * FastAPI 스키마(src/api/schemas.py)와 동일한 구조를 TypeScript 타입으로 정의합니다.
 * 타입을 정의하면 오타나 잘못된 데이터 전달을 컴파일 타임에 잡을 수 있습니다.
 */

// ─────────────────────────────────────────────
// 지원 값 목록 (schema.py와 동일하게 유지)
// ─────────────────────────────────────────────

/** 지원하는 병명 목록 */
export const DIAGNOSES = [
  "cold",
  "flu",
  "asthma",
  "pneumonia",
  "diabetes",
  "hypertension",
  "heart_disease",
] as const;

/** 병명 한글 레이블 매핑 */
export const DIAGNOSIS_LABELS: Record<Diagnosis, string> = {
  cold: "감기",
  flu: "독감",
  asthma: "천식",
  pneumonia: "폐렴",
  diabetes: "당뇨병",
  hypertension: "고혈압",
  heart_disease: "심장질환",
};

/** 기저질환 목록 */
export const COMORBIDITIES = [
  "none",
  "diabetes",
  "hypertension",
  "heart_disease",
  "obesity",
] as const;

/** 기저질환 한글 레이블 매핑 */
export const COMORBIDITY_LABELS: Record<Comorbidity, string> = {
  none: "없음",
  diabetes: "당뇨병",
  hypertension: "고혈압",
  heart_disease: "심장질환",
  obesity: "비만",
};

// ─────────────────────────────────────────────
// 유니온 타입 정의
// ─────────────────────────────────────────────

/** 병명 타입 */
export type Diagnosis = (typeof DIAGNOSES)[number];

/** 성별 타입 */
export type Gender = "M" | "F";

/** 기저질환 타입 */
export type Comorbidity = (typeof COMORBIDITIES)[number];

/** 검사 이상 여부 타입 */
export type LabResult = "yes" | "no";

// ─────────────────────────────────────────────
// API 요청/응답 타입 (FastAPI 스키마와 1:1 매핑)
// ─────────────────────────────────────────────

/** POST /predict 요청 본문 타입 */
export interface PredictRequest {
  age: number;
  gender: Gender;
  diagnosis: Diagnosis;
  comorbidity: Comorbidity;
  num_medications: number;
  prior_admissions: number;
  lab_result_abnormal: LabResult;
}

/** POST /predict 응답 본문 타입 */
export interface PredictResponse {
  predicted_days: number;         // 예측 입원일수 (소수점 포함)
  predicted_days_rounded: number; // 반올림된 예측 입원일수
  input_summary: {
    age: number;
    gender: Gender;
    diagnosis: Diagnosis;
    comorbidity: Comorbidity;
    lab_result_abnormal: LabResult;
  };
}

/** GET /health 응답 타입 */
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

/** GET /model/info 응답 타입 */
export interface ModelInfoResponse {
  model_type: string;
  model_path: string;
  features: string[];
}
