/**
 * FastAPI 연동 API 모듈
 *
 * 모든 HTTP 요청 로직을 이 파일에 모아 관리합니다.
 * 컴포넌트에서 fetch를 직접 사용하지 않고 이 함수들을 호출합니다.
 * (관심사 분리: UI 로직과 네트워크 로직을 분리)
 */

import type {
  HealthResponse,
  ModelInfoResponse,
  PredictRequest,
  PredictResponse,
} from "../types";

/**
 * Vite 프록시 설정(/api → localhost:8000)을 통해
 * CORS 문제 없이 FastAPI에 요청할 수 있습니다.
 * vite.config.ts 의 proxy 설정 참고
 */
const BASE_URL = "/api";

// ─────────────────────────────────────────────
// 커스텀 에러 클래스
// ─────────────────────────────────────────────

/** API 요청 실패 시 던지는 에러 */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ─────────────────────────────────────────────
// 공통 fetch 래퍼
// ─────────────────────────────────────────────

/**
 * fetch를 래핑해 에러 처리를 일관되게 합니다.
 * 네트워크 오류와 HTTP 오류를 구분해서 처리합니다.
 */
async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    // 네트워크 오류 (서버가 실행되지 않은 경우)
    throw new ApiError(
      0,
      "FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.",
    );
  }

  if (!response.ok) {
    // HTTP 오류 (422 유효성 검사 실패, 503 모델 미로드 등)
    const body = await response.json().catch(() => ({}));
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail);
    throw new ApiError(response.status, detail || `HTTP ${response.status}`);
  }

  return response.json() as Promise<T>;
}

// ─────────────────────────────────────────────
// API 함수들
// ─────────────────────────────────────────────

/**
 * GET /health
 * 서버 헬스체크 및 모델 로드 여부를 확인합니다.
 */
export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}

/**
 * GET /model/info
 * 현재 로드된 모델의 종류, 경로, 피처 목록을 가져옵니다.
 */
export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  return apiFetch<ModelInfoResponse>("/model/info");
}

/**
 * POST /predict
 * 환자 정보를 서버에 보내 예상 입원일수를 예측합니다.
 *
 * @param data - 환자 정보 (PredictRequest 타입)
 * @returns 예측 결과 (PredictResponse 타입)
 */
export async function postPredict(
  data: PredictRequest,
): Promise<PredictResponse> {
  return apiFetch<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(data),
  });
}
