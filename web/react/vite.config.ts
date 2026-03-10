import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Vite 설정 파일
 *
 * - React 플러그인을 사용해 JSX/TSX 파일을 처리합니다.
 * - 개발 서버는 포트 3000에서 실행됩니다.
 * - FastAPI(8000) 로의 API 요청을 프록시해 CORS 문제를 방지합니다.
 *   예: /api/predict → http://localhost:8000/predict
 */
export default defineConfig({
  plugins: [react()],

  server: {
    port: 3000, // 개발 서버 포트

    proxy: {
      // /api 로 시작하는 요청을 FastAPI 서버로 전달
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""), // /api 접두사 제거
      },
    },
  },
});
