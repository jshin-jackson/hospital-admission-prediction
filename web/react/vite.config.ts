import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Vite 설정 파일
 *
 * - React 플러그인을 사용해 JSX/TSX 파일을 처리합니다.
 * - 개발 서버는 포트 3000에서 실행됩니다.
 * - FastAPI(8000) 로의 API 요청을 프록시해 CORS 문제를 방지합니다.
 *   예: /api/predict → http://localhost:8000/predict
 */

/**
 * 서버 시작 완료 시 접속 URL을 콘솔에 출력하는 커스텀 Vite 플러그인.
 *
 * Vite의 configureServer 훅을 사용합니다.
 * httpServer의 'listening' 이벤트가 발생하면 서버가 준비된 것이므로
 * 그 시점에 URL을 출력합니다.
 */
function logServerUrlPlugin(): Plugin {
  return {
    name: "log-server-url",

    configureServer(server) {
      // 'listening' 이벤트: HTTP 서버가 포트에 바인딩 완료된 시점
      server.httpServer?.once("listening", () => {
        const port = server.config.server.port ?? 3000;
        const host = "localhost";

        // 구분선과 함께 접속 정보를 보기 좋게 출력
        console.log("\n" + "─".repeat(50));
        console.log("  🏥  병원 입원일수 예측 Web UI 실행 완료");
        console.log("─".repeat(50));
        console.log(`  ➜  Local:    http://${host}:${port}/`);
        console.log(`  ➜  API 서버: http://localhost:8000/docs`);
        console.log(`  ➜  MLflow:   http://localhost:5000`);
        console.log("─".repeat(50) + "\n");
      });
    },
  };
}

export default defineConfig({
  // logServerUrlPlugin을 react() 다음에 추가
  plugins: [react(), logServerUrlPlugin()],

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
