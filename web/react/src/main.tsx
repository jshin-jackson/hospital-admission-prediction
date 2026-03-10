/**
 * React 앱 진입점 (Entry Point)
 *
 * index.html의 <div id="root"> 에 React 앱을 마운트합니다.
 * React 18의 createRoot API를 사용합니다.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css"; // 전역 스타일 불러오기

// id="root" 엘리먼트를 찾아 React 앱을 렌더링합니다.
// ! (non-null assertion): TypeScript에게 null이 아님을 보장합니다.
createRoot(document.getElementById("root")!).render(
  // StrictMode: 개발 중 잠재적 문제를 감지해 경고를 표시합니다.
  // 프로덕션 빌드에서는 아무 영향이 없습니다.
  <StrictMode>
    <App />
  </StrictMode>,
);
