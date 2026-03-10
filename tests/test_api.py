"""
FastAPI 엔드포인트 통합 테스트.
"""

from __future__ import annotations

import pytest


VALID_PAYLOAD = {
    "age": 55,
    "gender": "M",
    "diagnosis": "pneumonia",
    "comorbidity": "hypertension",
    "num_medications": 4,
    "prior_admissions": 2,
    "lab_result_abnormal": "yes",
}


class TestHealthEndpoint:
    def test_returns_200(self, api_client) -> None:
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_model_loaded_true(self, api_client) -> None:
        resp = api_client.get("/health")
        assert resp.json()["model_loaded"] is True

    def test_status_ok(self, api_client) -> None:
        assert api_client.get("/health").json()["status"] == "ok"


class TestModelInfoEndpoint:
    def test_returns_200(self, api_client) -> None:
        resp = api_client.get("/model/info")
        assert resp.status_code == 200

    def test_contains_model_type(self, api_client) -> None:
        resp = api_client.get("/model/info")
        assert "model_type" in resp.json()

    def test_features_list_not_empty(self, api_client) -> None:
        resp = api_client.get("/model/info")
        assert len(resp.json()["features"]) > 0


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, api_client) -> None:
        resp = api_client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_predicted_days(self, api_client) -> None:
        resp = api_client.post("/predict", json=VALID_PAYLOAD)
        data = resp.json()
        assert "predicted_days" in data
        assert isinstance(data["predicted_days"], (int, float))

    def test_predicted_days_positive(self, api_client) -> None:
        resp = api_client.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["predicted_days"] > 0

    def test_predicted_days_rounded_is_int(self, api_client) -> None:
        resp = api_client.post("/predict", json=VALID_PAYLOAD)
        assert isinstance(resp.json()["predicted_days_rounded"], int)

    def test_invalid_age_returns_422(self, api_client) -> None:
        payload = {**VALID_PAYLOAD, "age": -5}
        resp = api_client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_diagnosis_returns_422(self, api_client) -> None:
        payload = {**VALID_PAYLOAD, "diagnosis": "unknown_disease"}
        resp = api_client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_gender_returns_422(self, api_client) -> None:
        payload = {**VALID_PAYLOAD, "gender": "X"}
        resp = api_client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_minimal_payload(self, api_client) -> None:
        minimal = {
            "age": 30,
            "gender": "F",
            "diagnosis": "cold",
        }
        resp = api_client.post("/predict", json=minimal)
        assert resp.status_code == 200

    def test_all_diagnoses_work(self, api_client) -> None:
        from src.data.schema import DIAGNOSES

        for diagnosis in DIAGNOSES:
            payload = {**VALID_PAYLOAD, "diagnosis": diagnosis}
            resp = api_client.post("/predict", json=payload)
            assert resp.status_code == 200, f"{diagnosis} 실패: {resp.text}"
