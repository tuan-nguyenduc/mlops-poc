import os
import time

os.environ["MODEL_STUB"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def wait_ready(client: TestClient) -> None:
    for _ in range(50):
        if client.get("/health/ready").status_code == 200:
            return
        time.sleep(0.1)
    raise AssertionError("model never became ready")


def test_predict_returns_totals():
    with TestClient(app) as client:
        wait_ready(client)
        r = client.post(
            "/predict",
            json={"items": [{"food": "grilled chicken breast", "grams": 150}, {"food": "white rice", "grams": 200}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["items"]) == 2
        assert body["total"]["calories"] > 0
        assert body["items"][0]["food"] == "grilled chicken breast"


def test_invalid_input_is_rejected():
    with TestClient(app) as client:
        wait_ready(client)
        bad = {"items": [{"food": "", "grams": 150}]}
        assert client.post("/predict", json=bad).status_code == 422


def test_metrics_are_exposed():
    with TestClient(app) as client:
        wait_ready(client)
        client.post("/predict", json={"items": [{"food": "apple", "grams": 100}]})
        body = client.get("/metrics").text
        assert "model_predicted_calories" in body
        assert 'http_requests_total{method="POST",path="/predict",status="200"}' in body
