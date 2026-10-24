import os
import time

os.environ["MODEL_STUB"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

GOOD = {
    "duration": 24, "credit_amount": 3_000, "installment_commitment": 2,
    "residence_since": 2, "age": 35, "existing_credits": 1, "num_dependents": 1,
    "checking_status": "no checking", "credit_history": "existing paid",
    "purpose": "radio/tv", "savings_status": "<100", "employment": "1<=X<4",
    "other_parties": "none", "property_magnitude": "car",
    "other_payment_plans": "none", "housing": "own", "job": "skilled",
    "own_telephone": "yes", "foreign_worker": "no",
}


def wait_ready(client: TestClient) -> None:
    for _ in range(50):
        if client.get("/health/ready").status_code == 200:
            return
        time.sleep(0.1)
    raise AssertionError("model never became ready")


def test_predict_returns_probability():
    with TestClient(app) as client:
        wait_ready(client)
        r = client.post("/predict", json={"applications": [GOOD, GOOD]})
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) == 2
        assert 0 <= preds[0]["default_probability"] <= 1
        assert preds[0]["decision"] in {"default", "repay"}


def test_invalid_input_is_rejected():
    with TestClient(app) as client:
        wait_ready(client)
        bad = {**GOOD, "housing": "mansion"}
        assert client.post("/predict", json={"applications": [bad]}).status_code == 422


def test_metrics_are_exposed():
    with TestClient(app) as client:
        wait_ready(client)
        client.post("/predict", json={"applications": [GOOD]})
        body = client.get("/metrics").text
        assert "model_predictions_total" in body
        assert 'http_requests_total{method="POST",path="/predict",status="200"}' in body
