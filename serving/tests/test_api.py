import os
import time

os.environ["MODEL_STUB"] = "1"

import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, unmatched_foods  # noqa: E402


class FakePipeline:
    """Mimics a real model's named_steps["prep"].transform() shape, so
    unmatched_foods can be tested without a trained model."""

    def __init__(self, zero_rows: set[int]):
        self.zero_rows = zero_rows
        self.named_steps = {"prep": self}

    def transform(self, X):
        rows = [0 if i in self.zero_rows else 1 for i in range(len(X))]
        return sp.csr_matrix([[r] for r in rows])


def test_unmatched_foods_flags_zero_signal_rows():
    model = FakePipeline(zero_rows={1})
    X = pd.DataFrame({"description": ["chicken breast", "prawn"]})
    assert unmatched_foods(model, X) == ["prawn"]


def test_unmatched_foods_empty_for_stub_model():
    class Stub:
        pass

    assert unmatched_foods(Stub(), pd.DataFrame({"description": ["anything"]})) == []


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
