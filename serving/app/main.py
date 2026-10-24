"""Model serving API for the loan-default model.

- Loads an exact model version from the MLflow registry (MODEL_VERSION comes
  from Git through the Helm values, so Git decides what runs).
- Loads in a background thread: the pod stays alive but reports NOT ready until
  the model is loaded, so Kubernetes never sends traffic to a pod without a model.
- Exposes Prometheus metrics used by Grafana and by the Argo Rollouts canary analysis.
- CHAOS_ERROR_RATE injects 5xx errors, to demo an automatic canary rollback.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("model-api")

MODEL_NAME = os.getenv("MODEL_NAME", "loan-default")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
CHAOS_ERROR_RATE = float(os.getenv("CHAOS_ERROR_RATE", "0"))
CHAOS_PREDICTION_BIAS = float(os.getenv("CHAOS_PREDICTION_BIAS", "0"))
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
USE_STUB = os.getenv("MODEL_STUB") == "1"  # tests only

FEATURES = ["income", "loan_amount", "credit_score", "employment_years", "debt_to_income", "late_payments"]

REQUESTS = Counter("http_requests_total", "HTTP requests", ["path", "method", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency", ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter("model_predictions_total", "Predictions by outcome", ["outcome"])
SCORES = Histogram(
    "model_prediction_score", "Predicted default probability",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
)
MODEL_INFO = Gauge("model_info", "Model currently loaded", ["name", "version"])
MODEL_LOADED = Gauge("model_loaded", "1 when the model is loaded and serving")

state: dict = {"model": None}


class StubModel:
    """Tiny stand-in model so tests run without an MLflow server."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = ((850 - X["credit_score"].to_numpy()) / 550).clip(0, 1)
        return np.column_stack([1 - p, p])


def load_model_forever() -> None:
    uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    while state["model"] is None:
        try:
            if USE_STUB:
                model = StubModel()
            else:
                import mlflow.sklearn  # imported lazily: heavy

                log.info("Loading %s", uri)
                model = mlflow.sklearn.load_model(uri)
            state["model"] = model
            MODEL_INFO.labels(MODEL_NAME, MODEL_VERSION).set(1)
            MODEL_LOADED.set(1)
            log.info("Model %s v%s loaded", MODEL_NAME, MODEL_VERSION)
        except Exception as exc:  # registry not ready yet, version missing, network...
            log.warning("Model load failed (%s); retrying in 10s", exc)
            time.sleep(10)


@asynccontextmanager
async def lifespan(_: FastAPI):
    MODEL_LOADED.set(0)
    threading.Thread(target=load_model_forever, daemon=True).start()
    yield


app = FastAPI(title="Loan default model API", version="0.1.0", lifespan=lifespan)


class LoanApplication(BaseModel):
    income: float = Field(..., ge=0, le=5_000_000, examples=[65_000])
    loan_amount: float = Field(..., ge=1_000, le=1_000_000, examples=[120_000])
    credit_score: float = Field(..., ge=300, le=850, examples=[690])
    employment_years: float = Field(..., ge=0, le=50, examples=[4])
    debt_to_income: float = Field(..., ge=0, le=1, examples=[0.3])
    late_payments: int = Field(..., ge=0, le=50, examples=[1])


class PredictRequest(BaseModel):
    applications: list[LoanApplication] = Field(..., min_length=1, max_length=1_000)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        REQUESTS.labels(path, request.method, str(status)).inc()
        LATENCY.labels(path).observe(time.perf_counter() - start)


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready(response: Response):
    if state["model"] is None:
        response.status_code = 503
        return {"status": "loading", "model": MODEL_NAME, "version": MODEL_VERSION}
    return {"status": "ready", "model": MODEL_NAME, "version": MODEL_VERSION}


@app.get("/info")
def info():
    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "loaded": state["model"] is not None,
        "threshold": THRESHOLD,
        "chaos_error_rate": CHAOS_ERROR_RATE,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if CHAOS_ERROR_RATE and random.random() < CHAOS_ERROR_RATE:
        raise HTTPException(status_code=500, detail="chaos: injected failure")

    X = pd.DataFrame([a.model_dump() for a in req.applications])[FEATURES]
    proba = (model.predict_proba(X)[:, 1] + CHAOS_PREDICTION_BIAS).clip(0, 1)

    results = []
    for p in proba:
        outcome = "default" if p >= THRESHOLD else "repay"
        PREDICTIONS.labels(outcome).inc()
        SCORES.observe(float(p))
        results.append({"default_probability": round(float(p), 4), "decision": outcome})
    return {"model": MODEL_NAME, "version": MODEL_VERSION, "predictions": results}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
