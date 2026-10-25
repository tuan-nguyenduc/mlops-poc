"""Model serving API for the nutrition-estimator model.

- Loads an exact model version from the MLflow registry (MODEL_VERSION comes
  from Git through the Helm values, so Git decides what runs).
- Loads in a background thread: the pod stays alive but reports NOT ready until
  the model is loaded, so Kubernetes never sends traffic to a pod without a model.
- Predicts calories/protein/carbs/fat per 100g from a free-text food
  description, then scales by the grams the user entered and sums across
  items in the meal.
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
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("model-api")

MODEL_NAME = os.getenv("MODEL_NAME", "nutrition-estimator")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
CHAOS_ERROR_RATE = float(os.getenv("CHAOS_ERROR_RATE", "0"))
CHAOS_CALORIE_BIAS = float(os.getenv("CHAOS_CALORIE_BIAS", "0"))
USE_STUB = os.getenv("MODEL_STUB") == "1"  # tests only

TARGETS = ["calories", "protein_g", "carbs_g", "fat_g"]

REQUESTS = Counter("http_requests_total", "HTTP requests", ["path", "method", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency", ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTED_CALORIES = Histogram(
    "model_predicted_calories", "Predicted calories per meal item (per 100g, pre-scaling)",
    buckets=(50, 100, 150, 200, 300, 400, 500, 700, 900),
)
MODEL_INFO = Gauge("model_info", "Model currently loaded", ["name", "version"])
MODEL_LOADED = Gauge("model_loaded", "1 when the model is loaded and serving")

state: dict = {"model": None}


class StubModel:
    """Tiny stand-in model so tests run without an MLflow server."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        lengths = X["description"].str.len().to_numpy()
        calories = 150 + (lengths % 50) * 5
        return np.column_stack([calories, np.full(n, 10.0), np.full(n, 20.0), np.full(n, 8.0)])


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


app = FastAPI(title="Nutrition estimator API", version="0.1.0", lifespan=lifespan)
# Local POC only: the frontend calls this API cross-origin from its own
# container, so CORS has to be open. Scope this to the real frontend origin
# in production instead of allowing everything.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class MealItem(BaseModel):
    food: str = Field(..., min_length=1, max_length=200, examples=["grilled chicken breast"])
    grams: float = Field(..., gt=0, le=5_000, examples=[150])


class PredictRequest(BaseModel):
    items: list[MealItem] = Field(..., min_length=1, max_length=50)


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
        "chaos_error_rate": CHAOS_ERROR_RATE,
    }


@app.get("/schema")
def schema():
    """MealItem's JSON schema, so a frontend can build its per-food-row form from
    the same definition /predict validates against - one source of truth, not two."""
    return MealItem.model_json_schema()


def unmatched_foods(model, X: pd.DataFrame) -> list[str]:
    """Foods whose TF-IDF vector has zero matching vocabulary words - the model
    has no signal for these and would otherwise return an arbitrary nearest
    match dressed up as a confident number (e.g. "prawn" vs USDA's "shrimp")."""
    if not hasattr(model, "named_steps"):
        return []  # StubModel (tests): no vocabulary concept
    tfidf = model.named_steps["prep"].transform(X)
    zero_signal = np.asarray(tfidf.sum(axis=1)).ravel() == 0
    return X.loc[zero_signal, "description"].tolist()


@app.post("/predict")
def predict(req: PredictRequest):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if CHAOS_ERROR_RATE and random.random() < CHAOS_ERROR_RATE:
        raise HTTPException(status_code=500, detail="chaos: injected failure")

    X = pd.DataFrame({"description": [item.food for item in req.items]})
    unmatched = unmatched_foods(model, X)
    if unmatched:
        raise HTTPException(status_code=422, detail=f"Food(s) not recognized, try different wording: {unmatched}")

    per_100g = np.asarray(model.predict(X), dtype=float)
    per_100g[:, 0] = (per_100g[:, 0] + CHAOS_CALORIE_BIAS).clip(min=0)  # calories column
    per_100g = per_100g.clip(min=0)

    items_out, totals = [], {t: 0.0 for t in TARGETS}
    for item, row in zip(req.items, per_100g, strict=True):
        scale = item.grams / 100
        values = {t: round(float(v) * scale, 2) for t, v in zip(TARGETS, row, strict=True)}
        PREDICTED_CALORIES.observe(float(row[0]))
        for t in TARGETS:
            totals[t] += values[t]
        items_out.append({"food": item.food, "grams": item.grams, **values})

    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "items": items_out,
        "total": {t: round(v, 2) for t, v in totals.items()},
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
