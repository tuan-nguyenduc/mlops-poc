"""Training pipeline: ingest -> validate -> train -> evaluate -> register.

Predicts calories/protein/carbs/fat per 100g from a free-text food
description (TF-IDF features -> multi-output regression). Every run is
tracked in MLflow and registered as a new version of `nutrition-estimator`.
Registering does NOT deploy it. Deployment happens through promote.py + Git.

Examples:
    python pipeline.py                              # random forest, defaults
    python pipeline.py --algo ridge                 # a simpler linear baseline
    python pipeline.py --n-estimators 300 --max-depth 12  # try to beat the champion

CI note: pushes touching ml/** auto-trigger .github/workflows/train-promote-dev.yml
"""

from __future__ import annotations

import argparse
import os

import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from data import TARGETS, TEXT_FEATURE, generate, validate

MODEL_NAME = os.getenv("MODEL_NAME", "nutrition-estimator")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "nutrition-estimator")
HOLDOUT_SEED = 20_260_000  # fixed: the same 20% split every run, for a fair comparison
HOLDOUT_FRACTION = 0.2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["rf", "ridge"], default="rf")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--alpha", type=float, default=1.0, help="ridge regularisation")
    p.add_argument("--max-features", type=int, default=800, help="TF-IDF vocabulary size")
    p.add_argument("--rows", type=int, default=None, help="subsample the training pool (default: use all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drift", type=float, default=0.0, help="simulate population drift (see data.generate)")
    return p.parse_args()


def build_model(args: argparse.Namespace):
    preprocess = ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=args.max_features, stop_words="english"), TEXT_FEATURE),
    ])
    if args.algo == "ridge":
        reg = Ridge(alpha=args.alpha)
        params = {"alpha": args.alpha}
    else:
        reg = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.seed)
        )
        params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
    return Pipeline([("prep", preprocess), ("reg", reg)]), params


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    mlflow.set_experiment(EXPERIMENT)

    # 1. Ingest: the full real dataset, optionally drifted (demo only)
    df = generate(seed=args.seed, shift=args.drift)
    # 2. Validate (raises and stops the pipeline on bad data)
    report = validate(df)
    print(f"[validate] ok: {report}")

    # Every run is scored on the SAME fixed holdout split, so metrics are
    # comparable across runs and the promotion gate is fair.
    train_pool, holdout = train_test_split(df, test_size=HOLDOUT_FRACTION, random_state=HOLDOUT_SEED)
    if args.rows is not None:
        train_pool = train_pool.sample(n=args.rows, random_state=args.seed, replace=args.rows > len(train_pool))

    X_train, y_train = train_pool[[TEXT_FEATURE]], train_pool[TARGETS]
    X_test, y_test = holdout[[TEXT_FEATURE]], holdout[TARGETS]
    model, params = build_model(args)

    with mlflow.start_run() as run:
        mlflow.set_tags({"algo": args.algo, "pipeline": "ml/pipeline.py"})
        mlflow.log_params({**params, "algo": args.algo, "rows": len(X_train), "seed": args.seed,
                           "drift": args.drift, "max_features": args.max_features,
                           "holdout_seed": HOLDOUT_SEED, "holdout_rows": len(X_test)})
        mlflow.log_dict(report, "data/validation_report.json")
        # Training target means: a baseline for drift checks later.
        mlflow.log_dict(y_train.mean().round(3).to_dict(), "data/feature_baseline.json")

        # 3. Train
        model.fit(X_train, y_train)

        # 4. Evaluate: per-target R^2/MAE, plus an aggregate r2 for the promotion gate
        pred = np.asarray(model.predict(X_test))
        r2_per_target = {t: r2_score(y_test[t], pred[:, i]) for i, t in enumerate(TARGETS)}
        mae_per_target = {t: mean_absolute_error(y_test[t], pred[:, i]) for i, t in enumerate(TARGETS)}
        metrics = {"r2": float(np.mean(list(r2_per_target.values())))}
        metrics.update({f"r2_{t}": v for t, v in r2_per_target.items()})
        metrics.update({f"mae_{t}": v for t, v in mae_per_target.items()})
        mlflow.log_metrics(metrics)
        print(f"[evaluate] r2={metrics['r2']:.4f} | " + ", ".join(f"mae_{t}={mae_per_target[t]:.2f}" for t in TARGETS))

        # 5. Register
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(3),
            registered_model_name=MODEL_NAME,
        )

    versions = MlflowClient().search_model_versions(f"run_id='{run.info.run_id}'")
    version = versions[0].version if versions else "?"
    print(f"[register] {MODEL_NAME} version {version} (run {run.info.run_id})")
    print(f"Next: python promote.py --version {version}")


if __name__ == "__main__":
    main()
