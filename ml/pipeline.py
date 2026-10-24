"""Training pipeline: ingest -> validate -> train -> evaluate -> register.

Every run is tracked in MLflow (params, metrics, data report, model artifact)
and the model is registered as a new version of `loan-default`.
Registering does NOT deploy it. Deployment happens through promote.py + Git.

Examples:
    python pipeline.py                                   # gradient boosting, defaults
    python pipeline.py --algo logreg --C 0.5             # a weaker baseline
    python pipeline.py --n-estimators 300 --max-depth 4  # try to beat the champion

CI note: pushes touching ml/** auto-trigger .github/workflows/train-promote-dev.yml
"""

from __future__ import annotations

import argparse
import os

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import FEATURES, TARGET, generate, validate

MODEL_NAME = os.getenv("MODEL_NAME", "loan-default")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "loan-default")
HOLDOUT_SEED = 20_260_000  # fixed: never used for training
HOLDOUT_ROWS = 10_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["gbm", "logreg"], default="gbm")
    p.add_argument("--n-estimators", type=int, default=150)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--C", type=float, default=1.0, help="logreg regularisation")
    p.add_argument("--rows", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5, help="decision threshold")
    return p.parse_args()


def build_model(args: argparse.Namespace):
    if args.algo == "logreg":
        clf = LogisticRegression(C=args.C, max_iter=1_000)
        params = {"C": args.C}
    else:
        clf = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.seed,
        )
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        }
    return Pipeline([("scale", StandardScaler()), ("clf", clf)]), params


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    mlflow.set_experiment(EXPERIMENT)

    # 1. Ingest
    df = generate(n=args.rows, seed=args.seed)
    # 2. Validate (raises and stops the pipeline on bad data)
    report = validate(df)
    print(f"[validate] ok: {report}")

    X_train, y_train = df[FEATURES], df[TARGET]
    # Every run is scored on the SAME fixed holdout set, so metrics are comparable
    # across runs and the promotion gate is fair.
    holdout = generate(n=HOLDOUT_ROWS, seed=HOLDOUT_SEED)
    X_test, y_test = holdout[FEATURES], holdout[TARGET]
    model, params = build_model(args)

    with mlflow.start_run() as run:
        mlflow.set_tags({"algo": args.algo, "pipeline": "ml/pipeline.py"})
        mlflow.log_params({**params, "algo": args.algo, "rows": args.rows,
                           "seed": args.seed, "threshold": args.threshold,
                           "holdout_seed": HOLDOUT_SEED, "holdout_rows": HOLDOUT_ROWS})
        mlflow.log_dict(report, "data/validation_report.json")
        # Training feature means: a simple baseline for drift checks later.
        mlflow.log_dict(X_train.mean().round(4).to_dict(), "data/feature_baseline.json")

        # 3. Train
        model.fit(X_train, y_train)

        # 4. Evaluate
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= args.threshold).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y_test, proba),
            "f1": f1_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred),
        }
        mlflow.log_metrics(metrics)
        print("[evaluate] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

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
