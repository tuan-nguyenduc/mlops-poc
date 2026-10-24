"""Compare freshly observed feature means against a champion's logged baseline.

Real teams run this against live production request logs. Here "live" data
is simulated with the same generator pipeline.py trains on, optionally
shifted with --drift to demonstrate detection (pipeline.py's --drift flag
uses the same mechanism to produce a genuinely worse model when needed).

Example:
    python check_drift.py                    # compare prod's live traffic pattern
    python check_drift.py --env staging       # compare staging's champion
    python check_drift.py --drift 1.0         # simulate a drifted population
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from data import FEATURES, generate

MODEL_NAME = os.getenv("MODEL_NAME", "loan-default")
ALIASES = {"dev": "dev-champion", "staging": "staging-champion", "prod": "champion"}
DEFAULT_THRESHOLD = 0.10  # 10% relative mean shift on any feature counts as drift


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=ALIASES, default="prod")
    p.add_argument("--drift", type=float, default=0.0, help="simulate drift in the live sample (demo only)")
    p.add_argument("--rows", type=int, default=5_000)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    client = MlflowClient()

    mv = client.get_model_version_by_alias(MODEL_NAME, ALIASES[args.env])
    with tempfile.TemporaryDirectory() as tmp:
        path = client.download_artifacts(mv.run_id, "data/feature_baseline.json", tmp)
        baseline = json.loads(Path(path).read_text())

    live = generate(n=args.rows, seed=int(time.time()), shift=args.drift)
    live_means = live[FEATURES].mean()

    report, drifted = [], False
    for feat in FEATURES:
        base, now = baseline[feat], live_means[feat]
        rel = abs(now - base) / (abs(base) + 1e-9)
        flag = rel > args.threshold
        drifted = drifted or flag
        report.append(f"{'DRIFT' if flag else 'ok   '} {feat}: baseline={base:.3f} live={now:.3f} ({rel:.1%} shift)")

    print(f"Comparing live sample against {args.env} champion v{mv.version}:")
    print("\n".join(report))
    if drifted:
        sys.exit(f"DRIFT DETECTED in {args.env} - retraining is warranted")
    print("No significant drift detected.")


if __name__ == "__main__":
    main()
