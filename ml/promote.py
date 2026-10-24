"""Champion/challenger promotion gate.

Compares a candidate model version with the current `champion` on a metric.
If the candidate is better, it:
  1. moves the `champion` alias in the MLflow registry, and
  2. writes the version into the GitOps values file (modelVersion).
Committing and pushing that file is what deploys the model: ArgoCD syncs it and
Argo Rollouts runs a canary release with automated analysis.

Examples:
    python promote.py                    # promote the latest version if it wins
    python promote.py --version 3
    python promote.py --version 2 --force   # roll back to an older model on purpose
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "loan-default")
ALIAS = "champion"
DEFAULT_VALUES = Path(__file__).resolve().parent.parent / "gitops/model-api/values.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", help="candidate version (default: latest)")
    p.add_argument("--metric", default="roc_auc")
    p.add_argument("--min-improvement", type=float, default=0.0)
    p.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    p.add_argument("--force", action="store_true", help="skip the metric gate")
    return p.parse_args()


def metric_of(client: MlflowClient, version, metric: str) -> float:
    value = client.get_run(version.run_id).data.metrics.get(metric)
    if value is None:
        sys.exit(f"Version {version.version} has no metric '{metric}'")
    return value


def set_values_version(path: Path, version: str) -> None:
    text = path.read_text()
    new, count = re.subn(r'^(modelVersion:\s*).*$', rf'\g<1>"{version}"', text, flags=re.M)
    if count != 1:
        sys.exit(f"Could not find a top-level 'modelVersion:' line in {path}")
    path.write_text(new)


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        sys.exit(f"No versions of '{MODEL_NAME}' registered yet. Run pipeline.py first.")
    if args.version:
        matches = [v for v in versions if v.version == str(args.version)]
        if not matches:
            sys.exit(f"Version {args.version} not found")
        candidate = matches[0]
    else:
        candidate = max(versions, key=lambda v: int(v.version))

    cand_score = metric_of(client, candidate, args.metric)
    print(f"Candidate v{candidate.version}: {args.metric}={cand_score:.4f}")

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException:
        champion = None

    if champion is not None:
        champ_score = metric_of(client, champion, args.metric)
        print(f"Champion  v{champion.version}: {args.metric}={champ_score:.4f}")
        if champion.version == candidate.version:
            print("Candidate is already the champion. Nothing to do.")
            set_values_version(args.values, candidate.version)
            return
        if not args.force and cand_score < champ_score + args.min_improvement:
            sys.exit(
                f"REJECTED: v{candidate.version} does not beat v{champion.version} "
                f"by at least {args.min_improvement}. Champion unchanged."
            )

    client.set_registered_model_alias(MODEL_NAME, ALIAS, candidate.version)
    set_values_version(args.values, candidate.version)
    print(f"PROMOTED: v{candidate.version} is now '{ALIAS}'. Updated {args.values}.")
    print("Deploy it with:  make release MSG='promote model v%s'" % candidate.version)


if __name__ == "__main__":
    main()
