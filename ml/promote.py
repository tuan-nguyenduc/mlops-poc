"""Champion/challenger promotion gate, per environment.

Each environment has its own MLflow alias (so dev iterating fast can't stomp
on what staging/prod consider their champion) and, for staging/prod, a
minimum soak time: the candidate must already have been promoted to the
previous environment at least that long ago. Soak time is tracked with an
MLflow tag (promoted_<env>_at) set on the version at promotion time - no
extra infra needed.

Promoting:
  1. moves the environment's alias in the MLflow registry,
  2. tags the version with promoted_<env>_at, and
  3. writes the version into that environment's GitOps values file.
Committing and pushing that file is what deploys the model.

Examples:
    python promote.py                                     # prod, latest version, gated
    python promote.py --env dev                            # dev, latest version, gated
    python promote.py --env staging --version 6 --force    # deliberate staging bump
    python promote.py --env prod --version 6 --force --skip-soak  # emergency override
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "loan-default")
REPO_ROOT = Path(__file__).resolve().parent.parent

# requires: the environment that must already hold this version before it can
# be promoted here. min_soak_hours: how long it must have held it, at minimum.
ENVIRONMENTS = {
    "dev": {
        "values": REPO_ROOT / "gitops/model-api/overlays/dev/values.yaml",
        "alias": "dev-champion",
        "requires": None,
        "min_soak_hours": 0,
    },
    "staging": {
        "values": REPO_ROOT / "gitops/model-api/overlays/staging/values.yaml",
        "alias": "staging-champion",
        "requires": "dev",
        "min_soak_hours": 1,
    },
    "prod": {
        "values": REPO_ROOT / "gitops/model-api/values.yaml",
        "alias": "champion",
        "requires": "staging",
        "min_soak_hours": 24,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=ENVIRONMENTS, default="prod")
    p.add_argument("--version", help="candidate version (default: latest)")
    p.add_argument("--metric", default="roc_auc")
    p.add_argument("--min-improvement", type=float, default=0.0)
    p.add_argument("--values", type=Path, help="override the environment's default values file")
    p.add_argument("--min-soak-hours", type=float, help="override the environment's default soak requirement")
    p.add_argument("--force", action="store_true", help="skip the metric gate")
    p.add_argument("--skip-soak", action="store_true", help="skip the soak-time gate (emergency only)")
    return p.parse_args()


def metric_of(client: MlflowClient, version, metric: str) -> float:
    value = client.get_run(version.run_id).data.metrics.get(metric)
    if value is None:
        sys.exit(f"Version {version.version} has no metric '{metric}'")
    return value


def check_soak(client: MlflowClient, candidate, env: str, min_hours: float) -> None:
    prev_env = ENVIRONMENTS[env]["requires"]
    if prev_env is None or min_hours <= 0:
        return
    tag = f"promoted_{prev_env}_at"
    mv = client.get_model_version(MODEL_NAME, candidate.version)
    promoted_at = mv.tags.get(tag)
    if not promoted_at:
        sys.exit(f"BLOCKED: v{candidate.version} has never been promoted to {prev_env}; cannot promote to {env}.")
    elapsed_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(promoted_at)).total_seconds() / 3600
    if elapsed_hours < min_hours:
        sys.exit(
            f"BLOCKED: v{candidate.version} has only soaked in {prev_env} for {elapsed_hours:.1f}h "
            f"(needs {min_hours}h before promoting to {env}). Use --skip-soak to override."
        )
    print(f"[soak] v{candidate.version} has soaked in {prev_env} for {elapsed_hours:.1f}h (>= {min_hours}h required)")


def set_values_version(path: Path, version: str) -> None:
    text = path.read_text()
    new, count = re.subn(r'^(modelVersion:\s*).*$', rf'\g<1>"{version}"', text, flags=re.M)
    if count != 1:
        sys.exit(f"Could not find a top-level 'modelVersion:' line in {path}")
    path.write_text(new)


def main() -> None:
    args = parse_args()
    cfg = ENVIRONMENTS[args.env]
    alias = cfg["alias"]
    values_path = args.values or cfg["values"]
    min_soak_hours = args.min_soak_hours if args.min_soak_hours is not None else cfg["min_soak_hours"]

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

    if not args.skip_soak:
        check_soak(client, candidate, args.env, min_soak_hours)

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        champion = None

    if champion is not None:
        champ_score = metric_of(client, champion, args.metric)
        print(f"{alias} v{champion.version}: {args.metric}={champ_score:.4f}")
        if champion.version == candidate.version:
            print(f"Candidate is already '{alias}'. Nothing to do.")
            set_values_version(values_path, candidate.version)
            return
        if not args.force and cand_score < champ_score + args.min_improvement:
            sys.exit(
                f"REJECTED: v{candidate.version} does not beat v{champion.version} "
                f"by at least {args.min_improvement}. '{alias}' unchanged."
            )

    client.set_registered_model_alias(MODEL_NAME, alias, candidate.version)
    client.set_model_version_tag(MODEL_NAME, candidate.version, f"promoted_{args.env}_at", datetime.now(timezone.utc).isoformat())
    set_values_version(values_path, candidate.version)
    print(f"PROMOTED: v{candidate.version} is now '{alias}' ({args.env}). Updated {values_path}.")
    print(f"Deploy it with:  make release MSG='promote model v{candidate.version} to {args.env}'")


if __name__ == "__main__":
    main()
