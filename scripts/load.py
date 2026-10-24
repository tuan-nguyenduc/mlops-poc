"""Send steady prediction traffic to the model API (needed for the canary analysis).

    python scripts/load.py --rps 10
"""

from __future__ import annotations

import argparse
import random
import time

import requests


# Must match ml/data.py's CATEGORY_VALUES/NUMERIC_RANGES (duplicated here since
# this script runs standalone against the live API, not the training code).
CATEGORY_VALUES = {
    "checking_status": ["0<=X<200", "<0", ">=200", "no checking"],
    "credit_history": [
        "all paid", "critical/other existing credit", "delayed previously",
        "existing paid", "no credits/all paid",
    ],
    "purpose": [
        "business", "domestic appliance", "education", "furniture/equipment",
        "new car", "other", "radio/tv", "repairs", "retraining", "used car",
    ],
    "savings_status": ["100<=X<500", "500<=X<1000", "<100", ">=1000", "no known savings"],
    "employment": ["1<=X<4", "4<=X<7", "<1", ">=7", "unemployed"],
    "other_parties": ["co applicant", "guarantor", "none"],
    "property_magnitude": ["car", "life insurance", "no known property", "real estate"],
    "other_payment_plans": ["bank", "none", "stores"],
    "housing": ["for free", "own", "rent"],
    "job": ["high qualif/self emp/mgmt", "skilled", "unemp/unskilled non res", "unskilled resident"],
    "own_telephone": ["none", "yes"],
    "foreign_worker": ["no", "yes"],
}


def application() -> dict:
    app = {
        "duration": random.randint(4, 72),
        "credit_amount": round(random.uniform(250, 18_000), 2),
        "installment_commitment": random.randint(1, 4),
        "residence_since": random.randint(1, 4),
        "age": random.randint(19, 75),
        "existing_credits": random.randint(1, 4),
        "num_dependents": random.randint(1, 2),
    }
    for col, choices in CATEGORY_VALUES.items():
        app[col] = random.choice(choices)
    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8800/predict")
    p.add_argument("--rps", type=float, default=10)
    args = p.parse_args()

    ok = err = 0
    versions: dict[str, int] = {}
    last = time.time()
    print(f"Sending ~{args.rps} req/s to {args.url} (Ctrl+C to stop)")
    while True:
        try:
            r = requests.post(args.url, json={"applications": [application()]}, timeout=5)
            if r.ok:
                ok += 1
                v = r.json().get("version", "?")
                versions[v] = versions.get(v, 0) + 1
            else:
                err += 1
        except requests.RequestException:
            err += 1
        if time.time() - last >= 10:
            mix = ", ".join(f"v{k}: {n}" for k, n in sorted(versions.items()))
            print(f"ok={ok} errors={err} | served by model {mix}")
            ok = err = 0
            versions.clear()
            last = time.time()
        time.sleep(1 / args.rps)


if __name__ == "__main__":
    main()
