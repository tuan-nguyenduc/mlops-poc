"""Send steady prediction traffic to the model API (needed for the canary analysis).

    python scripts/load.py --rps 10
"""

from __future__ import annotations

import argparse
import random
import time

import requests


def application() -> dict:
    return {
        "income": round(random.lognormvariate(11, 0.5), 2),
        "loan_amount": round(random.uniform(5_000, 300_000), 2),
        "credit_score": round(min(850, max(300, random.gauss(680, 60)))),
        "employment_years": round(min(40, random.expovariate(1 / 6)), 1),
        "debt_to_income": round(random.betavariate(2, 5), 3),
        "late_payments": min(10, int(random.expovariate(1.2))),
    }


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
