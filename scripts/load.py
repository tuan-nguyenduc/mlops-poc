"""Send steady prediction traffic to the model API (needed for the canary analysis).

    python scripts/load.py --rps 10
"""

from __future__ import annotations

import argparse
import random
import time

import requests

# A small sample of plausible meal items - doesn't need to match the training
# vocabulary exactly, just needs to exercise the live API with realistic text.
FOODS = [
    "grilled chicken breast", "white rice", "steamed broccoli", "scrambled eggs",
    "whole wheat toast", "greek yogurt", "banana", "salmon fillet", "sweet potato",
    "black beans", "avocado", "oatmeal", "almonds", "mixed green salad",
    "ground beef", "pasta", "cheddar cheese", "orange juice", "peanut butter",
    "grilled salmon with lemon",
]


def meal_item() -> dict:
    return {"food": random.choice(FOODS), "grams": round(random.uniform(50, 400), 1)}


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
            items = [meal_item() for _ in range(random.randint(1, 3))]
            r = requests.post(args.url, json={"items": items}, timeout=5)
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
