"""Real credit-risk dataset: UCI "German Credit" (OpenML id 31, "credit-g").

1000 real historical loan applications with a known outcome - genuinely real
labeled data, not synthetic. Loaded from a committed CSV snapshot
(ml/data/credit-g.csv) so training never depends on OpenML being reachable;
refresh it with `python data.py --refresh`.

`personal_status` is dropped on purpose: it encodes sex and marital status
combined, a protected attribute that shouldn't drive a credit decision.

Being real and small (1000 rows) cuts both ways versus the old synthetic
generator: metrics are noisier run-to-run, and there's no "just generate
more rows" knob - `--rows` now subsamples the fixed training pool instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET = "defaulted"
CSV_PATH = Path(__file__).resolve().parent / "data" / "credit-g.csv"

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]
CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed values per categorical column - used for validation here, and by the
# serving API / load generator so synthetic requests stay realistic.
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
NUMERIC_RANGES = {
    "duration": (1, 100),
    "credit_amount": (0, 30_000),
    "installment_commitment": (1, 4),
    "residence_since": (1, 4),
    "age": (18, 100),
    "existing_credits": (1, 4),
    "num_dependents": (1, 2),
}


def _fetch_and_cache() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    raw = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto").frame
    df = raw.copy()
    df[TARGET] = (df["class"] == "bad").astype(int)
    df = df.drop(columns=["class", "personal_status"])
    CSV_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


def _load_raw() -> pd.DataFrame:
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return _fetch_and_cache()


def generate(n: int | None = None, seed: int = 42, shift: float = 0.0) -> pd.DataFrame:
    """Return a reproducible sample of the real dataset.

    `shift` > 0 inflates credit_amount, standing in for "live" drift since
    there's no real production traffic in this POC (see check_drift.py).
    `n` subsamples (with replacement if n exceeds the dataset size).
    """
    df = _load_raw().sample(frac=1, random_state=seed).reset_index(drop=True)
    if n is not None and n != len(df):
        df = df.sample(n=n, random_state=seed, replace=n > len(df)).reset_index(drop=True)
    if shift:
        df = df.copy()
        df["credit_amount"] = (df["credit_amount"] * (1 + 0.5 * shift)).round(0)
    return df


def validate(df: pd.DataFrame) -> dict:
    """Fail fast if the data is not fit for training. Returns a small report."""
    problems = []

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    nulls = df[FEATURES + [TARGET]].isna().sum()
    for col, count in nulls[nulls > 0].items():
        problems.append(f"{col}: {count} null values")

    for col, (low, high) in NUMERIC_RANGES.items():
        bad = int(((df[col] < low) | (df[col] > high)).sum())
        if bad:
            problems.append(f"{col}: {bad} values outside [{low}, {high}]")

    for col, allowed in CATEGORY_VALUES.items():
        bad = int((~df[col].isin(allowed)).sum())
        if bad:
            problems.append(f"{col}: {bad} values outside {allowed}")

    if not set(df[TARGET].unique()) <= {0, 1}:
        problems.append(f"{TARGET} must be 0 or 1")

    default_rate = float(df[TARGET].mean())
    if not 0.05 <= default_rate <= 0.60:
        problems.append(f"default rate {default_rate:.2%} is outside the expected 5%-60%")

    if len(df) < 100:
        problems.append(f"only {len(df)} rows; need at least 100")

    if problems:
        raise ValueError("Data validation failed:\n  - " + "\n  - ".join(problems))

    return {"rows": len(df), "default_rate": round(default_rate, 4)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Refresh the cached dataset snapshot from OpenML")
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    if args.refresh:
        df = _fetch_and_cache()
        print(f"Refreshed {CSV_PATH} ({len(df)} rows)")
