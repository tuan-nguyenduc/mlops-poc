"""Synthetic loan-default data plus a data-validation step.

The data imitates a digital lending use case: each row is a loan application,
and the target says whether the borrower defaulted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "late_payments",
]
TARGET = "defaulted"

# Allowed ranges, checked by validate(). The serving API enforces the same limits.
RANGES = {
    "income": (0, 5_000_000),
    "loan_amount": (1_000, 1_000_000),
    "credit_score": (300, 850),
    "employment_years": (0, 50),
    "debt_to_income": (0, 1),
    "late_payments": (0, 50),
}


def generate(n: int = 20_000, seed: int = 42, shift: float = 0.0) -> pd.DataFrame:
    """Generate applications. `shift` > 0 makes credit scores worse, to simulate drift."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=11.0, sigma=0.5, size=n)
    loan_amount = rng.uniform(5_000, 300_000, size=n)
    credit_score = (rng.normal(680, 60, size=n) - shift * 50).clip(300, 850)
    employment_years = rng.exponential(6, size=n).clip(0, 40)
    debt_to_income = rng.beta(2, 5, size=n)
    late_payments = rng.poisson(0.8, size=n)

    logit = (
        -1.6
        - 0.012 * (credit_score - 680)
        + 2.5 * (debt_to_income - 0.28)
        + 0.45 * late_payments
        + 0.30 * (loan_amount / income - 2.5)
        - 0.05 * (employment_years - 6)
    )
    prob = 1 / (1 + np.exp(-logit))
    defaulted = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "credit_score": credit_score.round(0),
            "employment_years": employment_years.round(1),
            "debt_to_income": debt_to_income.round(3),
            "late_payments": late_payments,
            TARGET: defaulted,
        }
    )


def validate(df: pd.DataFrame) -> dict:
    """Fail fast if the data is not fit for training. Returns a small report."""
    problems = []

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    nulls = df[FEATURES + [TARGET]].isna().sum()
    for col, count in nulls[nulls > 0].items():
        problems.append(f"{col}: {count} null values")

    for col, (low, high) in RANGES.items():
        bad = int(((df[col] < low) | (df[col] > high)).sum())
        if bad:
            problems.append(f"{col}: {bad} values outside [{low}, {high}]")

    if not set(df[TARGET].unique()) <= {0, 1}:
        problems.append(f"{TARGET} must be 0 or 1")

    default_rate = float(df[TARGET].mean())
    if not 0.02 <= default_rate <= 0.60:
        problems.append(f"default rate {default_rate:.2%} is outside the expected 2%-60%")

    if len(df) < 1_000:
        problems.append(f"only {len(df)} rows; need at least 1000")

    if problems:
        raise ValueError("Data validation failed:\n  - " + "\n  - ".join(problems))

    return {"rows": len(df), "default_rate": round(default_rate, 4)}
