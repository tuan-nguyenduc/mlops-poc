"""Real nutrition dataset: USDA FoodData Central, SR Legacy release (2018-04).

7,793 real foods with measured calories/protein/carbs/fat per 100g, built
from USDA's official bulk CSV export (food.csv + food_nutrient.csv joined on
the four Atwater macro nutrient IDs: 1008 Energy, 1003 Protein, 1004 Total
lipid (fat), 1005 Carbohydrate). Loaded from a committed CSV snapshot
(ml/data/nutrition.csv) so training doesn't depend on USDA's site being
reachable; refresh it by re-running the join against a fresh USDA export
(see https://fdc.nal.usda.gov/download-datasets for the latest zip).

This is a text-to-numbers regression problem, not tabular classification:
given a free-text food description, predict its macros per 100g. The
serving API accepts a list of (food description, grams) items and scales/
sums these per-100g predictions to get a whole meal's totals.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TEXT_FEATURE = "description"
TARGETS = ["calories", "protein_g", "carbs_g", "fat_g"]
CSV_PATH = Path(__file__).resolve().parent / "data" / "nutrition.csv"

# Per-100g sanity ranges, checked by validate(). The serving API enforces the
# same limits on its predictions.
RANGES = {
    "calories": (0, 950),
    "protein_g": (0, 100),
    "carbs_g": (0, 100),
    "fat_g": (0, 100),
}


def generate(n: int | None = None, seed: int = 42, shift: float = 0.0) -> pd.DataFrame:
    """Return a reproducible sample of the real dataset.

    `shift` > 0 inflates calories, standing in for "live" drift since there's
    no real production traffic in this POC (see check_drift.py) - e.g. a
    USDA data revision or a shift toward richer foods being logged.
    """
    df = pd.read_csv(CSV_PATH).sample(frac=1, random_state=seed).reset_index(drop=True)
    if n is not None and n != len(df):
        df = df.sample(n=n, random_state=seed, replace=n > len(df)).reset_index(drop=True)
    if shift:
        df = df.copy()
        df["calories"] = (df["calories"] * (1 + 0.5 * shift)).round(1)
    return df


def validate(df: pd.DataFrame) -> dict:
    """Fail fast if the data is not fit for training. Returns a small report."""
    problems = []

    missing = [c for c in [TEXT_FEATURE, *TARGETS] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    nulls = df[[TEXT_FEATURE, *TARGETS]].isna().sum()
    for col, count in nulls[nulls > 0].items():
        problems.append(f"{col}: {count} null values")

    empty_text = int((df[TEXT_FEATURE].astype(str).str.strip() == "").sum())
    if empty_text:
        problems.append(f"{TEXT_FEATURE}: {empty_text} empty descriptions")

    for col, (low, high) in RANGES.items():
        bad = int(((df[col] < low) | (df[col] > high)).sum())
        if bad:
            problems.append(f"{col}: {bad} values outside [{low}, {high}]")

    if len(df) < 100:
        problems.append(f"only {len(df)} rows; need at least 100")

    if problems:
        raise ValueError("Data validation failed:\n  - " + "\n  - ".join(problems))

    return {"rows": len(df), "mean_calories": round(float(df["calories"].mean()), 1)}
