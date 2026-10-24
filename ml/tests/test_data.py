import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import FEATURES, TARGET, generate, validate  # noqa: E402


def test_generate_has_expected_columns():
    df = generate(n=2_000)
    assert list(df.columns) == FEATURES + [TARGET]


def test_validate_passes_on_clean_data():
    report = validate(generate(n=2_000))
    assert report["rows"] == 2_000
    assert 0.02 <= report["default_rate"] <= 0.60


def test_validate_rejects_out_of_range_values():
    df = generate(n=2_000)
    df.loc[0, "credit_score"] = 9_999
    with pytest.raises(ValueError, match="credit_score"):
        validate(df)


def test_validate_rejects_nulls():
    df = generate(n=2_000)
    df.loc[0, "income"] = None
    with pytest.raises(ValueError, match="null"):
        validate(df)
