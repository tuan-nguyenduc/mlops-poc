import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import FEATURES, TARGET, generate, validate  # noqa: E402


def test_generate_has_expected_columns():
    df = generate()
    assert set(df.columns) == set(FEATURES + [TARGET])


def test_validate_passes_on_clean_data():
    report = validate(generate())
    assert report["rows"] == 1_000
    assert 0.05 <= report["default_rate"] <= 0.60


def test_validate_rejects_out_of_range_values():
    df = generate()
    df.loc[0, "age"] = 9_999
    with pytest.raises(ValueError, match="age"):
        validate(df)


def test_validate_rejects_unknown_category():
    df = generate()
    df.loc[0, "housing"] = "mansion"
    with pytest.raises(ValueError, match="housing"):
        validate(df)


def test_validate_rejects_nulls():
    df = generate()
    df.loc[0, "credit_amount"] = None
    with pytest.raises(ValueError, match="null"):
        validate(df)
