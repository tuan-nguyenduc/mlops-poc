import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import TARGETS, TEXT_FEATURE, generate, validate  # noqa: E402


def test_generate_has_expected_columns():
    df = generate()
    assert set(df.columns) == {TEXT_FEATURE, *TARGETS}


def test_validate_passes_on_clean_data():
    report = validate(generate())
    assert report["rows"] == 7_793
    assert report["mean_calories"] > 0


def test_validate_rejects_out_of_range_values():
    df = generate()
    df.loc[0, "calories"] = 99_999
    with pytest.raises(ValueError, match="calories"):
        validate(df)


def test_validate_rejects_empty_description():
    df = generate()
    df.loc[0, "description"] = "  "
    with pytest.raises(ValueError, match="description"):
        validate(df)


def test_validate_rejects_nulls():
    df = generate()
    df.loc[0, "protein_g"] = None
    with pytest.raises(ValueError, match="null"):
        validate(df)
