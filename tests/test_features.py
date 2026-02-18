from __future__ import annotations

import pandas as pd
import pytest

from src.config import get_feature_columns
from src.features.build_features import build_features
from src.features.validate_features import FeatureValidationError, validate_engineered_features



def _base_df(n_days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "gatun_level": 80 + (pd.Series(range(n_days)) * 0.01),
            "rain_mm": 5 + (pd.Series(range(n_days)) % 7),
        }
    )



def test_build_features_generates_expected_columns() -> None:
    feature_df = build_features(_base_df())
    expected = get_feature_columns()
    missing = [c for c in expected if c not in feature_df.columns]
    assert not missing



def test_validate_features_missing_rain_lag1_clear_error() -> None:
    feature_df = build_features(_base_df()).drop(columns=["rain_mm_lag1"])

    with pytest.raises(FeatureValidationError) as exc:
        validate_engineered_features(feature_df)

    msg = str(exc.value)
    assert "rain_mm_lag1" in msg
    assert "incomplete" in msg.lower()
