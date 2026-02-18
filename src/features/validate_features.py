"""Validation checks to prevent feature-mismatch runtime errors."""

from __future__ import annotations

import pandas as pd

from src.config import REQUIRED_MET_COLUMNS, REQUIRED_RESERVOIR_COLUMNS, get_feature_columns


class FeatureValidationError(ValueError):
    """Raised when required model features are missing."""



def _check_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Please verify input schema and preprocessing."
        )



def validate_raw_inputs(
    reservoir_df: pd.DataFrame,
    met_df: pd.DataFrame,
) -> None:
    _check_columns(reservoir_df, REQUIRED_RESERVOIR_COLUMNS, "reservoir_daily.csv")
    _check_columns(met_df, REQUIRED_MET_COLUMNS, "met_daily.csv")



def validate_engineered_features(
    feature_df: pd.DataFrame,
    required_features: list[str] | None = None,
) -> None:
    required = required_features or get_feature_columns()
    missing = [col for col in required if col not in feature_df.columns]
    if missing:
        raise FeatureValidationError(
            f"Engineered feature set is incomplete. Missing: {missing}."
            f" Required features count: {len(required)}."
        )
