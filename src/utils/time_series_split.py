"""Utilities for time-series cross-validation."""

from __future__ import annotations

from sklearn.model_selection import TimeSeriesSplit



def make_time_series_split(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)
