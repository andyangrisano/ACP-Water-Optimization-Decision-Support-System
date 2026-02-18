"""Feature engineering for multi-horizon reservoir forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MAX_LAG_DAYS



def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("date").reset_index(drop=True)

    for base_col in ("rain_mm", "gatun_level"):
        for lag in range(1, MAX_LAG_DAYS + 1):
            data[f"{base_col}_lag{lag}"] = data[base_col].shift(lag)

    data["rain_7d_sum"] = data["rain_mm"].rolling(window=7, min_periods=1).sum()
    data["rain_14d_sum"] = data["rain_mm"].rolling(window=14, min_periods=1).sum()
    data["rain_7d_mean"] = data["rain_mm"].rolling(window=7, min_periods=1).mean()
    data["rain_14d_mean"] = data["rain_mm"].rolling(window=14, min_periods=1).mean()

    day_of_year = data["date"].dt.dayofyear
    data["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    data["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
    data["month"] = data["date"].dt.month.astype(int)

    return data
