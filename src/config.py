"""Central configuration for the Canal Water DSS."""

from __future__ import annotations

from pathlib import Path

HORIZONS = [1, 7, 14, 30]
TARGETS = ["gatun_level"]
MAX_LAG_DAYS = 14

REQUIRED_RESERVOIR_COLUMNS = ["date", "gatun_level"]
OPTIONAL_RESERVOIR_COLUMNS = [
    "gatun_release",
    "spill",
    "water_for_locks",
    "water_for_municipal",
    "panamax_transits",
    "neopanamax_transits",
]

REQUIRED_MET_COLUMNS = ["date", "rain_mm"]
OPTIONAL_MET_COLUMNS = ["temp_c", "et0_mm"]

RAW_FILENAMES = {
    "reservoir": "reservoir_daily.csv",
    "met": "met_daily.csv",
}

# Ship-passage demand defaults. Tune these with ACP operational data.
DEFAULT_PANAMAX_WATER_PER_TRANSIT_HM3 = 0.19
DEFAULT_NEOPANAMAX_WATER_PER_TRANSIT_HM3 = 0.23


def get_lag_feature_columns() -> list[str]:
    cols: list[str] = []
    for base in ("rain_mm", "gatun_level"):
        cols.extend([f"{base}_lag{i}" for i in range(1, MAX_LAG_DAYS + 1)])
    return cols


def get_rolling_feature_columns() -> list[str]:
    return ["rain_7d_sum", "rain_14d_sum", "rain_7d_mean", "rain_14d_mean"]


def get_seasonal_feature_columns() -> list[str]:
    return ["sin_day_of_year", "cos_day_of_year", "month"]


def get_feature_columns() -> list[str]:
    return (
        get_lag_feature_columns()
        + get_rolling_feature_columns()
        + get_seasonal_feature_columns()
    )


def default_latest_forecast_path() -> Path:
    return Path("data") / "processed" / "latest_forecast.csv"
