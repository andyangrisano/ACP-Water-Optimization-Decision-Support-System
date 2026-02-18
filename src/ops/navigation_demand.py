"""Navigation-demand estimation from ship passages and lake level."""

from __future__ import annotations

import pandas as pd

from src.config import (
    DEFAULT_NEOPANAMAX_WATER_PER_TRANSIT_HM3,
    DEFAULT_PANAMAX_WATER_PER_TRANSIT_HM3,
)


def panamax_level_factor(gatun_level: float) -> float:
    """Approximate level-dependent Panamax water factor.

    At lower lake levels operators tend to use stronger conservation protocols.
    This factor is intentionally conservative and should be calibrated with ACP data.
    """
    if gatun_level < 80.0:
        return 0.88
    if gatun_level < 82.0:
        return 0.94
    if gatun_level < 84.0:
        return 1.00
    return 1.05


def estimate_navigation_hm3(row: pd.Series) -> float:
    if pd.notna(row.get("water_for_locks")):
        return float(row["water_for_locks"])

    panamax = float(row.get("panamax_transits", 0.0) or 0.0)
    neo = float(row.get("neopanamax_transits", 0.0) or 0.0)
    level = float(row.get("gatun_level", 82.0) or 82.0)

    panamax_hm3 = panamax * DEFAULT_PANAMAX_WATER_PER_TRANSIT_HM3 * panamax_level_factor(level)
    neo_hm3 = neo * DEFAULT_NEOPANAMAX_WATER_PER_TRANSIT_HM3
    return panamax_hm3 + neo_hm3


def estimate_navigation_series_hm3(df: pd.DataFrame) -> pd.Series:
    return df.apply(estimate_navigation_hm3, axis=1)
