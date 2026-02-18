"""Merge raw inputs into a daily model-ready table."""

from __future__ import annotations

import pandas as pd

from src.ops.navigation_demand import estimate_navigation_hm3


def merge_inputs(
    reservoir_df: pd.DataFrame,
    met_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge(reservoir_df, met_df, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Build navigation water demand from lock usage directly if present,
    # otherwise estimate from ship passage counts.
    if "water_for_locks" not in merged.columns:
        merged["water_for_locks"] = merged.apply(estimate_navigation_hm3, axis=1)
    else:
        null_mask = merged["water_for_locks"].isna()
        if null_mask.any():
            merged.loc[null_mask, "water_for_locks"] = merged[null_mask].apply(
                estimate_navigation_hm3, axis=1
            )

    return merged
