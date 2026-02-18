"""Heuristic water-allocation guidance for Gatun operations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from src.ops.navigation_demand import estimate_navigation_series_hm3

# Historical context used as soft priors (ACP Sustainability Report 2011,
# 2000-2010 watershed balance): locks ~49%, hydro ~30%, human ~7%, excess ~14%.
HISTORICAL_LOCKS_SHARE = 0.49
HISTORICAL_HYDRO_SHARE = 0.30
HISTORICAL_HUMAN_SHARE = 0.07
HISTORICAL_EXCESS_SHARE = 0.14

# FY-2024 lock water use: 2,015 hm3 / 366 days ~= 5.5 hm3/day.
DEFAULT_LOCKS_HM3_PER_DAY = 5.5

# FY-2024 salinity mitigation outflows: 72 / 2015 = 3.57% of lock water use.
DEFAULT_FLUSHING_TO_LOCKS_RATIO = 72.0 / 2015.0

# Environmental flow insight (Rio Indio study): Q_env = 3.4 m3/s.
# Kept explicit as a non-negotiable demand-stack term.
ENV_FLOW_M3_PER_S = 3.4
ENV_FLOW_HM3_PER_DAY = ENV_FLOW_M3_PER_S * 86400.0 / 1_000_000.0
ENV_FLOW_DEFAULT_NET_HEAD_M = 71.91

# Approximate Gatun conversion using surface area ~425 km2:
# 1 ft level change ~= 129.54 hm3 storage change.
GATUN_HM3_PER_FOOT = 129.54

# Hydropower estimate assumptions (site-adjustable):
# P = rho * g * Q * H * eta.
WATER_DENSITY_KG_PER_M3 = 1000.0
GRAVITY_M_PER_S2 = 9.81
DEFAULT_HEAD_M = 24.0
DEFAULT_TURBINE_EFFICIENCY = 0.88


@dataclass(frozen=True)
class ScenarioAllocation:
    hydro_discretionary_share: float
    flushing_discretionary_share: float
    reserve_discretionary_share: float


SCENARIO_ALLOCATION = {
    # More risk-averse shares to align with security-first constraints.
    "Conservative": ScenarioAllocation(
        hydro_discretionary_share=0.10,
        flushing_discretionary_share=0.40,
        reserve_discretionary_share=0.50,
    ),
    "Baseline": ScenarioAllocation(
        hydro_discretionary_share=0.20,
        flushing_discretionary_share=0.35,
        reserve_discretionary_share=0.45,
    ),
    "Aggressive": ScenarioAllocation(
        hydro_discretionary_share=0.35,
        flushing_discretionary_share=0.30,
        reserve_discretionary_share=0.35,
    ),
}


def _estimate_municipal_hm3_per_day(recent_df: pd.DataFrame | None) -> float:
    if recent_df is None or recent_df.empty:
        return (HISTORICAL_HUMAN_SHARE / HISTORICAL_LOCKS_SHARE) * DEFAULT_LOCKS_HM3_PER_DAY

    if "water_for_municipal" in recent_df.columns:
        raw = float(recent_df["water_for_municipal"].tail(14).mean())
        if raw > 0:
            return raw

    return (HISTORICAL_HUMAN_SHARE / HISTORICAL_LOCKS_SHARE) * DEFAULT_LOCKS_HM3_PER_DAY


def _estimate_lock_hm3_per_day(recent_df: pd.DataFrame | None) -> float:
    if recent_df is None or recent_df.empty:
        return DEFAULT_LOCKS_HM3_PER_DAY

    if "water_for_locks" in recent_df.columns:
        raw = float(recent_df["water_for_locks"].tail(14).mean())
        if raw > 0:
            return raw

    if {"panamax_transits", "neopanamax_transits"}.issubset(recent_df.columns):
        est = estimate_navigation_series_hm3(recent_df.tail(14))
        raw = float(est.mean())
        if raw > 0:
            return raw

    return DEFAULT_LOCKS_HM3_PER_DAY


def _hydropower_mwh(volume_hm3: float, head_m: float = DEFAULT_HEAD_M) -> float:
    volume_m3 = max(volume_hm3, 0.0) * 1_000_000.0
    energy_j = (
        WATER_DENSITY_KG_PER_M3
        * GRAVITY_M_PER_S2
        * head_m
        * DEFAULT_TURBINE_EFFICIENCY
        * volume_m3
    )
    return energy_j / 3_600_000_000.0


def compute_scenarios(
    predicted_change: float,
    horizon_days: int,
    recent_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, float | str]]:
    lock_hm3_day = _estimate_lock_hm3_per_day(recent_df)
    municipal_hm3_day = _estimate_municipal_hm3_per_day(recent_df)
    flushing_hm3_day = lock_hm3_day * DEFAULT_FLUSHING_TO_LOCKS_RATIO

    locks_need = lock_hm3_day * horizon_days
    municipal_need = municipal_hm3_day * horizon_days
    flushing_min = flushing_hm3_day * horizon_days
    env_flow_min = ENV_FLOW_HM3_PER_DAY * horizon_days

    # Non-negotiable demand stack from study constraints.
    required_non_power = locks_need + municipal_need + flushing_min + env_flow_min

    storage_change_hm3 = predicted_change * GATUN_HM3_PER_FOOT
    discretionary_hm3 = max(storage_change_hm3, 0.0)
    net_balance_hm3 = storage_change_hm3

    # Environmental flow is modeled as must-run release with energy recovery.
    mandatory_hydro_mwh = _hydropower_mwh(env_flow_min, head_m=ENV_FLOW_DEFAULT_NET_HEAD_M)

    results: dict[str, dict[str, float | str]] = {}
    for name, allocation in SCENARIO_ALLOCATION.items():
        hydro_extra_hm3 = discretionary_hm3 * allocation.hydro_discretionary_share
        flushing_extra_hm3 = discretionary_hm3 * allocation.flushing_discretionary_share
        reserve_hm3 = discretionary_hm3 * allocation.reserve_discretionary_share

        total_flushing_hm3 = flushing_min + env_flow_min + flushing_extra_hm3
        total_hydropower_water_hm3 = env_flow_min + hydro_extra_hm3
        total_hydro_mwh = mandatory_hydro_mwh + _hydropower_mwh(hydro_extra_hm3)

        if net_balance_hm3 < 0 and name == "Aggressive":
            recommendation = "Deficit forecast: aggressive releases are not advised; prioritize demand stack and storage."
        elif net_balance_hm3 < 0 and name == "Conservative":
            recommendation = "Deficit forecast: meet M&I/navigation/env-flow first and preserve storage."
        elif net_balance_hm3 >= 0 and name == "Aggressive":
            recommendation = "Surplus forecast: additional hydropower recovery is possible after mandatory demands."
        else:
            recommendation = "Operate near normal releases while tracking updated forecast and salinity risk."

        results[name] = {
            "horizon_days": float(horizon_days),
            "net_balance_hm3": net_balance_hm3,
            "required_non_power_hm3": required_non_power,
            "required_navigation_hm3": locks_need,
            "required_municipal_hm3": municipal_need,
            "required_env_flow_hm3": env_flow_min,
            "hydropower_water_hm3": total_hydropower_water_hm3,
            "hydropower_extra_water_hm3": hydro_extra_hm3,
            "flushing_water_hm3": total_flushing_hm3,
            "reserve_water_hm3": reserve_hm3,
            "hydropower_share_pct_of_discretionary": allocation.hydro_discretionary_share * 100.0,
            "flushing_share_pct_of_discretionary": allocation.flushing_discretionary_share * 100.0,
            "estimated_hydropower_mwh": total_hydro_mwh,
            "mandatory_hydropower_mwh_from_env_flow": mandatory_hydro_mwh,
            "recommendation": recommendation,
        }
    return results


def scenarios_as_frame(predicted_change: float, recent_df: pd.DataFrame | None = None) -> pd.DataFrame:
    return scenarios_as_frame_for_horizon(
        predicted_change=predicted_change,
        horizon_days=7,
        recent_df=recent_df,
    )


def scenarios_as_frame_for_horizon(
    predicted_change: float,
    horizon_days: int,
    recent_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    scenarios = compute_scenarios(
        predicted_change=predicted_change,
        horizon_days=horizon_days,
        recent_df=recent_df,
    )
    rows = []
    for name, vals in scenarios.items():
        rows.append({"scenario": name, **vals})
    return pd.DataFrame(rows)
