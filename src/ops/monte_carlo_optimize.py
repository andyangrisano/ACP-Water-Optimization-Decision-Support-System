"""Multi-objective Monte Carlo optimizer for discretionary water allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_NEOPANAMAX_WATER_PER_TRANSIT_HM3,
    DEFAULT_PANAMAX_WATER_PER_TRANSIT_HM3,
)
from src.ops.navigation_demand import estimate_navigation_series_hm3
from src.ops.water_budget import (
    DEFAULT_FLUSHING_TO_LOCKS_RATIO,
    DEFAULT_LOCKS_HM3_PER_DAY,
    ENV_FLOW_DEFAULT_NET_HEAD_M,
    ENV_FLOW_HM3_PER_DAY,
    GATUN_HM3_PER_FOOT,
    HISTORICAL_HUMAN_SHARE,
    HISTORICAL_LOCKS_SHARE,
    _hydropower_mwh,
)


@dataclass(frozen=True)
class OptimizationInputs:
    predicted_change_ft: float
    forecast_std_ft: float
    horizon_days: int
    lock_hm3_per_day: float
    municipal_hm3_per_day: float
    hm3_per_transit: float


def _estimate_lock_hm3_per_day(recent_df: pd.DataFrame | None) -> float:
    if recent_df is not None and not recent_df.empty and "water_for_locks" in recent_df.columns:
        val = float(recent_df["water_for_locks"].tail(14).mean())
        if val > 0:
            return val

    if recent_df is not None and not recent_df.empty and {"panamax_transits", "neopanamax_transits"}.issubset(
        recent_df.columns
    ):
        est = estimate_navigation_series_hm3(recent_df.tail(14))
        val = float(est.mean())
        if val > 0:
            return val

    return DEFAULT_LOCKS_HM3_PER_DAY


def _estimate_hm3_per_transit(recent_df: pd.DataFrame | None) -> float:
    if recent_df is None or recent_df.empty:
        return 0.5 * (DEFAULT_PANAMAX_WATER_PER_TRANSIT_HM3 + DEFAULT_NEOPANAMAX_WATER_PER_TRANSIT_HM3)

    if {"water_for_locks", "panamax_transits", "neopanamax_transits"}.issubset(recent_df.columns):
        tail = recent_df.tail(14).copy()
        transits = tail[["panamax_transits", "neopanamax_transits"]].sum(axis=1)
        valid = transits > 0
        if valid.any():
            ratio = (tail.loc[valid, "water_for_locks"] / transits.loc[valid]).mean()
            if pd.notna(ratio) and ratio > 0:
                return float(ratio)

    return 0.5 * (DEFAULT_PANAMAX_WATER_PER_TRANSIT_HM3 + DEFAULT_NEOPANAMAX_WATER_PER_TRANSIT_HM3)


def _estimate_municipal_hm3_per_day(recent_df: pd.DataFrame | None) -> float:
    if recent_df is not None and not recent_df.empty and "water_for_municipal" in recent_df.columns:
        val = float(recent_df["water_for_municipal"].tail(14).mean())
        if val > 0:
            return val
    return (HISTORICAL_HUMAN_SHARE / HISTORICAL_LOCKS_SHARE) * DEFAULT_LOCKS_HM3_PER_DAY


def _build_inputs(
    predicted_change_ft: float,
    forecast_std_ft: float,
    horizon_days: int,
    recent_df: pd.DataFrame | None,
) -> OptimizationInputs:
    return OptimizationInputs(
        predicted_change_ft=float(predicted_change_ft),
        forecast_std_ft=max(float(forecast_std_ft), 0.001),
        horizon_days=int(horizon_days),
        lock_hm3_per_day=_estimate_lock_hm3_per_day(recent_df),
        municipal_hm3_per_day=_estimate_municipal_hm3_per_day(recent_df),
        hm3_per_transit=_estimate_hm3_per_transit(recent_df),
    )


def _simulate_policy(
    sampled_change_hm3: np.ndarray,
    hydro_share: float,
    flushing_extra_share: float,
    ship_support_share: float,
    mandatory_hydro_mwh: float,
    hm3_per_transit: float,
    min_flushing_share: float,
) -> dict[str, np.ndarray]:
    discretionary = np.maximum(sampled_change_hm3, 0.0)

    hydro_extra = discretionary * hydro_share
    flushing_extra = discretionary * flushing_extra_share
    ship_support = discretionary * ship_support_share
    reserve = np.maximum(discretionary - hydro_extra - flushing_extra - ship_support, 0.0)

    energy = mandatory_hydro_mwh + np.array([_hydropower_mwh(v) for v in hydro_extra])
    deficit = np.maximum(-sampled_change_hm3, 0.0)
    flushing_shortfall = np.maximum(min_flushing_share * discretionary - flushing_extra, 0.0)
    additional_transits = ship_support / max(hm3_per_transit, 1e-6)

    return {
        "discretionary_hm3": discretionary,
        "energy_mwh": energy,
        "deficit_hm3": deficit,
        "reserve_hm3": reserve,
        "flushing_extra_hm3": flushing_extra,
        "flushing_shortfall_hm3": flushing_shortfall,
        "ship_support_hm3": ship_support,
        "additional_transits": additional_transits,
    }


def _objective_score(
    objective_mode: str,
    sim: dict[str, np.ndarray],
    energy_price_per_mwh: float,
    ship_value_per_transit: float,
    deficit_penalty_per_hm3: float,
    salinity_penalty_per_hm3: float,
    reserve_bonus_per_hm3: float,
    flushing_bonus_per_hm3: float,
) -> np.ndarray:
    energy_value = sim["energy_mwh"] * energy_price_per_mwh
    ship_value = sim["additional_transits"] * ship_value_per_transit
    penalties = sim["deficit_hm3"] * deficit_penalty_per_hm3 + sim["flushing_shortfall_hm3"] * salinity_penalty_per_hm3
    resilience = sim["reserve_hm3"] * reserve_bonus_per_hm3 + sim["flushing_extra_hm3"] * flushing_bonus_per_hm3

    if objective_mode == "hydropower":
        return 1.25 * energy_value + 0.15 * ship_value + 0.5 * resilience - penalties
    if objective_mode == "ships":
        return 0.35 * energy_value + 1.25 * ship_value + 0.5 * resilience - penalties
    # balanced
    return 0.9 * energy_value + 0.9 * ship_value + resilience - penalties


def _single_objective_optimization(
    objective_mode: str,
    sampled_change_ft: np.ndarray,
    sampled_change_hm3: np.ndarray,
    inputs: OptimizationInputs,
    mandatory_hydro_mwh: float,
    min_flushing_share: float,
    energy_price_per_mwh: float,
    ship_value_per_transit: float,
    deficit_penalty_per_hm3: float,
    salinity_penalty_per_hm3: float,
    reserve_bonus_per_hm3: float,
    flushing_bonus_per_hm3: float,
) -> dict[str, object]:
    hydro_grid = np.linspace(0.00, 0.65, 14)
    flushing_grid = np.linspace(0.20, 0.60, 9)
    ship_grid = np.linspace(0.00, 0.55, 12)

    leaderboard: list[dict[str, float]] = []
    best_policy: dict[str, float] | None = None
    best_sim: dict[str, np.ndarray] | None = None
    best_expected = -1e18

    for hydro_share in hydro_grid:
        for flushing_share in flushing_grid:
            for ship_share in ship_grid:
                if hydro_share + flushing_share + ship_share > 0.95:
                    continue

                sim = _simulate_policy(
                    sampled_change_hm3=sampled_change_hm3,
                    hydro_share=float(hydro_share),
                    flushing_extra_share=float(flushing_share),
                    ship_support_share=float(ship_share),
                    mandatory_hydro_mwh=mandatory_hydro_mwh,
                    hm3_per_transit=inputs.hm3_per_transit,
                    min_flushing_share=min_flushing_share,
                )

                score = _objective_score(
                    objective_mode=objective_mode,
                    sim=sim,
                    energy_price_per_mwh=energy_price_per_mwh,
                    ship_value_per_transit=ship_value_per_transit,
                    deficit_penalty_per_hm3=deficit_penalty_per_hm3,
                    salinity_penalty_per_hm3=salinity_penalty_per_hm3,
                    reserve_bonus_per_hm3=reserve_bonus_per_hm3,
                    flushing_bonus_per_hm3=flushing_bonus_per_hm3,
                )
                expected_score = float(score.mean())

                record = {
                    "objective": objective_mode,
                    "hydro_share_pct": float(hydro_share * 100.0),
                    "flushing_extra_share_pct": float(flushing_share * 100.0),
                    "ship_support_share_pct": float(ship_share * 100.0),
                    "reserve_share_pct": float((1.0 - hydro_share - flushing_share - ship_share) * 100.0),
                    "expected_score": expected_score,
                    "p10_score": float(np.quantile(score, 0.10)),
                    "deficit_probability": float((sim["deficit_hm3"] > 0).mean()),
                    "expected_energy_mwh": float(sim["energy_mwh"].mean()),
                    "expected_additional_transits": float(sim["additional_transits"].mean()),
                }
                leaderboard.append(record)

                if expected_score > best_expected:
                    best_expected = expected_score
                    best_policy = record
                    best_sim = sim

    if best_policy is None or best_sim is None:
        raise ValueError(f"Optimization failed for objective={objective_mode}")

    baseline_sim = _simulate_policy(
        sampled_change_hm3=sampled_change_hm3,
        hydro_share=0.20,
        flushing_extra_share=max(min_flushing_share, 0.35),
        ship_support_share=0.15,
        mandatory_hydro_mwh=mandatory_hydro_mwh,
        hm3_per_transit=inputs.hm3_per_transit,
        min_flushing_share=min_flushing_share,
    )
    baseline_score = _objective_score(
        objective_mode=objective_mode,
        sim=baseline_sim,
        energy_price_per_mwh=energy_price_per_mwh,
        ship_value_per_transit=ship_value_per_transit,
        deficit_penalty_per_hm3=deficit_penalty_per_hm3,
        salinity_penalty_per_hm3=salinity_penalty_per_hm3,
        reserve_bonus_per_hm3=reserve_bonus_per_hm3,
        flushing_bonus_per_hm3=flushing_bonus_per_hm3,
    )

    process_df = pd.DataFrame(
        {
            "sampled_change_ft": sampled_change_ft,
            "sampled_change_hm3": sampled_change_hm3,
            "best_score": _objective_score(
                objective_mode=objective_mode,
                sim=best_sim,
                energy_price_per_mwh=energy_price_per_mwh,
                ship_value_per_transit=ship_value_per_transit,
                deficit_penalty_per_hm3=deficit_penalty_per_hm3,
                salinity_penalty_per_hm3=salinity_penalty_per_hm3,
                reserve_bonus_per_hm3=reserve_bonus_per_hm3,
                flushing_bonus_per_hm3=flushing_bonus_per_hm3,
            ),
            "baseline_score": baseline_score,
            "best_energy_mwh": best_sim["energy_mwh"],
            "best_additional_transits": best_sim["additional_transits"],
        }
    )

    return {
        "best_policy": best_policy,
        "summary": {
            "objective": objective_mode,
            "expected_score_best": float(best_expected),
            "expected_score_baseline": float(baseline_score.mean()),
            "expected_score_gain": float(best_expected - baseline_score.mean()),
            "expected_energy_mwh_best": float(best_sim["energy_mwh"].mean()),
            "expected_additional_transits_best": float(best_sim["additional_transits"].mean()),
            "deficit_probability_best": float((best_sim["deficit_hm3"] > 0).mean()),
        },
        "leaderboard": pd.DataFrame(leaderboard).sort_values("expected_score", ascending=False),
        "simulation_samples": process_df,
    }


def optimize_with_monte_carlo(
    predicted_change_ft: float,
    forecast_std_ft: float,
    horizon_days: int,
    recent_df: pd.DataFrame | None = None,
    n_sims: int = 5000,
    energy_price_per_mwh: float = 90.0,
    ship_value_per_transit: float = 12000.0,
    deficit_penalty_per_hm3: float = 130.0,
    reserve_bonus_per_hm3: float = 10.0,
    flushing_bonus_per_hm3: float = 8.0,
    min_flushing_share: float = 0.30,
    salinity_penalty_per_hm3: float = 140.0,
    random_seed: int = 42,
) -> dict[str, object]:
    inputs = _build_inputs(
        predicted_change_ft=predicted_change_ft,
        forecast_std_ft=forecast_std_ft,
        horizon_days=horizon_days,
        recent_df=recent_df,
    )

    rng = np.random.default_rng(seed=random_seed)
    sampled_change_ft = rng.normal(
        loc=inputs.predicted_change_ft,
        scale=inputs.forecast_std_ft,
        size=n_sims,
    )
    sampled_change_hm3 = sampled_change_ft * GATUN_HM3_PER_FOOT

    env_flow_total = ENV_FLOW_HM3_PER_DAY * inputs.horizon_days
    mandatory_hydro_mwh = _hydropower_mwh(env_flow_total, head_m=ENV_FLOW_DEFAULT_NET_HEAD_M)

    results: dict[str, dict[str, object]] = {}
    for mode in ["hydropower", "ships", "balanced"]:
        results[mode] = _single_objective_optimization(
            objective_mode=mode,
            sampled_change_ft=sampled_change_ft,
            sampled_change_hm3=sampled_change_hm3,
            inputs=inputs,
            mandatory_hydro_mwh=mandatory_hydro_mwh,
            min_flushing_share=min_flushing_share,
            energy_price_per_mwh=energy_price_per_mwh,
            ship_value_per_transit=ship_value_per_transit,
            deficit_penalty_per_hm3=deficit_penalty_per_hm3,
            salinity_penalty_per_hm3=salinity_penalty_per_hm3,
            reserve_bonus_per_hm3=reserve_bonus_per_hm3,
            flushing_bonus_per_hm3=flushing_bonus_per_hm3,
        )

    # Overall best impact: choose the best "balanced" result and provide cross-objective comparison.
    overall = results["balanced"]

    lock_total = inputs.lock_hm3_per_day * inputs.horizon_days
    municipal_total = inputs.municipal_hm3_per_day * inputs.horizon_days
    flushing_min_total = lock_total * DEFAULT_FLUSHING_TO_LOCKS_RATIO

    objective_comparison = pd.DataFrame(
        [
            {
                "objective": mode,
                "expected_score_best": float(results[mode]["summary"]["expected_score_best"]),
                "score_gain_vs_baseline": float(results[mode]["summary"]["expected_score_gain"]),
                "expected_energy_mwh_best": float(results[mode]["summary"]["expected_energy_mwh_best"]),
                "expected_additional_transits_best": float(
                    results[mode]["summary"]["expected_additional_transits_best"]
                ),
                "deficit_probability_best": float(results[mode]["summary"]["deficit_probability_best"]),
            }
            for mode in ["hydropower", "ships", "balanced"]
        ]
    )

    return {
        "hydropower": results["hydropower"],
        "ships": results["ships"],
        "balanced": results["balanced"],
        "overall_best": overall,
        "objective_comparison": objective_comparison,
        "global_summary": {
            "required_locks_hm3": lock_total,
            "required_municipal_hm3": municipal_total,
            "minimum_flushing_hm3": flushing_min_total,
            "minimum_environmental_flow_hm3": env_flow_total,
            "mandatory_hydropower_mwh": mandatory_hydro_mwh,
            "minimum_discretionary_flushing_share": min_flushing_share,
            "hm3_per_transit_assumption": inputs.hm3_per_transit,
        },
    }
