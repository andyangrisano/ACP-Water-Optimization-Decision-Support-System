"""Streamlit dashboard for Gatun Lake water decision support."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure imports from `src` resolve when launched via:
# `streamlit run app/streamlit_app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import HORIZONS, default_latest_forecast_path
from src.features.build_features import build_features
from src.features.validate_features import (
    FeatureValidationError,
    validate_engineered_features,
    validate_raw_inputs,
)
from src.io.load_data import load_all_raw_data
from src.io.merge_data import merge_inputs
from src.io.update_from_bulk_export import DEFAULT_BULK_URL, update_raw_files
from src.models.predict import generate_forecasts
from src.models.registry import model_path
from src.ops.monte_carlo_optimize import optimize_with_monte_carlo

st.set_page_config(page_title="Gatun Lake Water DSS", layout="wide")
st.title("Gatun Lake Water Decision Support System")
st.caption(
    "Forecasts Gatun lake level and provides practical guidance for water operations."
)


@st.cache_data
def load_feature_table(data_dir: str = "data/raw") -> pd.DataFrame:
    reservoir_df, met_df = load_all_raw_data(data_dir)
    validate_raw_inputs(reservoir_df, met_df)
    merged = merge_inputs(reservoir_df, met_df)
    feature_df = build_features(merged)
    validate_engineered_features(feature_df)
    return feature_df


@st.cache_data
def load_forecasts(path: str | Path = default_latest_forecast_path()) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def source_freshness_table(data_dir: str = "data/raw") -> pd.DataFrame:
    rows = []
    today = pd.Timestamp.today().normalize()

    for fname in ["reservoir_daily.csv", "met_daily.csv"]:
        path = Path(data_dir) / fname
        if not path.exists():
            rows.append(
                {
                    "source_file": fname,
                    "latest_date": None,
                    "days_old": None,
                    "status": "missing",
                }
            )
            continue

        df = pd.read_csv(path)
        if "date" not in df.columns:
            rows.append(
                {
                    "source_file": fname,
                    "latest_date": None,
                    "days_old": None,
                    "status": "invalid (no date column)",
                }
            )
            continue

        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            rows.append(
                {
                    "source_file": fname,
                    "latest_date": None,
                    "days_old": None,
                    "status": "invalid dates",
                }
            )
            continue

        latest = dates.max().normalize()
        age = int((today - latest).days)
        rows.append(
            {
                "source_file": fname,
                "latest_date": latest.date().isoformat(),
                "days_old": age,
                "status": "fresh" if age <= 1 else "stale",
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def compute_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()

    z = (numeric - numeric.mean()) / numeric.std(ddof=0)
    mask = z.abs() > 3

    rows = []
    for col in numeric.columns:
        idx = mask.index[mask[col].fillna(False)]
        for i in idx:
            rows.append({"column": col, "row_index": int(i), "value": float(df.loc[i, col])})
    return pd.DataFrame(rows)


def friendly_feature_name(feature: str) -> str:
    if feature.startswith("rain_mm_lag"):
        day = feature.replace("rain_mm_lag", "")
        return f"Rainfall {day} day(s) ago"
    if feature.startswith("gatun_level_lag"):
        day = feature.replace("gatun_level_lag", "")
        return f"Gatun level {day} day(s) ago"

    mapping = {
        "rain_7d_sum": "Rainfall total (last 7 days)",
        "rain_14d_sum": "Rainfall total (last 14 days)",
        "rain_7d_mean": "Rainfall average (last 7 days)",
        "rain_14d_mean": "Rainfall average (last 14 days)",
        "sin_day_of_year": "Seasonality (sine)",
        "cos_day_of_year": "Seasonality (cosine)",
        "month": "Month of year",
    }
    return mapping.get(feature, feature)


try:
    feature_df = load_feature_table("data/raw")
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    st.stop()

forecast_df = load_forecasts()
freshness_df = source_freshness_table("data/raw")
latest = feature_df.iloc[-1]
latest_data_date = pd.Timestamp(latest["date"]).date()

overview_tab, forecast_tab, drivers_tab, budget_tab, glossary_tab, qa_tab = st.tabs(
    [
        "Overview",
        "Forecasts",
        "What Drives Forecasts",
        "Water Optimization",
        "Glossary",
        "Data Quality",
    ]
)

with overview_tab:
    st.subheader("Current Status")
    st.write(
        "This section shows the latest observed lake level and how recent each data source is."
    )

    days_old = (pd.Timestamp.today().normalize() - pd.Timestamp(latest_data_date)).days
    freshness = "Up-to-date" if days_old <= 1 else "Needs refresh"

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Gatun Level", f"{latest['gatun_level']:.3f}")
    c2.metric("Model data as of", str(latest_data_date))
    c3.metric("Freshness", f"{freshness} ({days_old} days old)")

    st.markdown("**Source freshness check**")
    st.dataframe(freshness_df, use_container_width=True)

    if st.button("Update + Reforecast now"):
        with st.spinner("Updating raw files and regenerating forecasts..."):
            try:
                result = update_raw_files(data_dir="data/raw", bulk_url=DEFAULT_BULK_URL)
                forecast = generate_forecasts(
                    data_dir="data/raw",
                    model_dir="models",
                    output_csv=default_latest_forecast_path(),
                )
                st.success(
                    "Data updated. "
                    f"Latest imported date: {result['latest_date']} | "
                    f"Rows imported: {result['rows_imported']} | "
                    f"Forecast rows regenerated: {len(forecast)}"
                )
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(
                    "Update failed. If this is a model artifact issue, run training once: "
                    "`python -m src.models.train --data-dir data/raw --model-dir models`. "
                    f"Details: {exc}"
                )

    stale = freshness_df[freshness_df["status"] == "stale"]
    if not stale.empty:
        st.warning(
            "Some sources are stale. If `reservoir_daily.csv` and `met_daily.csv` have the same last date, "
            "this usually means the external source has not published newer daily values yet, or the refresh job has not run."
        )
        st.code("python -m src.io.update_from_bulk_export --data-dir data/raw", language="bash")

    st.subheader("Recent 30-Day Trend")
    tail = feature_df.tail(30)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tail["date"], tail["gatun_level"], label="Gatun level")
    ax.set_xlabel("Date")
    ax.set_ylabel("Level")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)

with forecast_tab:
    st.subheader("Upcoming Lake-Level Forecast")
    st.write(
        "Forecasts are generated from the latest observed day. T+7 means seven days after the data-as-of date."
    )

    if forecast_df.empty:
        st.warning("No forecast file found. Run: python -m src.models.predict --data-dir data/raw --model-dir models")
    else:
        view = forecast_df[forecast_df["target"] == "gatun_level"].copy()
        view = view.sort_values("horizon_days")

        st.info(
            f"Data-as-of date: {view['base_date'].iloc[0]} | "
            f"T+1: {view.loc[view['horizon_days']==1, 'forecast_date'].iloc[0]} | "
            f"T+7: {view.loc[view['horizon_days']==7, 'forecast_date'].iloc[0]} | "
            f"T+14: {view.loc[view['horizon_days']==14, 'forecast_date'].iloc[0]} | "
            f"T+30: {view.loc[view['horizon_days']==30, 'forecast_date'].iloc[0]}"
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(view["horizon_days"], view["predicted_level"], marker="o")
        if {"predicted_level_p05", "predicted_level_p95"}.issubset(view.columns):
            ax.fill_between(
                view["horizon_days"],
                view["predicted_level_p05"],
                view["predicted_level_p95"],
                alpha=0.2,
                label="Approx. 90% range",
            )
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Forecast horizon (days ahead)")
        ax.set_ylabel("Predicted Gatun level")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)

        pretty = view[
            [
                "horizon_days",
                "base_date",
                "forecast_date",
                "model_algorithm",
                "current_level",
                "predicted_level",
                "predicted_level_p05",
                "predicted_level_p95",
                "predicted_change",
                "forecast_std",
            ]
        ].rename(
            columns={
                "horizon_days": "Days ahead (T+)",
                "base_date": "Data as of",
                "forecast_date": "Forecast date",
                "model_algorithm": "Selected model",
                "current_level": "Current level",
                "predicted_level": "Predicted level",
                "predicted_level_p05": "Predicted level (P05)",
                "predicted_level_p95": "Predicted level (P95)",
                "predicted_change": "Expected change",
                "forecast_std": "Forecast uncertainty (std)",
            }
        )
        st.dataframe(pretty, use_container_width=True)

with drivers_tab:
    st.subheader("What Influences the Forecast")
    st.write(
        "This section ranks which inputs mattered most to the model (for the selected forecast horizon)."
    )

    selected_horizon = st.selectbox("Choose forecast horizon", HORIZONS, index=1)
    model_file = model_path("models", "gatun_level", selected_horizon)

    if not model_file.exists():
        st.warning(f"Model not found: {model_file}. Train models first.")
    else:
        bundle = joblib.load(model_file)
        model = bundle["model"]
        feature_cols = bundle["feature_columns"]
        metrics = bundle.get("metrics", {})
        st.caption(
            f"Selected algorithm for T+{selected_horizon}: {metrics.get('algorithm', 'unknown')} "
            f"(RMSE={metrics.get('rmse', float('nan')):.4f})"
        )

        leaderboard = metrics.get("leaderboard", [])
        if leaderboard:
            st.markdown("**Model comparison (cross-validation)**")
            st.dataframe(pd.DataFrame(leaderboard), use_container_width=True)

        if hasattr(model, "feature_importances_"):
            importances = pd.DataFrame(
                {"feature": feature_cols, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            importances["feature_friendly"] = importances["feature"].apply(friendly_feature_name)

            top = importances.head(15)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(top["feature_friendly"][::-1], top["importance"][::-1])
            ax.set_xlabel("Relative importance")
            st.pyplot(fig)
        else:
            st.info("Selected model type does not expose feature importances.")

    st.caption("Climate-index features were removed in this configuration.")

with budget_tab:
    st.subheader("Water Optimization")
    st.write(
        "This section runs multi-objective Monte Carlo optimization and compares best policies for "
        "hydropower generation, ship throughput, and balanced overall impact."
    )
    st.markdown(
        "Key terms: [M&I demand](#term-mi-demand), [environmental flow](#term-environmental-flow), "
        "[navigation demand](#term-navigation-demand), [energy value](#term-energy-price), "
        "[ship value](#term-ship-value), [deficit penalty](#term-deficit-penalty), "
        "[salinity penalty](#term-salinity-penalty), [minimum flushing share](#term-min-flushing-share). "
        "Open the `Glossary` tab for full definitions."
    )

    if forecast_df.empty:
        st.warning("Forecasts are required for optimization.")
    else:
        view = forecast_df[forecast_df["target"] == "gatun_level"].copy().sort_values("horizon_days")
        horizon = st.selectbox("Choose horizon", HORIZONS, index=1)
        selected = view[view["horizon_days"] == horizon].iloc[0]

        st.caption(
            f"Using forecast from {selected['base_date']} to {selected['forecast_date']} (T+{horizon})."
        )

        st.markdown("**Optimization controls**")
        c1, c2 = st.columns(2)
        n_sims = c1.slider("Monte Carlo simulations", min_value=1000, max_value=30000, step=1000, value=6000)
        energy_price = c2.number_input("Energy value ($/MWh)", min_value=10.0, max_value=1000.0, value=90.0)

        c3, c4, c5, c6 = st.columns(4)
        ship_value = c3.number_input("Ship value ($/additional transit)", min_value=100.0, max_value=100000.0, value=12000.0)
        deficit_penalty = c4.number_input(
            "Deficit penalty ($/hm3)", min_value=10.0, max_value=1000.0, value=130.0
        )
        reserve_bonus = c5.number_input("Reserve bonus ($/hm3)", min_value=0.0, max_value=500.0, value=10.0)
        flushing_bonus = c6.number_input(
            "Flushing bonus ($/hm3)", min_value=0.0, max_value=200.0, value=8.0
        )

        c7, c8 = st.columns(2)
        min_flushing_share = c7.slider(
            "Minimum flushing share of discretionary water",
            min_value=0.10,
            max_value=0.70,
            step=0.05,
            value=0.30,
        )
        salinity_penalty = c8.number_input(
            "Salinity penalty ($/hm3 shortfall)",
            min_value=10.0,
            max_value=1500.0,
            value=140.0,
        )

        result = optimize_with_monte_carlo(
            predicted_change_ft=float(selected["predicted_change"]),
            forecast_std_ft=float(selected.get("forecast_std", 0.02)),
            horizon_days=int(horizon),
            recent_df=feature_df.tail(14),
            n_sims=int(n_sims),
            energy_price_per_mwh=float(energy_price),
            ship_value_per_transit=float(ship_value),
            deficit_penalty_per_hm3=float(deficit_penalty),
            reserve_bonus_per_hm3=float(reserve_bonus),
            flushing_bonus_per_hm3=float(flushing_bonus),
            min_flushing_share=float(min_flushing_share),
            salinity_penalty_per_hm3=float(salinity_penalty),
        )

        st.markdown("**Objective comparison**")
        st.dataframe(result["objective_comparison"], use_container_width=True)

        st.markdown("**Best policy by objective**")
        for mode, title in [("hydropower", "Max Hydropower"), ("ships", "Max Ships Passed"), ("balanced", "Balanced Impact")]:
            best = result[mode]["best_policy"]
            summary = result[mode]["summary"]
            st.markdown(f"**{title}**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Hydro share", f"{best['hydro_share_pct']:.1f}%")
            k2.metric("Flushing share", f"{best['flushing_extra_share_pct']:.1f}%")
            k3.metric("Ship-support share", f"{best['ship_support_share_pct']:.1f}%")
            k4.metric("Reserve share", f"{best['reserve_share_pct']:.1f}%")
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected energy", f"{summary['expected_energy_mwh_best']:.2f} MWh")
            m2.metric("Expected extra transits", f"{summary['expected_additional_transits_best']:.2f}")
            m3.metric("Deficit probability", f"{100*summary['deficit_probability_best']:.1f}%")

        st.markdown("**Overall best impact (balanced objective)**")
        overall = result["overall_best"]
        st.dataframe(overall["leaderboard"].head(10), use_container_width=True)

        st.markdown("**Simulation visualization (overall best impact)**")
        sample = overall["simulation_samples"]
        fig1, ax1 = plt.subplots(figsize=(9, 4))
        ax1.hist(sample["best_score"], bins=50, alpha=0.6, label="Best policy")
        ax1.hist(sample["baseline_score"], bins=50, alpha=0.6, label="Baseline policy")
        ax1.set_xlabel("Simulated optimization score")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.scatter(sample["best_energy_mwh"], sample["best_additional_transits"], alpha=0.2, s=8)
        ax2.set_xlabel("Simulated energy (MWh)")
        ax2.set_ylabel("Simulated additional transits")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)

        gs = result["global_summary"]
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Mandatory environmental flow", f"{gs['minimum_environmental_flow_hm3']:.2f} hm3")
        g2.metric("Mandatory hydro from env flow", f"{gs['mandatory_hydropower_mwh']:.2f} MWh")
        g3.metric("Min discretionary flushing share", f"{100*gs['minimum_discretionary_flushing_share']:.1f}%")
        g4.metric("Water per transit assumption", f"{gs['hm3_per_transit_assumption']:.3f} hm3/transit")

        st.markdown("**Assumptions**")
        st.caption(
            "Level-change forecasts are converted to storage change and allocated to hydropower, flushing, "
            "ship-support, and reserve under objective-specific scoring. Environmental flow, M&I demand, and "
            "navigation demand remain mandatory baseline constraints."
        )

with glossary_tab:
    st.subheader("Definitions and Concepts")
    st.markdown("### <a name='term-mi-demand'></a>M&I demand", unsafe_allow_html=True)
    st.write("Municipal and industrial demand. This is treated as mandatory and should be fully met.")
    st.markdown("### <a name='term-navigation-demand'></a>Navigation demand", unsafe_allow_html=True)
    st.write(
        "Water needed for lock operations and canal transits. Treated as mandatory. "
        "If `water_for_locks` is missing, the app estimates demand from `panamax_transits` and "
        "`neopanamax_transits` with level-dependent Panamax factors."
    )
    st.markdown("### <a name='term-environmental-flow'></a>Environmental flow", unsafe_allow_html=True)
    st.write(
        "Minimum river/ecosystem release requirement. In this DSS it is modeled as a hard constraint and "
        "as must-run flow with energy recovery."
    )
    st.markdown("### <a name='term-salinity-control'></a>Salinity control / flushing", unsafe_allow_html=True)
    st.write(
        "Operational releases used to reduce salinity intrusion risk near drinking-water intakes and key reaches."
    )
    st.markdown("### <a name='term-discretionary-water'></a>Discretionary water", unsafe_allow_html=True)
    st.write(
        "Water available after mandatory demands are covered. Optimizer allocates this among hydropower, "
        "extra flushing, ship-support, and storage reserve."
    )
    st.markdown("### <a name='term-energy-price'></a>Energy value ($/MWh)", unsafe_allow_html=True)
    st.write("Economic value assigned to generated electricity in the objective scoring.")
    st.markdown("### <a name='term-ship-value'></a>Ship value ($/additional transit)", unsafe_allow_html=True)
    st.write("Estimated value/benefit for one additional supported ship transit.")
    st.markdown("### <a name='term-deficit-penalty'></a>Deficit penalty ($/hm3)", unsafe_allow_html=True)
    st.write("Penalty for negative water-balance outcomes that imply storage stress/risk.")
    st.markdown("### <a name='term-salinity-penalty'></a>Salinity penalty ($/hm3 shortfall)", unsafe_allow_html=True)
    st.write("Penalty when discretionary flushing is below minimum salinity-control target.")
    st.markdown("### <a name='term-min-flushing-share'></a>Minimum flushing share", unsafe_allow_html=True)
    st.write(
        "Minimum fraction of discretionary water that should be allocated to flushing to protect water quality."
    )
    st.markdown("### Guide Curves and Zones")
    st.write(
        "Guide curves define monthly level bands (inactive/buffer/conservation/flood/spill) that govern release rules. "
        "The current app uses simplified logic and is structured to accept explicit monthly zone tables."
    )

with qa_tab:
    st.subheader("Data Quality Checks")
    st.write(
        "These checks help confirm the forecast inputs are complete and reliable before decisions are made."
    )

    st.markdown("**Missing values by column**")
    missing = feature_df.isna().sum().to_frame("missing_count")
    missing["missing_pct"] = (missing["missing_count"] / len(feature_df)) * 100
    st.dataframe(missing.sort_values("missing_count", ascending=False), use_container_width=True)

    st.markdown("**Outliers (z-score > 3)**")
    outliers = compute_outliers(feature_df)
    if outliers.empty:
        st.success("No extreme outliers detected by this rule.")
    else:
        st.dataframe(outliers.head(200), use_container_width=True)

    st.markdown("**Feature validation**")
    try:
        validate_engineered_features(feature_df)
        st.success("All required engineered features are present.")
    except FeatureValidationError as exc:
        st.error(str(exc))
