import math
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -----------------------------
# Config / thresholds (Gatún)
# -----------------------------
MAX_LEVEL = 89.0
MIN_LEVEL = 75.0

# Soft guidance levels (you can tune these)
HIGH_ALERT = 88.0   # near-max buffer
LOW_BUFFER = 78.0   # storage protection buffer

DATA_PATH = "data/processed/gatun_daily_with_oni.csv"
  # switch later to ..._with_oni.csv if desired
MODEL_SINGLE_PATH = "src/models/model_rf.joblib"     # optional single-point model
MODELS_MULTI_DIR = "src/models/multi"                # rf_h01..rf_h14


# -----------------------------
# Helpers
# -----------------------------
def norm_cdf(x, mu, sigma):
    if sigma <= 1e-9:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def build_feature_row(df):
    work = df.copy().sort_index()

    work["level_lag1"] = work["lake_level_ft"].shift(1)
    work["level_change_1d"] = work["lake_level_ft"].diff(1)

    work["rain_3d"] = work["rainfall_mm"].rolling(3).sum()
    work["rain_7d"] = work["rainfall_mm"].rolling(7).sum()
    work["rain_14d"] = work["rainfall_mm"].rolling(14).sum()

    # ENSO features (only if ONI exists in the dataset)
    if "oni" in work.columns:
        work.loc[work["oni"] <= -90, "oni"] = pd.NA  # safety in case -99.9 appears
        work["oni"] = work["oni"].ffill()
        work["oni_3mo"] = work["oni"].rolling(3, min_periods=1).mean()
        work["oni_trend_3mo"] = work["oni"] - work["oni"].shift(3)

    work["month"] = work.index.month
    work["dayofyear"] = work.index.dayofyear

    return work.dropna().iloc[-1:]

def multi_horizon_forecast(latest_row: pd.DataFrame, models_dir: str = MODELS_MULTI_DIR) -> pd.DataFrame:
    models_dir = Path(models_dir)
    rows = []
    for h in range(1, 15):
        bundle = joblib.load(models_dir / f"rf_h{h:02d}.joblib")
        model = bundle["model"]
        feats = bundle["features"]
        rmse = float(bundle.get("rmse", 0.7))

        yhat = float(model.predict(latest_row[feats])[0])

        rows.append(
            {
                "horizon_days": h,
                "pred_ft": yhat,
                "rmse_ft": rmse,
                "lower_ft": yhat - 1.0 * rmse,
                "upper_ft": yhat + 1.0 * rmse,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Panama Canal Water Decision Support System (MVP)", layout="wide")

st.title("Panama Canal Water Decision Support System — MVP (Gatún Lake)")
st.subheader("Short-term lake level forecast & operational guidance")

# Climate context selector (manual scenario stress-test)
enso = st.selectbox(
    "Climate context (ENSO)",
    ["La Niña (wetter)", "Neutral", "El Niño (drier)"],
    help="Used to stress-test decisions under different large-scale climate regimes.",
)

# Bias thresholds based on scenario
over_max_trigger = 0.25
below_min_trigger = 0.10
if "La Niña" in enso:
    over_max_trigger = 0.20  # act earlier on overflow risk
elif "El Niño" in enso:
    below_min_trigger = 0.07  # act earlier on drought risk


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()

today = df.index.max().date()
forecast_end = today + timedelta(days=14)
st.caption(f"**Status as of {today}** · Forecast window: **{today} → {forecast_end}**")

current = float(df["lake_level_ft"].iloc[-1])
rain_7d = float(df["rainfall_mm"].tail(7).sum())

# Build latest row + multi-horizon forecast
latest_row = build_feature_row(df)
fc = multi_horizon_forecast(latest_row)

# Day 7 headline
pred7 = float(fc.loc[fc["horizon_days"] == 7, "pred_ft"].iloc[0])
rmse7 = float(fc.loc[fc["horizon_days"] == 7, "rmse_ft"].iloc[0])

# Risk per horizon (normal approx using rmse per horizon)
fc["p_over_max"] = 1 - fc.apply(lambda r: norm_cdf(MAX_LEVEL, r["pred_ft"], r["rmse_ft"]), axis=1)
fc["p_below_min"] = fc.apply(lambda r: norm_cdf(MIN_LEVEL, r["pred_ft"], r["rmse_ft"]), axis=1)

# Summary risk: take max across horizons for quick “worst-case soon”
max_p_over = float(fc["p_over_max"].max())
max_p_below = float(fc["p_below_min"].max())
worst_over_day = int(fc.loc[fc["p_over_max"].idxmax(), "horizon_days"])
worst_below_day = int(fc.loc[fc["p_below_min"].idxmax(), "horizon_days"])


# -----------------------------
# Top summary cards
# -----------------------------
st.subheader("Current Status & Outlook")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current lake level", f"{current:.2f} ft")
c2.metric("Expected level in 7 days", f"{pred7:.2f} ft")
c3.metric("Rainfall (last 7 days)", f"{rain_7d:.1f} mm")
c4.metric("7-day uncertainty (RMSE)", f"{rmse7:.2f} ft")

st.info(f"Operational reference levels · Minimum: **{MIN_LEVEL} ft** · Maximum: **{MAX_LEVEL} ft**")


# -----------------------------
# Forecast chart
# -----------------------------
st.subheader("Expected Gatún Lake Level – Next 14 Days")

plot_df = fc.set_index("horizon_days")[["pred_ft", "lower_ft", "upper_ft"]]
st.line_chart(plot_df)
st.caption("Uncertainty band shown as ±1×RMSE per horizon (MVP uncertainty).")


# -----------------------------
# Recommendation
# -----------------------------
st.subheader("Risk & Recommendation")

r1, r2 = st.columns(2)
r1.metric(f"Max P(exceed {MAX_LEVEL} ft)", f"{max_p_over*100:.1f}%", help=f"Worst day: +{worst_over_day}d")
r2.metric(f"Max P(drop below {MIN_LEVEL} ft)", f"{max_p_below*100:.1f}%", help=f"Worst day: +{worst_below_day}d")

# Simple policy logic (uses scenario-biased triggers)
if current >= HIGH_ALERT or max_p_over >= over_max_trigger:
    action = "🔺 Use excess water proactively: prioritize **hydropower + flushing** to create buffer and avoid forced spill."
elif current <= LOW_BUFFER or max_p_below >= below_min_trigger:
    action = "🔻 Protect storage: **conserve water** (limit discretionary releases; be cautious with flushing)."
else:
    action = "✅ Balanced posture: **moderate hydropower/flushing** while keeping storage stable."

st.markdown("### Recommended operational posture")
st.success(action)

st.caption(
    f"Scenario trigger thresholds (this selection: **{enso}**) · "
    f"Overflow trigger: **{over_max_trigger*100:.0f}%** · "
    f"Drought trigger: **{below_min_trigger*100:.0f}%**"
)


# -----------------------------
# Details (collapsed)
# -----------------------------
with st.expander("View detailed daily risk table"):
    st.dataframe(
        fc[["horizon_days", "pred_ft", "rmse_ft", "lower_ft", "upper_ft", "p_over_max", "p_below_min"]],
        use_container_width=True,
    )

with st.expander("View recent history (last 12 months)"):
    st.line_chart(df[["lake_level_ft"]].tail(365))

with st.expander("Technical notes"):
    st.write(
        "- Forecast uses one model per horizon (1–14 days).\n"
        "- Risk probabilities are approximated with a Normal distribution using horizon RMSE.\n"
        "- Next upgrades: integrate ONI directly (ENSO index) and train on longer historical record (1960s–present)."
    )

# (Optional) Single-point model compatibility if you still want it
# Kept here but not shown unless the file exists
if Path(MODEL_SINGLE_PATH).exists():
    with st.expander("Legacy single-point forecast (optional)"):
        bundle = joblib.load(MODEL_SINGLE_PATH)
        st.write(f"Model horizon: {bundle.get('horizon_days', 'unknown')} days")
        st.write("This section is optional; multi-horizon forecast is the primary view.")
