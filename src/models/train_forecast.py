import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

DATA_PATH = "data/processed/gatun_daily_clean.csv"
MODEL_PATH = "src/models/model_rf.joblib"

HORIZON_DAYS = 7  # predict level 7 days ahead

df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()

# --- Feature engineering (simple + strong MVP) ---
df["level_lag1"] = df["lake_level_ft"].shift(1)
df["level_change_1d"] = df["lake_level_ft"].diff(1)

df["rain_3d"] = df["rainfall_mm"].rolling(3).sum()
df["rain_7d"] = df["rainfall_mm"].rolling(7).sum()
df["rain_14d"] = df["rainfall_mm"].rolling(14).sum()

df["evap_7d"] = df["evaporation_hm3"].rolling(7).mean()

# Seasonality features
df["month"] = df.index.month
df["dayofyear"] = df.index.dayofyear

# Target: lake level in HORIZON_DAYS
df["y"] = df["lake_level_ft"].shift(-HORIZON_DAYS)

df = df.dropna()

features = [
    "lake_level_ft", "level_lag1", "level_change_1d",
    "rain_3d", "rain_7d", "rain_14d",
    "evap_7d", "month", "dayofyear"
]

X = df[features]
y = df["y"]

# Time-based split (no shuffling)
split_idx = int(len(df) * 0.85)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=12
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
residuals = (y_test.values - pred)
rmse = float(np.sqrt(np.mean(residuals**2)))
joblib.dump(
    {"model": model, "features": features, "horizon_days": HORIZON_DAYS, "rmse": rmse},
    MODEL_PATH
)
print(f"✅ Test RMSE: {rmse:.3f} ft")

mae = mean_absolute_error(y_test, pred)

Path("src/models").mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model, "features": features, "horizon_days": HORIZON_DAYS}, MODEL_PATH)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Horizon: {HORIZON_DAYS} days")
print(f"✅ Test MAE: {mae:.3f} ft")
