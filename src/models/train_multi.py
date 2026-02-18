import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from features.build_features import build_features


DATA_PATH = "data/processed/gatun_daily_with_oni.csv"
OUT_DIR = Path("src/models/multi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 15))  # 1..14 days

df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()

# --- CORE model uses long-history variables only (drop evap if it causes truncation) ---
core = df[["lake_level_ft", "rainfall_mm", "oni"]].copy()
  # keep long history
core = build_features(core)

base_features = [
    "lake_level_ft", "level_lag1", "level_change_1d",
    "rain_3d", "rain_7d", "rain_14d",
    "month", "dayofyear",
    "oni", "oni_3mo", "oni_trend_3mo"
]


core = core.dropna()

metrics = {}

for h in HORIZONS:
    temp = core.copy()
    temp["y"] = temp["lake_level_ft"].shift(-h)
    temp = temp.dropna()

    X = temp[base_features]
    y = temp["y"]

    split = int(len(temp) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1, max_depth=14
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    joblib.dump(
        {"model": model, "features": base_features, "horizon": h, "rmse": rmse},
        OUT_DIR / f"rf_h{h:02d}.joblib"
    )
    metrics[h] = rmse
    print(f"h={h:02d} | RMSE={rmse:.3f} ft | n={len(temp)}")

joblib.dump(metrics, OUT_DIR / "rmse_by_horizon.joblib")
print("✅ Saved models to", OUT_DIR)
