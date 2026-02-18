from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.predict import generate_forecasts
from src.models.train import train_models



def _write_synthetic_raw_data(data_dir: Path, n_days: int = 220) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = pd.Series(range(n_days), dtype=float)

    reservoir = pd.DataFrame(
        {
            "date": dates,
            "gatun_level": 80 + 0.02 * t + 0.3 * (t % 10),
            "gatun_release": 100 + (t % 5),
            "panamax_transits": 18 + (t % 7),
            "neopanamax_transits": 9 + (t % 4),
            "water_for_municipal": 10 + (t % 3),
        }
    )
    reservoir.to_csv(data_dir / "reservoir_daily.csv", index=False)

    met = pd.DataFrame(
        {
            "date": dates,
            "rain_mm": 5 + (t % 12),
            "temp_c": 27 + (t % 4) * 0.2,
            "et0_mm": 3 + (t % 5) * 0.1,
        }
    )
    met.to_csv(data_dir / "met_daily.csv", index=False)



def test_pipeline_smoke(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    model_dir = tmp_path / "models"
    forecast_path = tmp_path / "latest_forecast.csv"

    _write_synthetic_raw_data(raw_dir)

    train_report = train_models(raw_dir, model_dir)
    assert train_report.shape[0] == 4

    forecast_df = generate_forecasts(raw_dir, model_dir, forecast_path)

    assert forecast_df.shape[0] == 4
    assert {"reservoir", "horizon_days", "predicted_level", "predicted_change"}.issubset(
        set(forecast_df.columns)
    )
    assert forecast_path.exists()
