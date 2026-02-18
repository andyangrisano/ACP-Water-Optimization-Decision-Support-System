"""Generate multi-horizon reservoir forecasts from trained models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import HORIZONS, TARGETS, default_latest_forecast_path
from src.features.build_features import build_features
from src.features.validate_features import FeatureValidationError, validate_engineered_features, validate_raw_inputs
from src.io.load_data import load_all_raw_data
from src.io.merge_data import merge_inputs
from src.models.registry import model_path



def _load_feature_table(data_dir: str | Path) -> pd.DataFrame:
    reservoir_df, met_df = load_all_raw_data(data_dir)
    validate_raw_inputs(reservoir_df, met_df)
    merged = merge_inputs(reservoir_df, met_df)
    feature_df = build_features(merged)
    validate_engineered_features(feature_df)
    return feature_df



def generate_forecasts(
    data_dir: str | Path,
    model_dir: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    feature_df = _load_feature_table(data_dir)
    model_dir = Path(model_dir)

    rows: list[dict[str, object]] = []

    for target in TARGETS:
        for horizon in HORIZONS:
            path = model_path(model_dir, target, horizon)
            if not path.exists():
                raise FileNotFoundError(f"Trained model not found: {path}")

            bundle = joblib.load(path)
            model = bundle["model"]
            expected_features = bundle["feature_columns"]

            missing_features = [c for c in expected_features if c not in feature_df.columns]
            if missing_features:
                raise FeatureValidationError(
                    f"Prediction features do not match model expectations for {target} h={horizon}. "
                    f"Missing: {missing_features}"
                )

            valid_df = feature_df.dropna(subset=expected_features + [target])
            if valid_df.empty:
                raise ValueError(f"No valid rows available for prediction of {target} h={horizon}")

            latest_row = valid_df.iloc[-1]
            pred = float(model.predict(valid_df[expected_features].iloc[[-1]])[0])
            current_level = float(latest_row[target])
            base_date = pd.Timestamp(latest_row["date"])
            metrics = bundle.get("metrics", {})
            forecast_std = float(metrics.get("residual_std", metrics.get("rmse", 0.0)))
            interval_radius = 1.645 * forecast_std

            rows.append(
                {
                    "generated_at": pd.Timestamp.utcnow().isoformat(),
                    "reservoir": target.replace("_level", ""),
                    "target": target,
                    "model_algorithm": metrics.get("algorithm", "unknown"),
                    "horizon_days": horizon,
                    "base_date": base_date.date().isoformat(),
                    "forecast_date": (base_date + pd.Timedelta(days=horizon)).date().isoformat(),
                    "current_level": current_level,
                    "predicted_level": pred,
                    "predicted_change": pred - current_level,
                    "forecast_std": forecast_std,
                    "predicted_level_p05": pred - interval_radius,
                    "predicted_level_p95": pred + interval_radius,
                }
            )

    forecast_df = pd.DataFrame(rows).sort_values(["target", "horizon_days"]).reset_index(drop=True)

    if output_csv is None:
        output_csv = default_latest_forecast_path()

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_path, index=False)

    return forecast_df



def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Canal Water DSS forecasts")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(default_latest_forecast_path()),
        help="Path to write latest forecast CSV",
    )
    return parser



def main() -> None:
    args = _build_arg_parser().parse_args()
    forecast = generate_forecasts(args.data_dir, args.model_dir, args.output_csv)
    print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
