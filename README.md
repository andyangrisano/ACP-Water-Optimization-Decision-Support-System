# Panama Canal Water Optimization DSS

AI-powered decision support system for forecasting Gatun Lake levels and translating forecasts into operational water allocation guidance.

## Tech Stack

- Python 3.11+
- pandas, numpy
- scikit-learn
- xgboost (optional at runtime, auto-fallback to sklearn GradientBoosting)
- joblib
- matplotlib
- streamlit
- pytest

## Project Structure

```text
canal-water-dss/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── config.py
│   ├── io/
│   │   ├── load_data.py
│   │   └── merge_data.py
│   ├── features/
│   │   ├── build_features.py
│   │   └── validate_features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── registry.py
│   ├── ops/
│   │   ├── water_budget.py
│   │   └── monte_carlo_optimize.py
│   └── utils/
│       └── time_series_split.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── tests/
│   ├── test_features.py
│   └── test_pipeline_smoke.py
├── requirements.txt
└── README.md
```

## Required Raw CSV Schema

Place CSVs in `data/raw/`.

### `reservoir_daily.csv`
Required columns:
- `date`
- `gatun_level`

Optional columns:
- `gatun_release`
- `spill`
- `water_for_locks`
- `water_for_municipal`
- `panamax_transits`
- `neopanamax_transits`

### `met_daily.csv`
Required columns:
- `date`
- `rain_mm`

Optional columns:
- `temp_c`
- `et0_mm`

## Operational Water Usage (Ships)

The DSS handles navigation demand in this order:
- If `water_for_locks` exists, it uses it directly.
- Otherwise, it estimates lock water from ship passages:
  - `panamax_transits`
  - `neopanamax_transits`

Panamax demand is modeled as level-sensitive (conservation behavior under lower Gatun levels), while NeoPanamax is modeled with a fixed per-transit factor. Tune constants in `src/config.py` and logic in `src/ops/navigation_demand.py`.

## Feature Engineering

- Lags: `rain_mm_lag1..14`, `gatun_level_lag1..14`
- Rolling: `rain_7d_sum`, `rain_14d_sum`, `rain_7d_mean`, `rain_14d_mean`
- Seasonality: `sin_day_of_year`, `cos_day_of_year`, `month`

## Forecast Horizons

Separate models are trained for each horizon:
- Target: `gatun_level`
- Horizons: 1, 7, 14, 30 days

Total models: 4.

## Model Selection + Uncertainty

- For each horizon, the trainer evaluates multiple algorithms with time-series CV:
  - Gradient Boosting
  - Random Forest
  - Extra Trees
  - Ridge / Linear Regression
  - XGBoost (if installed)
- The best model (lowest RMSE) is saved per horizon.
- Prediction output includes:
  - selected algorithm
  - forecast uncertainty (`forecast_std`)
  - approximate P05/P95 level range

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.models.train --data-dir data/raw --model-dir models
```

## Predict

```bash
python -m src.models.predict --data-dir data/raw --model-dir models
```

Writes forecasts to `data/processed/latest_forecast.csv`.

## Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Tabs include:
- Overview (current levels, source freshness, in-app data refresh)
- Forecasts (1/7/14/30 day projections)
- Drivers (feature importance and model comparison)
- Water Optimization (multi-objective optimization: maximize hydropower, maximize ships, balanced impact)
- Glossary (definitions of optimization controls and operational terms)
- Data QA (missingness, outliers, feature checks)

## Testing

```bash
pytest
```

Includes:
- Feature engineering/validation tests
- End-to-end smoke pipeline test (synthetic data)

## Daily Data Refresh (Panama Bulk Export)

```bash
python -m src.io.update_from_bulk_export --data-dir data/raw
```

The app includes an in-app refresh button in `Overview` that triggers this same updater.

## Run Online (GitHub + Streamlit Cloud)

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the GitHub repo.
3. Set app entrypoint to:

```text
app/streamlit_app.py
```

This repo includes:
- `runtime.txt` (Python 3.11)
- `.streamlit/config.toml`
- `.github/workflows/ci.yml` for automated test runs on push/PR
