"""Train multi-horizon Gatun forecasting models with model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import HORIZONS, TARGETS, get_feature_columns
from src.features.build_features import build_features
from src.features.validate_features import validate_engineered_features, validate_raw_inputs
from src.io.load_data import load_all_raw_data
from src.io.merge_data import merge_inputs
from src.models.registry import get_model_candidates, metadata_path, model_path
from src.utils.time_series_split import make_time_series_split


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def _prepare_training_frame(
    feature_df: pd.DataFrame,
    target: str,
    horizon: int,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    train_df = feature_df.copy()
    train_df["target_future"] = train_df[target].shift(-horizon)

    required = feature_columns + [target, "target_future"]
    train_df = train_df.dropna(subset=required).reset_index(drop=True)

    X = train_df[feature_columns]
    y = train_df["target_future"]
    baseline_current = train_df[target]
    return X, y, baseline_current


def _evaluate_baseline_rmse(
    X: pd.DataFrame,
    y: pd.Series,
    baseline_current: pd.Series,
    n_splits: int = 5,
) -> float:
    splitter = make_time_series_split(n_splits=n_splits)
    base_rmses: list[float] = []

    for _, test_idx in splitter.split(X):
        y_true = y.iloc[test_idx].to_numpy()
        base_preds = baseline_current.iloc[test_idx].to_numpy()
        fold_base = _compute_metrics(y_true, base_preds)
        base_rmses.append(fold_base["rmse"])

    return float(np.mean(base_rmses))


def _evaluate_model_cv(
    model_template,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict[str, float]:
    splitter = make_time_series_split(n_splits=n_splits)

    model_rmses: list[float] = []
    model_maes: list[float] = []
    residuals: list[float] = []

    for train_idx, test_idx in splitter.split(X):
        model = clone(model_template)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])

        y_true = y.iloc[test_idx].to_numpy()
        fold_model = _compute_metrics(y_true, preds)

        model_rmses.append(fold_model["rmse"])
        model_maes.append(fold_model["mae"])
        residuals.extend((y_true - preds).tolist())

    return {
        "rmse": float(np.mean(model_rmses)),
        "rmse_std": float(np.std(model_rmses, ddof=0)),
        "mae": float(np.mean(model_maes)),
        "residual_std": float(np.std(np.array(residuals), ddof=0)) if residuals else 0.0,
    }


def _select_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    baseline_current: pd.Series,
) -> tuple[str, object, dict[str, object]]:
    candidates = get_model_candidates(random_state=42)
    baseline_rmse = _evaluate_baseline_rmse(X=X, y=y, baseline_current=baseline_current)

    leaderboard: list[dict[str, float | str]] = []
    best_name = ""
    best_model = None
    best_rmse = float("inf")
    best_stats: dict[str, float] = {}

    for name, model_template in candidates.items():
        stats = _evaluate_model_cv(model_template=model_template, X=X, y=y)
        skill = float(1.0 - (stats["rmse"] / baseline_rmse)) if baseline_rmse > 0 else 0.0

        leaderboard.append(
            {
                "algorithm": name,
                "rmse": stats["rmse"],
                "rmse_std": stats["rmse_std"],
                "mae": stats["mae"],
                "residual_std": stats["residual_std"],
                "baseline_rmse": baseline_rmse,
                "skill_vs_persistence": skill,
            }
        )

        if stats["rmse"] < best_rmse:
            best_rmse = stats["rmse"]
            best_name = name
            best_model = model_template
            best_stats = stats

    if best_model is None:
        raise ValueError("No model candidates were available for training.")

    best_skill = float(1.0 - (best_stats["rmse"] / baseline_rmse)) if baseline_rmse > 0 else 0.0
    metrics: dict[str, object] = {
        "algorithm": best_name,
        "rmse": best_stats["rmse"],
        "rmse_std": best_stats["rmse_std"],
        "mae": best_stats["mae"],
        "residual_std": best_stats["residual_std"],
        "baseline_rmse": baseline_rmse,
        "skill_vs_persistence": best_skill,
        "leaderboard": sorted(leaderboard, key=lambda row: float(row["rmse"])),
    }
    return best_name, best_model, metrics


def train_models(data_dir: str | Path, model_dir: str | Path) -> pd.DataFrame:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    reservoir_df, met_df = load_all_raw_data(data_dir)
    validate_raw_inputs(reservoir_df, met_df)

    merged = merge_inputs(reservoir_df, met_df)
    feature_df = build_features(merged)

    feature_columns = get_feature_columns()
    validate_engineered_features(feature_df, feature_columns)

    rows: list[dict[str, float | str | int]] = []

    for target in TARGETS:
        for horizon in HORIZONS:
            X, y, baseline_current = _prepare_training_frame(
                feature_df, target, horizon, feature_columns
            )
            if len(X) < 30:
                raise ValueError(
                    f"Not enough training rows for {target} horizon={horizon}. "
                    f"Need at least 30 rows after lag/target alignment, got {len(X)}."
                )

            best_name, best_model_template, metrics = _select_best_model(
                X=X,
                y=y,
                baseline_current=baseline_current,
            )

            final_model = clone(best_model_template)
            final_model.fit(X, y)

            bundle = {
                "model": final_model,
                "feature_columns": feature_columns,
                "target": target,
                "horizon": horizon,
                "metrics": metrics,
            }

            joblib.dump(bundle, model_path(model_dir, target, horizon))
            with open(metadata_path(model_dir, target, horizon), "w", encoding="utf-8") as f:
                json.dump(bundle["metrics"], f, indent=2)

            rows.append(
                {
                    "target": target,
                    "horizon": horizon,
                    "algorithm": best_name,
                    "rmse": float(metrics["rmse"]),
                    "rmse_std": float(metrics["rmse_std"]),
                    "mae": float(metrics["mae"]),
                    "baseline_rmse": float(metrics["baseline_rmse"]),
                    "skill_vs_persistence": float(metrics["skill_vs_persistence"]),
                }
            )

    return pd.DataFrame(rows).sort_values(["target", "horizon"]).reset_index(drop=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Canal Water DSS models")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--model-dir", type=str, default="models")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = train_models(args.data_dir, args.model_dir)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
