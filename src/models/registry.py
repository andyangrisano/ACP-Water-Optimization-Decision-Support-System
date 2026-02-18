"""Model registry and artifact naming helpers."""

from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


def get_model_candidates(random_state: int = 42) -> dict[str, object]:
    candidates: dict[str, object] = {
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ridge": Ridge(alpha=1.0),
        "linear_regression": LinearRegression(),
    }

    try:
        from xgboost import XGBRegressor

        candidates["xgboost"] = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    except Exception:
        pass

    return candidates


def create_regressor(random_state: int = 42):
    candidates = get_model_candidates(random_state=random_state)
    if "xgboost" in candidates:
        return candidates["xgboost"], "xgboost"
    return candidates["gradient_boosting"], "gradient_boosting"


def artifact_stem(target: str, horizon: int) -> str:
    return f"{target}_h{horizon:02d}"


def model_path(model_dir: str | Path, target: str, horizon: int) -> Path:
    return Path(model_dir) / f"{artifact_stem(target, horizon)}.joblib"


def metadata_path(model_dir: str | Path, target: str, horizon: int) -> Path:
    return Path(model_dir) / f"{artifact_stem(target, horizon)}.json"
