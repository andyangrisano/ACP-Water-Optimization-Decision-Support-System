"""Load raw CSV inputs with schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_FILENAMES, REQUIRED_MET_COLUMNS, REQUIRED_RESERVOIR_COLUMNS


class DataSchemaError(ValueError):
    """Raised when required columns are missing from an input dataset."""



def _ensure_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataSchemaError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Expected at least: {required}."
        )



def _load_csv_with_date(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise DataSchemaError(f"{dataset_name} must include a 'date' column: {path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        raise DataSchemaError(f"{dataset_name} contains {bad} invalid date values in {path}")

    return df.sort_values("date").reset_index(drop=True)



def load_reservoir_data(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    path = data_dir / RAW_FILENAMES["reservoir"]
    df = _load_csv_with_date(path, "reservoir_daily")
    _ensure_columns(df, REQUIRED_RESERVOIR_COLUMNS, "reservoir_daily")
    return df



def load_met_data(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    path = data_dir / RAW_FILENAMES["met"]
    df = _load_csv_with_date(path, "met_daily")
    _ensure_columns(df, REQUIRED_MET_COLUMNS, "met_daily")
    return df



def load_all_raw_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        load_reservoir_data(data_dir),
        load_met_data(data_dir),
    )
