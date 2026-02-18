"""Update DSS raw files from Panama Aquatic Informatics bulk export CSV."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


DEFAULT_BULK_URL = (
    "https://panama.aquaticinformatics.net/Export/BulkExport"
    "?DateRange=EntirePeriodOfRecord&TimeZone=-5&Calendar=CALENDARYEAR"
    "&Interval=Daily&Step=1&ExportFormat=csv&TimeAligned=True&RoundData=False"
    "&IncludeGradeCodes=undefined&IncludeApprovalLevels=undefined"
    "&IncludeQualifiers=undefined&IncludeInterpolationTypes=False"
    "&Datasets[0].DatasetName=Lake-Res%20elevation.Telem%20AVG%40GAT"
    "&Datasets[0].Calculation=Aggregate&Datasets[0].UnitId=70"
    "&Datasets[1].DatasetName=Precipitation.Daily%20Telemetria%40GAT"
    "&Datasets[1].Calculation=Aggregate&Datasets[1].UnitId=75"
    "&Datasets[2].DatasetName=Total%20Storage.V%20Evap%20GAT%200.7%40GAT"
    "&Datasets[2].Calculation=Aggregate&Datasets[2].UnitId=181"
)


class BulkExportError(ValueError):
    """Raised when the bulk export payload cannot be parsed."""



def _download_csv_text(url: str, timeout_sec: int = 45) -> str:
    with urlopen(url, timeout=timeout_sec) as resp:
        return resp.read().decode("utf-8", errors="replace")



def _find_header_row(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        lower = line.lower()
        if "start of interval" in lower and "(" in line and ")" in line:
            return i
    raise BulkExportError(
        "Could not find bulk-export header row. Expected a row containing 'Start of Interval'."
    )



def _pick_column(columns: list[str], candidates: list[str], label: str) -> str:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        for c_lower, c_raw in lowered.items():
            if cand in c_lower:
                return c_raw
    raise BulkExportError(f"Could not map required column for {label}. Available columns: {columns}")



def parse_bulk_export_to_frame(csv_text: str) -> pd.DataFrame:
    lines = csv_text.splitlines()
    header_idx = _find_header_row(lines)
    payload = "\n".join(lines[header_idx:])

    df = pd.read_csv(StringIO(payload))
    df.columns = [c.strip() for c in df.columns]

    date_col = _pick_column(df.columns.tolist(), ["start of interval"], "date")
    level_col = _pick_column(df.columns.tolist(), ["average (ft)", "average (m)", "lake"], "gatun_level")
    rain_col = _pick_column(df.columns.tolist(), ["total (mm)", "precip"], "rain_mm")

    evap_col = None
    for cand in ["evap", "hm^3", "storage"]:
        for c in df.columns:
            if cand in c.lower():
                evap_col = c
                break
        if evap_col is not None:
            break

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["gatun_level"] = pd.to_numeric(df[level_col], errors="coerce")
    out["rain_mm"] = pd.to_numeric(df[rain_col], errors="coerce")
    if evap_col is not None:
        out["gatun_release"] = pd.to_numeric(df[evap_col], errors="coerce")

    out = out.dropna(subset=["date", "gatun_level", "rain_mm"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    if out.empty:
        raise BulkExportError("Parsed bulk export has no usable rows after cleaning.")

    return out



def _merge_update(base: pd.DataFrame, updates: pd.DataFrame, cols_to_update: list[str]) -> pd.DataFrame:
    base = base.copy()
    updates = updates.copy()

    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    updates["date"] = pd.to_datetime(updates["date"], errors="coerce")

    merged = base.merge(updates[["date"] + cols_to_update], on="date", how="outer", suffixes=("", "_new"))
    for c in cols_to_update:
        new_col = f"{c}_new"
        if new_col in merged.columns:
            merged[c] = merged[new_col].combine_first(merged.get(c))
            merged = merged.drop(columns=[new_col])

    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return merged



def update_raw_files(data_dir: str | Path, bulk_url: str) -> dict[str, str]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    reservoir_path = data_dir / "reservoir_daily.csv"
    met_path = data_dir / "met_daily.csv"

    if not reservoir_path.exists() or not met_path.exists():
        raise FileNotFoundError(
            "Expected existing data/raw/reservoir_daily.csv and data/raw/met_daily.csv before update."
        )

    csv_text = _download_csv_text(bulk_url)
    updates = parse_bulk_export_to_frame(csv_text)

    reservoir = pd.read_csv(reservoir_path)
    met = pd.read_csv(met_path)

    reservoir_updated = _merge_update(reservoir, updates, cols_to_update=["gatun_level"])
    met_updated = _merge_update(met, updates, cols_to_update=["rain_mm"])

    if "gatun_release" in updates.columns:
        reservoir_updated = _merge_update(reservoir_updated, updates, cols_to_update=["gatun_release"])

    reservoir_updated["date"] = pd.to_datetime(reservoir_updated["date"]).dt.date.astype(str)
    met_updated["date"] = pd.to_datetime(met_updated["date"]).dt.date.astype(str)

    reservoir_updated.to_csv(reservoir_path, index=False)
    met_updated.to_csv(met_path, index=False)

    return {
        "reservoir_path": str(reservoir_path),
        "met_path": str(met_path),
        "latest_date": str(pd.to_datetime(updates["date"]).max().date()),
        "rows_imported": str(len(updates)),
    }



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update raw DSS files from Panama bulk export")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--bulk-url", default=DEFAULT_BULK_URL, help="Aquatic Informatics bulk export URL")
    return parser



def main() -> None:
    args = _build_parser().parse_args()
    result = update_raw_files(data_dir=args.data_dir, bulk_url=args.bulk_url)
    print("Updated files:")
    print(f"- reservoir: {result['reservoir_path']}")
    print(f"- met: {result['met_path']}")
    print(f"- rows imported: {result['rows_imported']}")
    print(f"- latest imported date: {result['latest_date']}")


if __name__ == "__main__":
    main()
