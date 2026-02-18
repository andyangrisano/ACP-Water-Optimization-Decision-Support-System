import pandas as pd

RAW_PATH = "data/raw/gatun_mvp_daily.csv"
OUT_PATH = "data/processed/gatun_daily_clean.csv"

df = pd.read_csv(RAW_PATH, header=4)
df.columns = [c.strip() for c in df.columns]

# Rename columns based on what your file actually contains
rename_map = {
    "Start of Interval (UTC-05:00)": "date",
    "Average (ft)": "lake_level_ft",
    "Total (mm)": "rainfall_mm",
    "Average (hm^3)": "evaporation_hm3",
}
df = df.rename(columns=rename_map)

# Keep only the columns we need (ignore "End of Interval")
df = df[["date", "lake_level_ft", "rainfall_mm", "evaporation_hm3"]].copy()

# Parse + index
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").drop_duplicates("date").set_index("date")

# Numeric conversion
for col in ["lake_level_ft", "rainfall_mm", "evaporation_hm3"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill small gaps, drop big missing areas
df = df.interpolate(limit=3).dropna()

df.to_csv(OUT_PATH)

print("✅ Saved:", OUT_PATH)
print("✅ Date range:", df.index.min(), "→", df.index.max())
print(df.tail())
