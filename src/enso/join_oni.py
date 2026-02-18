import pandas as pd
from pathlib import Path

INP = Path("data/processed/gatun_daily_clean.csv")
ONI = Path("data/external/oni_monthly.csv")
OUT = Path("data/processed/gatun_daily_with_oni.csv")

def main():
    print("Loading:", INP)
    if not INP.exists():
        raise FileNotFoundError(f"Missing input file: {INP}")

    print("Loading:", ONI)
    if not ONI.exists():
        raise FileNotFoundError(f"Missing ONI file: {ONI} (run src/enso/get_oni.py first)")

    df = pd.read_csv(INP, parse_dates=["date"]).sort_values("date")
    oni = pd.read_csv(ONI, parse_dates=["date"]).sort_values("date")

    # Attach monthly ONI to each day by month
    df["month_key"] = df["date"].dt.to_period("M").dt.to_timestamp()
    oni["month_key"] = oni["date"].dt.to_period("M").dt.to_timestamp()

    merged = df.merge(oni[["month_key", "oni"]], on="month_key", how="left")
    merged["oni"] = merged["oni"].ffill()
    merged.loc[merged["oni"] <= -90, "oni"] = pd.NA

# Forward-fill again after cleaning
    merged["oni"] = merged["oni"].ffill()   


    # ENSO features
    merged["oni_3mo"] = merged["oni"].rolling(3, min_periods=1).mean()
    merged["oni_trend_3mo"] = merged["oni"] - merged["oni"].shift(3)

    def phase(x):
        if pd.isna(x): return "unknown"
        if x >= 0.5: return "el_nino"
        if x <= -0.5: return "la_nina"
        return "neutral"

    merged["enso_phase"] = merged["oni"].apply(phase)

    merged = merged.drop(columns=["month_key"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)

    print("✅ Saved:", OUT)
    print("Rows:", len(merged))
    print(merged[["date", "oni", "enso_phase"]].tail(3).to_string(index=False))

if __name__ == "__main__":
    main()
