import pandas as pd
import requests
from pathlib import Path

OUT = Path("data/external/oni_monthly.csv")
ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"

MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    text = requests.get(ONI_URL, timeout=30).text

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip header-ish lines
        if line.lower().startswith(("yr", "year")):
            continue

        parts = line.split()
        # Expect: YEAR + 12 values (sometimes fewer; we'll handle)
        if len(parts) < 2:
            continue

        # First token should be year
        try:
            year = int(parts[0])
        except:
            continue

        vals = parts[1:]
        for i, v in enumerate(vals[:12], start=1):
            try:
                oni = float(v)
            except:
                continue
            rows.append({"date": pd.Timestamp(year=year, month=i, day=1), "oni": oni})

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(OUT, index=False)

    print("✅ Saved:", OUT)
    print("Rows:", len(df))
    print("Range:", df["date"].min().date(), "→", df["date"].max().date())
    print(df.tail(5).to_string(index=False))

if __name__ == "__main__":
    main()
