import pandas as pd
from pathlib import Path

raw_path = Path(r"C:\Users\HP\Documents\EmpProd\data\raw")
processed_path = Path(r"C:\Users\HP\Documents\EmpProd\data\processed")
processed_path.mkdir(exist_ok=True)

files = [
    "Quebec Production Report_Chennai.csv",
    "TDM Hourly Production Report.csv"
]

for file in files:
    file_path = raw_path / file
    try:
        # Try reading as Excel
        df = pd.read_excel(file_path)
        print(f"📂 Read {file} as Excel ✅")
    except Exception:
        # Fallback: try reading as CSV
        df = pd.read_csv(file_path, encoding="utf-8", sep=",", on_bad_lines="skip")
        print(f"📂 Read {file} as CSV ✅")

    clean_path = processed_path / (file.replace(" ", "_").replace(".csv", "_clean.csv"))
    df.to_csv(clean_path, index=False)
    print(f"✅ Clean dataset saved at {clean_path} (rows={len(df)}, cols={len(df.columns)})")
    print("📊 Preview:")
    print(df.head(), "\n")
