
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "Scats Data October 2006.xls"

def validate_scats_data():
    print("COS30019 Assignment 2B - Data Validation")
    print("=" * 55)

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="Data",
        header=1
    )

    traffic_cols = [f"V{i:02d}" for i in range(96)]

    # 1. Check SCATS records
    print("\n1. Records Per SCATS Site:")
    print(df.groupby("SCATS Number").size().to_string())

    # 2. Check multiple records per site and date
    print("\n2. Multiple Records Per Site and Date:")
    grouped = df.groupby(
        ["SCATS Number", "Date"]
    ).size()

    multiple = grouped[grouped > 1]
    print(f"Site-date groups with multiple records: {len(multiple)}")
    print(multiple.head(10).to_string())

    # 3. Validate numeric traffic values
    traffic = df[traffic_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    print("\n3. Invalid Traffic Values:")
    print(f"Non-numeric or missing: {traffic.isna().sum().sum()}")
    print(f"Negative values: {(traffic < 0).sum().sum()}")

    # 4. Traffic statistics
    values = traffic.stack()

    print("\n4. Traffic Statistics:")
    print(values.describe().to_string())

    # 5. Check date coverage
    print("\n5. Unique Dates Per SCATS Site:")
    print(
        df.groupby("SCATS Number")["Date"]
        .nunique().to_string()
    )

    print("\nData validation completed.")

if __name__ == "__main__":
    validate_scats_data()
