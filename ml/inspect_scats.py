
from pathlib import Path
import pandas as pd

# Locate the dataset from the project root
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "Scats Data October 2006.xls"


def inspect_scats_data():
    print("COS30019 Assignment 2B - SCATS Data Inspection")
    print("=" * 55)

    # Check if the dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Load the original SCATS data
    df = pd.read_excel(
    DATA_PATH,
    sheet_name="Data",
    header=1
    )

    # Display dataset information
    print("\n1. Dataset Shape:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n2. Column Names:")
    print(df.columns.tolist())

    print("\n3. First Five Records:")
    print(df.head().to_string())

    print("\n4. Data Types:")
    print(df.dtypes.to_string())

    print("\n5. Missing Values:")
    print(df.isnull().sum().to_string())

    print("\n6. Duplicate Rows:")
    print(df.duplicated().sum())

    # Inspect SCATS sites
    if "SCATS Number" in df.columns:
        print("\n7. Unique SCATS Sites:")
        print(df["SCATS Number"].nunique())

    # Inspect date range
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")

        print("\n8. Date Range:")
        print(f"Start: {dates.min()}")
        print(f"End: {dates.max()}")
        print(f"Invalid dates: {dates.isna().sum()}")

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    inspect_scats_data()
