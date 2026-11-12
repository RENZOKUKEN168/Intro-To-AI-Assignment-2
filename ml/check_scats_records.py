
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR / "data" / "raw"
    / "Scats Data October 2006.xls"
)


def check_scats_records():

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="Data",
        header=1
    )

    # Select one SCATS site and date
    site = 970
    date = pd.Timestamp("2006-10-01")

    records = df[
        (df["SCATS Number"] == site)
        & (df["Date"].dt.normalize() == date)
    ]

    print("=" * 60)
    print("SCATS Record Structure Investigation")
    print("=" * 60)

    print("\n1. Records for SCATS 970 on 1 October:")
    print(f"Total records: {len(records)}")

    # Show the identifiers that may distinguish records
    metadata_cols = [
        "SCATS Number",
        "Location",
        "HF VicRoads Internal",
        "VR Internal Stat",
        "VR Internal Loc",
        "NB_TYPE_SURVEY",
        "Date"
    ]

    print("\n2. Record Metadata:")
    print(
        records[metadata_cols].to_string(index=False)
    )

    # Compare traffic values for each record
    traffic_cols = [f"V{i:02d}" for i in range(96)]

    print("\n3. Traffic Flow Comparison:")
    print(
        records[
            ["SCATS Number", "VR Internal Loc"]
            + traffic_cols[:8]
        ].to_string(index=False)
    )

    # Check whether metadata identifies unique channels
    channel_cols = [
        "HF VicRoads Internal",
        "VR Internal Stat",
        "VR Internal Loc",
        "NB_TYPE_SURVEY"
    ]

    print("\n4. Different Metadata Combinations:")
    channels = df[
        ["SCATS Number"] + channel_cols
    ].drop_duplicates()

    print(
        channels.groupby("SCATS Number")
        .size()
        .to_string()
    )

    print("\n5. Duplicate Site-Date-Channel Records:")
    duplicates = df.duplicated(
        subset=[
            "SCATS Number",
            "Date"
        ] + channel_cols
    )

    print(f"Duplicates: {duplicates.sum()}")

    print("\nInvestigation completed.")


if __name__ == "__main__":
    check_scats_records()
