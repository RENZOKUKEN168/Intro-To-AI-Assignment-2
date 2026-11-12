
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = (
    BASE_DIR / "data" / "raw"
    / "Scats Data October 2006.xls"
)

PROCESSED_PATH = (
    BASE_DIR / "data" / "processed"
    / "scats_traffic_long.csv"
)


def main():
    print("COS30019 Assignment 2B - Data Completeness")
    print("=" * 55)

    raw = pd.read_excel(
        RAW_PATH,
        sheet_name="Data",
        header=1
    )

    processed = pd.read_csv(
        PROCESSED_PATH,
        parse_dates=["timestamp"]
    )

    raw["Date"] = pd.to_datetime(raw["Date"]).dt.normalize()

    # Check complete daily coverage
    expected_dates = pd.date_range(
        "2006-10-01",
        "2006-10-31",
        freq="D"
    )

    print("\n1. Missing Dates by SCATS Site:")

    for site, group in raw.groupby("SCATS Number"):
        actual_dates = pd.DatetimeIndex(
            group["Date"].unique()
        )

        missing = expected_dates.difference(actual_dates)

        if len(missing) > 0:
            print(
                f"Site {site}: {len(missing)} missing days "
                f"- {missing.strftime('%Y-%m-%d').tolist()}"
            )

    # Check channel completeness
    print("\n2. Incomplete Site-Date Channels:")

    channel_cols = [
        "HF VicRoads Internal",
        "VR Internal Stat",
        "VR Internal Loc",
        "NB_TYPE_SURVEY"
    ]

    channel_counts = (
        raw.groupby("SCATS Number")[channel_cols]
        .apply(lambda x: len(x.drop_duplicates()))
    )

    daily_counts = (
        raw.groupby(["SCATS Number", "Date"])
        .size()
    )

    incomplete = []

    for (site, date), count in daily_counts.items():
        expected = channel_counts.loc[site]

        if count < expected:
            incomplete.append(
                (site, date, count, expected)
            )

    print(f"Incomplete site-date groups: {len(incomplete)}")

    for site, date, count, expected in incomplete[:20]:
        print(
            f"Site {site} | {date.date()} | "
            f"{count}/{expected} channels"
        )

    # Check time-series ordering
    print("\n3. Processed Data Checks:")

    print(
        "Duplicate timestamps per site:",
        processed.duplicated(
            ["scats_id", "timestamp"]
        ).sum()
    )

    print(
        "Negative traffic flow:",
        (processed["traffic_flow"] < 0).sum()
    )

    print(
        "Missing traffic flow:",
        processed["traffic_flow"].isna().sum()
    )

    print("\nCompleteness inspection completed.")


if __name__ == "__main__":
    main()
