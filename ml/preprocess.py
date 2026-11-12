
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = (
    BASE_DIR / "data" / "raw"
    / "Scats Data October 2006.xls"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

TRAFFIC_COLUMNS = [f"V{i:02d}" for i in range(96)]

CHANNEL_COLUMNS = [
    "HF VicRoads Internal",
    "VR Internal Stat",
    "VR Internal Loc",
    "NB_TYPE_SURVEY"
]


def load_scats_data():
    """Load and validate the SCATS Excel structure."""
    df = pd.read_excel(
        RAW_PATH,
        sheet_name="Data",
        header=1
    )

    required = (
        ["SCATS Number", "Date"]
        + CHANNEL_COLUMNS
        + TRAFFIC_COLUMNS
    )

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def clean_scats_data(df):
    """Validate identifiers, dates and traffic counts."""
    df = df.copy()

    df["SCATS Number"] = pd.to_numeric(
        df["SCATS Number"], errors="coerce"
    )

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    ).dt.normalize()

    df[TRAFFIC_COLUMNS] = df[TRAFFIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    if df["SCATS Number"].isna().any():
        raise ValueError("Invalid SCATS IDs.")

    if df["Date"].isna().any():
        raise ValueError("Invalid dates.")

    if df[CHANNEL_COLUMNS].isna().any().any():
        raise ValueError("Missing channel identifiers.")

    if df[TRAFFIC_COLUMNS].isna().any().any():
        raise ValueError("Missing or invalid traffic values.")

    if (df[TRAFFIC_COLUMNS] < 0).any().any():
        raise ValueError("Negative traffic values.")

    duplicate_keys = (
        ["SCATS Number", "Date"] + CHANNEL_COLUMNS
    )

    if df.duplicated(subset=duplicate_keys).any():
        raise ValueError("Duplicate site-date-channel records.")

    return df


def check_channel_completeness(df):
    """Identify site-days with incomplete channel coverage."""

    channel_keys = (
        ["SCATS Number"] + CHANNEL_COLUMNS
    )

    # Define each site's observed channel set.
    site_channels = df[channel_keys].drop_duplicates()

    expected_counts = (
        site_channels.groupby("SCATS Number")
        .size()
        .rename("expected_channels")
    )

    actual_counts = (
        df.groupby(["SCATS Number", "Date"])
        .size()
        .rename("actual_channels")
        .reset_index()
    )

    quality = actual_counts.merge(
        expected_counts,
        on="SCATS Number",
        how="left"
    )

    quality["is_complete"] = (
        quality["actual_channels"]
        == quality["expected_channels"]
    )

    return quality


def aggregate_complete_days(df, quality):
    """Aggregate traffic only for complete site-days."""

    complete_keys = quality.loc[
        quality["is_complete"],
        ["SCATS Number", "Date"]
    ]

    filtered = df.merge(
        complete_keys,
        on=["SCATS Number", "Date"],
        how="inner"
    )

    daily = (
        filtered.groupby(
            ["SCATS Number", "Date"],
            as_index=False
        )[TRAFFIC_COLUMNS]
        .sum()
    )

    return daily


def reshape_time_series(site_daily):
    """Reshape daily traffic to 15-minute records."""

    long_df = site_daily.melt(
        id_vars=["SCATS Number", "Date"],
        value_vars=TRAFFIC_COLUMNS,
        var_name="Interval",
        value_name="traffic_flow"
    )

    interval_index = (
        long_df["Interval"].str[1:].astype(int)
    )

    long_df["timestamp"] = (
        long_df["Date"]
        + pd.to_timedelta(
            interval_index * 15,
            unit="minutes"
        )
    )

    long_df = long_df.rename(
        columns={"SCATS Number": "scats_id"}
    )

    long_df = long_df[
        ["scats_id", "timestamp", "traffic_flow"]
    ]

    long_df = long_df.sort_values(
        ["scats_id", "timestamp"]
    ).reset_index(drop=True)

    return long_df


def main():
    print("COS30019 Assignment 2B - Clean Data Pipeline")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_scats_data()
    clean = clean_scats_data(raw)

    quality = check_channel_completeness(clean)

    complete_count = int(quality["is_complete"].sum())
    incomplete_count = int((~quality["is_complete"]).sum())

    print(f"\nRaw records: {len(raw)}")
    print(f"Validated records: {len(clean)}")
    print(f"Complete site-days: {complete_count}")
    print(f"Incomplete site-days: {incomplete_count}")

    daily = aggregate_complete_days(clean, quality)
    long_df = reshape_time_series(daily)

    # Save processed data
    data_path = OUTPUT_DIR / "scats_traffic_clean.csv"
    long_df.to_csv(data_path, index=False)

    # Save quality report
    quality_path = OUTPUT_DIR / "scats_data_quality.csv"
    quality.to_csv(quality_path, index=False)

    print(f"\nClean time-series records: {len(long_df)}")

    print("\nFirst 10 records:")
    print(long_df.head(10).to_string(index=False))

    print(f"\nClean dataset saved: {data_path}")
    print(f"Quality report saved: {quality_path}")

    print("\nData preprocessing completed.")


if __name__ == "__main__":
    main()
