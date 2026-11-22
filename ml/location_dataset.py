
"""Build location-level SCATS forecasting datasets.

Preserves each SCATS Number + VR Internal Loc series.
Sequences are assigned to splits by target timestamp, allowing validation
and test samples to use legitimate historical observations from earlier
periods.

This is a separate pipeline. It does not overwrite the existing site-level
datasets, models, or prediction interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ml.preprocess import (
    TRAFFIC_COLUMNS,
    clean_scats_data,
    load_scats_data,
)


BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR / "data" / "processed" / "location_level"
)

WINDOW_SIZES = (4, 8, 12)
INTERVAL = np.timedelta64(15, "m")

VALIDATION_START = pd.Timestamp("2006-10-22")
TEST_START = pd.Timestamp("2006-10-27")

LOCATION_COLUMN = "VR Internal Loc"


def _integer_column(series: pd.Series, name: str) -> pd.Series:
    """Convert a column to integers without silently rounding values."""

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().any():
        raise ValueError(f"{name} contains missing or invalid values.")

    values = numeric.to_numpy(dtype=np.float64)

    if not np.allclose(values, np.round(values)):
        raise ValueError(f"{name} contains non-integer values.")

    return pd.Series(
        np.round(values).astype(np.int32),
        index=series.index,
    )


def load_location_data() -> pd.DataFrame:
    """Load raw SCATS records and preserve individual physical locations."""

    raw = load_scats_data()
    clean = clean_scats_data(raw)

    required = [
        "SCATS Number",
        "Date",
        LOCATION_COLUMN,
        *TRAFFIC_COLUMNS,
    ]

    missing = [column for column in required if column not in clean.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean["SCATS Number"] = _integer_column(
        clean["SCATS Number"], "SCATS Number"
    )

    clean[LOCATION_COLUMN] = _integer_column(
        clean[LOCATION_COLUMN], LOCATION_COLUMN
    )

    # Keep useful physical-location metadata when available.
    optional_metadata = [
        "Location",
        "NB_LATITUDE",
        "NB_LONGITUDE",
        "CD_MELWAY",
    ]

    metadata_source = [
        column for column in optional_metadata
        if column in clean.columns
    ]

    id_columns = [
        "SCATS Number",
        "Date",
        LOCATION_COLUMN,
        *metadata_source,
    ]

    # Melt each original detector record independently.
    # Do not aggregate different VR Internal Loc values together.
    long_df = clean.melt(
        id_vars=id_columns,
        value_vars=TRAFFIC_COLUMNS,
        var_name="Interval",
        value_name="traffic_flow",
    )

    interval_numbers = long_df["Interval"].str.extract(
        r"^V(\d{2})$"
    )[0]

    if interval_numbers.isna().any():
        raise ValueError("Unexpected traffic interval column names.")

    long_df["timestamp"] = (
        long_df["Date"]
        + pd.to_timedelta(
            interval_numbers.astype(int) * 15,
            unit="minutes",
        )
    )

    long_df = long_df.rename(
        columns={
            "SCATS Number": "scats_id",
            LOCATION_COLUMN: "location_id",
            "Location": "location_name",
            "NB_LATITUDE": "latitude",
            "NB_LONGITUDE": "longitude",
            "CD_MELWAY": "cd_melway",
        }
    )

    metadata_fields = [
        {
            "Location": "location_name",
            "NB_LATITUDE": "latitude",
            "NB_LONGITUDE": "longitude",
            "CD_MELWAY": "cd_melway",
        }[column]
        for column in metadata_source
    ]

    long_df["scats_id"] = long_df["scats_id"].astype(np.int32)
    long_df["location_id"] = long_df["location_id"].astype(np.int32)

    output_columns = [
        "scats_id",
        "location_id",
        "timestamp",
        "traffic_flow",
        *metadata_fields,
    ]

    long_df = long_df[output_columns].copy()

    # A physical location must have no duplicate timestamp observations.
    duplicate_count = long_df.duplicated(
        ["scats_id", "location_id", "timestamp"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate location timestamps."
        )

    if long_df.isna().any().any():
        raise ValueError("Location-level dataset contains missing values.")

    if (long_df["traffic_flow"] < 0).any():
        raise ValueError("Negative traffic flow detected.")

    return long_df.sort_values(
        ["scats_id", "location_id", "timestamp"]
    ).reset_index(drop=True)


def create_location_sequences(
    df: pd.DataFrame,
    lookback: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Create sequences and assign them according to target timestamps."""

    if lookback not in WINDOW_SIZES:
        raise ValueError(f"lookback must be one of {WINDOW_SIZES}.")

    buckets = {
        name: {
            "X": [],
            "y": [],
            "scats_id": [],
            "location_id": [],
            "timestamp": [],
        }
        for name in ("train", "val", "test")
    }

    # A location is identified by its SCATS site and physical location ID.
    for (site_id, location_id), group in df.groupby(
        ["scats_id", "location_id"],
        sort=True,
    ):
        group = group.sort_values("timestamp")

        flows = group["scaled_flow"].to_numpy(dtype=np.float32)

        timestamps = group["timestamp"].to_numpy(
            dtype="datetime64[ns]"
        )

        for index in range(lookback, len(group)):
            # Include the target timestamp when checking continuity.
            window_times = timestamps[index - lookback:index + 1]

            gaps = np.diff(window_times)

            # Never create a sequence across missing 15-minute intervals.
            if not np.all(gaps == INTERVAL):
                continue

            target_time = pd.Timestamp(timestamps[index])

            # Assign by target time, not by the beginning of the input window.
            if target_time < VALIDATION_START:
                split_name = "train"
            elif target_time < TEST_START:
                split_name = "val"
            else:
                split_name = "test"

            bucket = buckets[split_name]

            bucket["X"].append(
                flows[index - lookback:index]
            )
            bucket["y"].append(flows[index])
            bucket["scats_id"].append(int(site_id))
            bucket["location_id"].append(int(location_id))
            bucket["timestamp"].append(target_time.isoformat())

    result = {}

    for split_name, bucket in buckets.items():
        if bucket["X"]:
            X = np.asarray(bucket["X"], dtype=np.float32)
            X = X.reshape(-1, lookback, 1)

            y = np.asarray(bucket["y"], dtype=np.float32)
        else:
            X = np.empty(
                (0, lookback, 1),
                dtype=np.float32,
            )
            y = np.empty((0,), dtype=np.float32)

        result[split_name] = {
            "X": X,
            "y": y,
            "scats_id": np.asarray(
                bucket["scats_id"], dtype=np.int32
            ),
            "location_id": np.asarray(
                bucket["location_id"], dtype=np.int32
            ),
            "timestamp": np.asarray(
                bucket["timestamp"], dtype=str
            ),
        }

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare location-level SCATS forecasting datasets."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate dataset sizes without writing files.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in the location-level output folder.",
    )

    args = parser.parse_args(argv)

    print("COS30019 Assignment 2B - Location-Level Dataset")
    print("=" * 65)

    df = load_location_data()

    location_count = df[
        ["scats_id", "location_id"]
    ].drop_duplicates().shape[0]

    print(f"Traffic observations: {len(df):,}")
    print(f"SCATS sites: {df['scats_id'].nunique()}")
    print(f"Physical location series: {location_count}")
    print("Location identity: SCATS Number + VR Internal Loc")

    # Fit the scaler using training-period observations only.
    train_mask = df["timestamp"] < VALIDATION_START

    if not train_mask.any():
        raise ValueError("No training-period observations were found.")

    scaler = MinMaxScaler()

    scaler.fit(
        df.loc[train_mask, ["traffic_flow"]]
    )

    df["scaled_flow"] = scaler.transform(
        df[["traffic_flow"]]
    ).ravel()

    print("\nChronological observation counts:")

    split_masks = {
        "train": df["timestamp"] < VALIDATION_START,
        "val": (
            (df["timestamp"] >= VALIDATION_START)
            & (df["timestamp"] < TEST_START)
        ),
        "test": df["timestamp"] >= TEST_START,
    }

    for name, mask in split_masks.items():
        print(f"  {name}: {int(mask.sum()):,}")

    print("\nScaler fitted using training observations only.")

    all_sequences = {}

    for lookback in WINDOW_SIZES:
        print(
            f"\nPreparing lookback={lookback} "
            f"({lookback * 15} minutes)"
        )

        sequences = create_location_sequences(df, lookback)

        for split_name, arrays in sequences.items():
            count = len(arrays["X"])

            print(
                f"  {split_name}: "
                f"{count:,} sequences | "
                f"X={arrays['X'].shape}"
            )

            if count == 0:
                raise ValueError(
                    f"No {split_name} sequences for lookback {lookback}."
                )

        all_sequences[lookback] = sequences

    if args.dry_run:
        print("\nDry run completed. No files were written.")
        return 0

    # Protect existing files unless the user explicitly allows overwriting.
    if (
        OUTPUT_DIR.exists()
        and any(OUTPUT_DIR.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"\nOutput folder already contains files: {OUTPUT_DIR}\n"
            "Review the existing files first. Use --overwrite only "
            "if you explicitly want to replace them."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the long-format location-level observations.
    df.to_csv(
        OUTPUT_DIR / "location_traffic_long.csv",
        index=False,
    )

    # Save one metadata record per physical location.
    metadata_fields = [
        column for column in (
            "location_name",
            "latitude",
            "longitude",
            "cd_melway",
        )
        if column in df.columns
    ]

    metadata = df[
        ["scats_id", "location_id", *metadata_fields]
    ].drop_duplicates()

    if metadata.duplicated(["scats_id", "location_id"]).any():
        raise ValueError(
            "Location metadata is inconsistent across observations."
        )

    metadata.to_csv(
        OUTPUT_DIR / "location_metadata.csv",
        index=False,
    )

    # Save a coverage summary for each physical location.
    summary = (
        df.groupby(
            ["scats_id", "location_id"],
            as_index=False,
        )
        .agg(
            observation_count=("traffic_flow", "size"),
            days_observed=(
                "timestamp",
                lambda values: values.dt.normalize().nunique(),
            ),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "location_summary.csv",
        index=False,
    )

    # Save the three splits for all lookback sizes.
    for lookback, sequences in all_sequences.items():
        for split_name, arrays in sequences.items():
            output_path = (
                OUTPUT_DIR
                / f"{split_name}_sequences_w{lookback}.npz"
            )

            np.savez_compressed(
                output_path,
                **arrays,
            )

    # Keep the same scaler parameter names used by the existing pipeline.
    np.savez(
        OUTPUT_DIR / "scaler_params.npz",
        data_min=scaler.data_min_,
        data_max=scaler.data_max_,
        scale=scaler.scale_,
        min_offset=scaler.min_,
    )

    print(f"\nLocation-level datasets saved to: {OUTPUT_DIR}")
    print("Dataset preparation completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
