
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "scats_traffic_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

WINDOW_SIZE = 8  # Backward-compatible default
WINDOW_SIZES = (4, 8, 12)  # 1, 2, and 3 hours at 15-minute intervals
INTERVAL = pd.Timedelta(minutes=15)


def load_data():
    """Load and validate the cleaned traffic dataset."""

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    required = ["scats_id", "timestamp", "traffic_flow"]

    if not set(required).issubset(df.columns):
        raise ValueError("Required columns are missing.")

    if df[required].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if df.duplicated(["scats_id", "timestamp"]).any():
        raise ValueError("Duplicate site timestamps found.")

    if (df["traffic_flow"] < 0).any():
        raise ValueError("Negative traffic flow found.")

    return df.sort_values(
        ["scats_id", "timestamp"]
    ).reset_index(drop=True)


def split_data(df):
    """Split by calendar time, not randomly."""

    train = df[
        df["timestamp"] < "2006-10-22"
    ].copy()

    val = df[
        (df["timestamp"] >= "2006-10-22")
        & (df["timestamp"] < "2006-10-27")
    ].copy()

    test = df[
        df["timestamp"] >= "2006-10-27"
    ].copy()

    return train, val, test


def create_sequences(df, window_size=WINDOW_SIZE):
    """Create consecutive 15-minute windows and the immediately following target."""

    if window_size not in WINDOW_SIZES:
        raise ValueError(f"window_size must be one of {WINDOW_SIZES}.")

    X, y = [], []
    sites, timestamps = [], []

    for site, group in df.groupby("scats_id"):
        group = group.sort_values("timestamp")

        flows = group["scaled_flow"].to_numpy(
            dtype=np.float32
        )

        times = group["timestamp"].to_numpy(
            dtype="datetime64[ns]"
        )

        for i in range(window_size, len(group)):
            window_times = times[i-window_size:i+1]

            # Reject windows crossing missing time periods
            gaps = np.diff(window_times)

            if not np.all(
                gaps == np.timedelta64(15, "m")
            ):
                continue

            X.append(flows[i-window_size:i])
            y.append(flows[i])
            sites.append(int(site))
            timestamps.append(str(times[i]))

    if not X:
        return (
            np.empty((0, window_size, 1), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.array([], dtype=str)
        )

    X = np.asarray(X, dtype=np.float32)
    X = X.reshape(-1, window_size, 1)

    return (
        X,
        np.asarray(y, dtype=np.float32),
        np.asarray(sites, dtype=np.int32),
        np.asarray(timestamps, dtype=str)
    )


def main():
    print("COS30019 Assignment 2B - 15-Minute Sequence Preparation")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train, val, test = split_data(df)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("One of the chronological data splits is empty.")

    scaler = MinMaxScaler()
    train.loc[:, "scaled_flow"] = scaler.fit_transform(train[["traffic_flow"]]).ravel()
    val.loc[:, "scaled_flow"] = scaler.transform(val[["traffic_flow"]]).ravel()
    test.loc[:, "scaled_flow"] = scaler.transform(test[["traffic_flow"]]).ravel()

    print(f"Records: train={len(train)}, validation={len(val)}, test={len(test)}")
    print("Scaler fitted using training records only.")

    splits = (("train", train), ("val", val), ("test", test))
    for window_size in WINDOW_SIZES:
        print(f"\nPreparing lookback={window_size} (" f"{window_size * 15} minutes)")
        for name, split in splits:
            X, y, sites, timestamps = create_sequences(split, window_size)
            if not len(X):
                raise ValueError(f"{name} has no valid sequences for lookback {window_size}.")
            output = OUTPUT_DIR / f"{name}_sequences_w{window_size}.npz"
            np.savez_compressed(
                output, X=X, y=y, scats_id=sites, timestamp=timestamps
            )
            # Keep existing Member 1 scripts/tests compatible with their 8-step inputs.
            if window_size == WINDOW_SIZE:
                np.savez_compressed(
                    OUTPUT_DIR / f"{name}_sequences.npz",
                    X=X, y=y, scats_id=sites, timestamp=timestamps
                )
            print(f"  {name}: X={X.shape}, y={y.shape} -> {output.name}")

    np.savez(
        OUTPUT_DIR / "scaler_params.npz",
        data_min=scaler.data_min_, data_max=scaler.data_max_,
        scale=scaler.scale_, min_offset=scaler.min_
    )
    print("\nSequence preparation completed for all three lookbacks.")


if __name__ == "__main__":
    main()
