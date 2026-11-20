
"""Location-level traffic prediction using trained global ML models.

Default model:
    Random Forest, 12-step lookback.

Input identity:
    SCATS Number + VR Internal Loc

Prediction:
    One 15-minute traffic-flow value.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed" / "location_level"

DATA_PATH = DATA_DIR / "location_traffic_long.csv"
METADATA_PATH = DATA_DIR / "location_metadata.csv"
SCALER_PATH = DATA_DIR / "scaler_params.npz"

DEFAULT_MODEL_RUN = "location_rf_full_20261119"
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "location_level"
    / DEFAULT_MODEL_RUN
    / "randomforest"
    / "window_12.joblib"
)

MODEL_NAME = "RandomForest"
LOOKBACK = 12
INTERVAL_MINUTES = 15


@dataclass(frozen=True)
class LocationTrafficPrediction:
    """Prediction result for one physical detector location."""

    scats_id: int
    location_id: int
    predicted_flow: float
    unit: str
    model_name: str
    lookback: int
    prediction_time: str
    history_start: str
    history_end: str


def _validate_integer(value, field_name: str) -> int:
    """Reject invalid or non-integral identifiers."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be an integer.")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be an integer."
        ) from exc

    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field_name} must be an integer.")

    return int(number)


@lru_cache(maxsize=1)
def _load_traffic_data() -> pd.DataFrame:
    """Load individual detector traffic observations."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Location-level traffic data not found: {DATA_PATH}"
        )

    frame = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    required = {
        "scats_id",
        "location_id",
        "timestamp",
        "traffic_flow",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Traffic data is missing columns: {sorted(missing)}"
        )

    frame["scats_id"] = pd.to_numeric(
        frame["scats_id"],
        errors="raise",
    ).astype("int64")

    frame["location_id"] = pd.to_numeric(
        frame["location_id"],
        errors="raise",
    ).astype("int64")

    frame["traffic_flow"] = pd.to_numeric(
        frame["traffic_flow"],
        errors="raise",
    )

    if frame[list(required)].isnull().any().any():
        raise ValueError(
            "Traffic data contains missing required values."
        )

    if not np.isfinite(
        frame["traffic_flow"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Traffic data contains non-finite flow values."
        )

    if (frame["traffic_flow"] < 0).any():
        raise ValueError(
            "Traffic data contains negative traffic counts."
        )

    duplicate_columns = [
        "scats_id",
        "location_id",
        "timestamp",
    ]

    if frame.duplicated(duplicate_columns).any():
        raise ValueError(
            "Duplicate SCATS/location/timestamp records detected."
        )

    return frame.sort_values(
        duplicate_columns
    ).reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_valid_locations() -> frozenset[tuple[int, int]]:
    """Load valid SCATS and physical-location combinations."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Location metadata not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    required = {"scats_id", "location_id"}
    missing = required.difference(metadata.columns)

    if missing:
        raise ValueError(
            f"Location metadata is missing columns: {sorted(missing)}"
        )

    if metadata[list(required)].isnull().any().any():
        raise ValueError(
            "Location metadata contains missing identifiers."
        )

    pairs = frozenset(
        (
            int(row.scats_id),
            int(row.location_id),
        )
        for row in metadata[
            ["scats_id", "location_id"]
        ].itertuples(index=False)
    )

    return pairs


@lru_cache(maxsize=1)
def _load_scaler() -> tuple[float, float]:
    """Load the scaler fitted using training observations."""

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler parameters not found: {SCALER_PATH}"
        )

    with np.load(
        SCALER_PATH,
        allow_pickle=False,
    ) as params:

        required = {"scale", "min_offset"}
        missing = required.difference(params.files)

        if missing:
            raise ValueError(
                f"Scaler is missing parameters: {sorted(missing)}"
            )

        scale = float(params["scale"][0])
        offset = float(params["min_offset"][0])

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid scaler scale.")

    if not np.isfinite(offset):
        raise ValueError("Invalid scaler offset.")

    return scale, offset


@lru_cache(maxsize=1)
def _load_model():
    """Load the selected location-level Random Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Location-level model not found: {MODEL_PATH}"
        )

    from models import RandomForestTrafficRegressor

    return RandomForestTrafficRegressor.load(MODEL_PATH)


def _validate_prediction_time(
    prediction_time: str | pd.Timestamp,
) -> pd.Timestamp:
    """Validate a timezone-naive 15-minute prediction boundary."""

    target = pd.Timestamp(prediction_time)

    if pd.isna(target):
        raise ValueError("Invalid prediction timestamp.")

    if target.tzinfo is not None:
        raise ValueError(
            "Use a timezone-naive prediction timestamp."
        )

    if (
        target.minute % INTERVAL_MINUTES != 0
        or target.second != 0
        or target.microsecond != 0
        or target.nanosecond != 0
    ):
        raise ValueError(
            "Prediction time must be on a 15-minute boundary."
        )

    return target


def predict_location_flow(
    scats_id: int,
    location_id: int,
    prediction_time: str | pd.Timestamp,
) -> LocationTrafficPrediction:
    """Predict the next 15-minute flow for one physical detector.

    The 12 observations immediately before prediction_time must
    exist and be consecutive 15-minute records.
    """

    site_id = _validate_integer(
        scats_id,
        "scats_id",
    )

    physical_location_id = _validate_integer(
        location_id,
        "location_id",
    )

    target_time = _validate_prediction_time(
        prediction_time
    )

    valid_locations = _load_valid_locations()

    if (site_id, physical_location_id) not in valid_locations:
        raise ValueError(
            f"Unknown physical detector: SCATS {site_id}, "
            f"location {physical_location_id}."
        )

    data = _load_traffic_data()

    history = data.loc[
        (data["scats_id"] == site_id)
        & (data["location_id"] == physical_location_id)
        & (data["timestamp"] < target_time)
    ].tail(LOOKBACK)

    if len(history) != LOOKBACK:
        raise ValueError(
            f"Insufficient history for SCATS {site_id}, "
            f"location {physical_location_id}. "
            f"Required {LOOKBACK} observations, found {len(history)}."
        )

    expected_timestamps = pd.date_range(
        end=target_time - pd.Timedelta(minutes=INTERVAL_MINUTES),
        periods=LOOKBACK,
        freq="15min",
    )

    observed_timestamps = history["timestamp"].to_numpy(
        dtype="datetime64[ns]"
    )

    if not np.array_equal(
        observed_timestamps,
        expected_timestamps.to_numpy(dtype="datetime64[ns]"),
    ):
        raise ValueError(
            "Historical observations must be consecutive "
            "15-minute intervals. Missing timestamps detected."
        )

    scale, offset = _load_scaler()

    raw_history = history["traffic_flow"].to_numpy(
        dtype=np.float32
    )

    scaled_history = (
        raw_history * scale + offset
    )

    X = scaled_history.reshape(
        1,
        LOOKBACK,
        1,
    ).astype(np.float32)

    model = _load_model()

    predicted_scaled = float(
        np.asarray(
            model.predict(X)
        ).reshape(-1)[0]
    )

    if not np.isfinite(predicted_scaled):
        raise ValueError(
            "Model returned a non-finite prediction."
        )

    predicted_flow = (
        predicted_scaled - offset
    ) / scale

    # Traffic flow cannot be negative.
    predicted_flow = max(0.0, float(predicted_flow))

    return LocationTrafficPrediction(
        scats_id=site_id,
        location_id=physical_location_id,
        predicted_flow=predicted_flow,
        unit="vehicles_per_15_minutes",
        model_name=MODEL_NAME,
        lookback=LOOKBACK,
        prediction_time=target_time.isoformat(),
        history_start=history.iloc[0]["timestamp"].isoformat(),
        history_end=history.iloc[-1]["timestamp"].isoformat(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict traffic flow for one physical SCATS detector."
    )

    parser.add_argument(
        "--scats-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--location-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--prediction-time",
        required=True,
        help='Example: "2006-10-27 08:00:00"',
    )

    args = parser.parse_args()

    prediction = predict_location_flow(
        scats_id=args.scats_id,
        location_id=args.location_id,
        prediction_time=args.prediction_time,
    )

    print("=" * 65)
    print("LOCATION-LEVEL TRAFFIC PREDICTION")
    print("=" * 65)

    for key, value in asdict(prediction).items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
