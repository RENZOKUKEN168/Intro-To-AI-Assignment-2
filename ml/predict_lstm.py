
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

# Allow importing the existing TBRGS module
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from tbrgs.travel_time import TrafficPrediction

MODEL_PATH = BASE_DIR / "models" / "lstm" / "lstm_model.keras"

SCALER_PATH = (
    BASE_DIR / "data" / "processed" / "scaler_params.npz"
)

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "scats_traffic_clean.csv"
)

WINDOW_SIZE = 8


def predict_traffic(scats_id, prediction_time):
    """
    Predict traffic flow for a SCATS site using LSTM.

    Parameters:
        scats_id: SCATS site number
        prediction_time: Target timestamp

    Returns:
        TrafficPrediction object
    """

    # =====================================
    # 1. Load the trained LSTM model
    # =====================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError("LSTM model not found.")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # =====================================
    # 2. Load preprocessing parameters
    # =====================================

    with np.load(SCALER_PATH) as params:
        scale = float(params["scale"][0])
        offset = float(params["min_offset"][0])

    if scale <= 0:
        raise ValueError("Invalid scaling parameter.")

    # =====================================
    # 3. Load traffic data
    # =====================================

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    target_time = pd.Timestamp(prediction_time)

    if pd.isna(target_time) or target_time.tzinfo is not None:
        raise ValueError(
            "Use a valid timezone-naive timestamp."
        )

    if (
        target_time.second != 0
        or target_time.microsecond != 0
        or target_time.minute % 15 != 0
    ):
        raise ValueError(
            "Prediction time must be on a 15-minute boundary."
        )

    site_data = df[
        (df["scats_id"] == int(scats_id))
        & (df["timestamp"] < target_time)
    ].copy()

    site_data = site_data.sort_values("timestamp")

    # =====================================
    # 4. Get previous 8 traffic readings
    # =====================================

    history = site_data.tail(WINDOW_SIZE)

    if len(history) != WINDOW_SIZE:
        raise ValueError(
            "Not enough historical traffic data."
        )

    expected_times = pd.date_range(
        end=target_time - pd.Timedelta(minutes=15),
        periods=WINDOW_SIZE,
        freq="15min"
    )

    if not np.array_equal(
        history["timestamp"].to_numpy(
            dtype="datetime64[ns]"
        ),
        expected_times.to_numpy(
            dtype="datetime64[ns]"
        )
    ):
        raise ValueError(
            "Historical observations must be consecutive "
            "15-minute intervals ending before prediction time."
        )

    # =====================================
    # 5. Scale the historical observations
    # =====================================

    values = history["traffic_flow"].to_numpy(
        dtype=np.float32
    )

    scaled_values = values * scale + offset

    X = scaled_values.reshape(
        1, WINDOW_SIZE, 1
    ).astype(np.float32)

    # =====================================
    # 6. Predict traffic flow using LSTM
    # =====================================

    prediction_scaled = model.predict(
        X,
        verbose=0
    )[0][0]

    # Convert to original vehicle units
    prediction = (
        float(prediction_scaled) - offset
    ) / scale

    # Traffic flow cannot be negative
    prediction = max(0.0, prediction)

    # =====================================
    # 7. Return integration-compatible data
    # =====================================

    return TrafficPrediction(
        scats_id=int(scats_id),
        predicted_flow=prediction,
        unit="vehicles_per_15_minutes",
        model_name="LSTM",
        prediction_time=target_time.isoformat()
    )


# =====================================
# Main - Example prediction
# =====================================

def main():

    print("COS30019 Assignment 2B - LSTM Prediction")
    print("=" * 55)

    result = predict_traffic(
        scats_id=2000,
        prediction_time="2006-10-27 08:00:00"
    )

    print(f"\nSCATS Site: {result.scats_id}")
    print(f"Prediction Time: {result.prediction_time}")
    print(f"Model: {result.model_name}")

    print(
        f"Predicted Traffic Flow: "
        f"{result.predicted_flow:.2f} vehicles/15 min"
    )

    print("\nLSTM Prediction completed.")


if __name__ == "__main__":
    main()
