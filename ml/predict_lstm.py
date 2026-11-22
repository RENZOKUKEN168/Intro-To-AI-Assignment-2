
"""Backward-compatible wrapper for LSTM traffic prediction."""

import math
from dataclasses import replace

from ml.predict_model import predict_traffic as _predict_traffic


def predict_traffic(scats_id, prediction_time, lookback=8):
    # Get original LSTM prediction
    result = _predict_traffic(
        scats_id,
        prediction_time,
        model_name="LSTM",
        lookback=lookback
    )

    # Round up traffic flow to a whole vehicle
    rounded_flow = math.ceil(max(0.0, result.predicted_flow))

    # Return a new result without modifying shared models
    return replace(result, predicted_flow=rounded_flow)


def main():
    result = predict_traffic(
        2000,
        "2006-10-27 08:00:00",
        lookback=8
    )

    print(f"SCATS Site: {result.scats_id}")
    print(f"Prediction Time: {result.prediction_time}")
    print(f"Model: {result.model_name}")
    print(f"Predicted Traffic Flow: {result.predicted_flow} vehicles/15 min")


if __name__ == "__main__":
    main()
