"""Backward-compatible wrapper for LSTM traffic prediction."""

from ml.predict_model import predict_traffic as _predict_traffic


def predict_traffic(scats_id, prediction_time, lookback=8):
    return _predict_traffic(
        scats_id, prediction_time, model_name="LSTM", lookback=lookback
    )


def main():
    result = predict_traffic(2000, "2006-10-27 08:00:00", lookback=8)
    print(f"SCATS {result.scats_id} at {result.prediction_time}")
    print(f"{result.predicted_flow:.2f} vehicles/15 min ({result.model_name})")


if __name__ == "__main__":
    main()
