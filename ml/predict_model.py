"""Predict one 15-minute traffic-flow step for an SCATS site.

LSTM, GRU and BiLSTM use Keras artifacts. Random Forest uses a joblib artifact
and deliberately does not require TensorFlow for inference.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "scats_traffic_clean.csv"
SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler_params.npz"
MODEL_DIR = BASE_DIR / "models"
WINDOW_SIZES = (4, 8, 12)
MODEL_NAMES = ("LSTM", "GRU", "BiLSTM", "RandomForest")


@lru_cache(maxsize=1)
def _load_traffic_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed traffic data not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    required = {"scats_id", "timestamp", "traffic_flow"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Processed traffic data is missing columns: {sorted(missing)}")
    return frame


@lru_cache(maxsize=12)
def _load_model(model_name: str, lookback: int):
    """Load a trained artifact, using joblib for Random Forest and Keras otherwise."""
    if model_name == "RandomForest":
        model_path = MODEL_DIR / "random_forest" / f"window_{lookback}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained Random Forest model for {lookback} steps not found at {model_path}. "
                "Run `python -m ml.train_random_forest` first."
            )
        # A Git LFS pointer is a tiny text file, not the model; joblib would fail
        # with an unhelpful KeyError, so explain the fix instead.
        with model_path.open("rb") as handle:
            if handle.read(40).startswith(b"version https://git-lfs"):
                raise FileNotFoundError(
                    f"{model_path} is a Git LFS pointer, not the trained model. "
                    "Install Git LFS, then run `git lfs install` and `git lfs pull`."
                )
        from models import RandomForestTrafficRegressor

        return RandomForestTrafficRegressor.load(model_path)

    candidates = [MODEL_DIR / model_name.lower() / f"window_{lookback}.keras"]
    if model_name == "LSTM" and lookback == 8:
        candidates.append(MODEL_DIR / "lstm" / "lstm_model.keras")
    model_path = next((path for path in candidates if path.exists()), None)
    if model_path is None:
        expected = candidates[0]
        raise FileNotFoundError(
            f"Trained {model_name} model for {lookback} steps not found at {expected}. "
            "Run `python -m ml.train_all_models` first."
        )
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "LSTM/GRU/BiLSTM inference requires TensorFlow. Install a compatible "
            "TensorFlow version using requirements-gru.txt."
        ) from exc
    return tf.keras.models.load_model(model_path, compile=False)


def predict_traffic(
    scats_id: int,
    prediction_time: str | pd.Timestamp,
    *,
    model_name: str = "LSTM",
    lookback: int = 8,
):
    """Predict the flow count at ``prediction_time`` from prior 15-minute rows."""
    from tbrgs.travel_time import TrafficPrediction

    aliases = {name.lower(): name for name in MODEL_NAMES}
    aliases.update({"random_forest": "RandomForest", "rf": "RandomForest"})
    normalized_name = aliases.get(str(model_name).strip().lower())
    if normalized_name is None:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}.")
    if lookback not in WINDOW_SIZES:
        raise ValueError(f"lookback must be one of {WINDOW_SIZES}.")
    try:
        site_id = int(scats_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("scats_id must be an integer.") from exc

    target_time = pd.Timestamp(prediction_time)
    if pd.isna(target_time) or target_time.tzinfo is not None:
        raise ValueError("Use a valid timezone-naive prediction timestamp.")
    if target_time.second or target_time.microsecond or target_time.minute % 15:
        raise ValueError("Prediction time must be on a 15-minute boundary.")

    data = _load_traffic_data()
    site_data = data[(data["scats_id"] == site_id) & (data["timestamp"] < target_time)]
    site_data = site_data.sort_values("timestamp").tail(lookback)
    if len(site_data) != lookback:
        raise ValueError(f"Not enough history for SCATS site {site_id}.")
    expected = pd.date_range(
        end=target_time - pd.Timedelta(minutes=15), periods=lookback, freq="15min"
    )
    observed = site_data["timestamp"].to_numpy(dtype="datetime64[ns]")
    if not np.array_equal(observed, expected.to_numpy(dtype="datetime64[ns]")):
        raise ValueError("Historical data must be consecutive 15-minute intervals.")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler parameters not found: {SCALER_PATH}")
    with np.load(SCALER_PATH) as params:
        scale = float(params["scale"][0])
        offset = float(params["min_offset"][0])
    if scale <= 0:
        raise ValueError("Invalid scaler parameters.")

    history = site_data["traffic_flow"].to_numpy(dtype=np.float32)
    X = (history * scale + offset).reshape(1, lookback, 1).astype(np.float32)
    model = _load_model(normalized_name, lookback)
    if normalized_name == "RandomForest":
        predicted_scaled = float(np.asarray(model.predict(X)).reshape(-1)[0])
    else:
        predicted_scaled = float(np.asarray(model.predict(X, verbose=0)).reshape(-1)[0])
    predicted_flow = max(0.0, (predicted_scaled - offset) / scale)
    return TrafficPrediction(
        scats_id=site_id,
        predicted_flow=predicted_flow,
        unit="vehicles_per_15_minutes",
        model_name=normalized_name,
        prediction_time=target_time.isoformat(),
    )
