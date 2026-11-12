"""Train LSTM, GRU, and BiLSTM for all required 15-minute lookbacks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import BiLSTMRegressor, GRURegressor

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
WINDOW_SIZES = (4, 8, 12)
MODEL_NAMES = ("LSTM", "GRU", "BiLSTM")
SEED = 42
EPOCHS = 20
BATCH_SIZE = 256


def _load_split(name: str, window_size: int) -> dict[str, np.ndarray]:
    path = DATA_DIR / f"{name}_sequences_w{window_size}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m ml.dataset` first."
        )
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    if not len(arrays["X"]):
        raise ValueError(f"{name} sequences for lookback {window_size} are empty.")
    return arrays


def _build_lstm(input_shape: tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _inverse_scale(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Invalid scaler: scale must be positive.")
    return (np.asarray(values).reshape(-1) - offset) / scale


def _predict(model_name: str, model: object, X: np.ndarray) -> np.ndarray:
    if model_name == "LSTM":
        return np.asarray(model.predict(X, batch_size=BATCH_SIZE, verbose=0)).reshape(-1)
    return np.asarray(model.predict(X, batch_size=BATCH_SIZE)).reshape(-1)


def _fit_one(model_name: str, window_size: int, epochs: int, batch_size: int) -> list[dict[str, float | int | str]]:
    tf.keras.utils.set_random_seed(SEED)
    train = _load_split("train", window_size)
    validation = _load_split("val", window_size)
    test = _load_split("test", window_size)
    with np.load(DATA_DIR / "scaler_params.npz") as params:
        scale = float(params["scale"][0])
        offset = float(params["min_offset"][0])

    if model_name == "LSTM":
        model = _build_lstm(tuple(train["X"].shape[1:]))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            )
        ]
        history = model.fit(
            train["X"], train["y"],
            validation_data=(validation["X"], validation["y"]),
            epochs=epochs, batch_size=batch_size, shuffle=False,
            callbacks=callbacks, verbose=1,
        )
        history_values = history.history
    else:
        model = GRURegressor(random_state=SEED) if model_name == "GRU" else BiLSTMRegressor(random_state=SEED)
        model.fit(
            train["X"], train["y"],
            validation_data=(validation["X"], validation["y"]),
            epochs=epochs, batch_size=batch_size, patience=4, verbose=1,
        )
        history_values = getattr(getattr(model, "_model", None), "history", None)
        history_values = history_values.history if history_values is not None else {}

    model_folder = MODEL_DIR / model_name.lower()
    model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder / f"window_{window_size}.keras"
    model.save(model_path)
    if model_name == "LSTM" and window_size == 8:
        model.save(MODEL_DIR / "lstm" / "lstm_model.keras")

    if history_values:
        pd.DataFrame(history_values).to_csv(
            model_folder / f"training_history_w{window_size}.csv", index=False
        )

    output_rows: list[dict[str, float | int | str]] = []
    for split_name, split in (("Validation", validation), ("Test", test)):
        predicted_scaled = _predict(model_name, model, split["X"])
        actual = _inverse_scale(split["y"], scale, offset)
        predicted = _inverse_scale(predicted_scaled, scale, offset)
        # Traffic flow cannot be negative; use the same constraint at evaluation and inference.
        predicted = np.maximum(predicted, 0.0)
        mse = float(mean_squared_error(actual, predicted))
        rmse = float(np.sqrt(mse))
        value_range = float(np.max(actual) - np.min(actual))
        nrmse = rmse / value_range if value_range > 0 else float("nan")
        row = {
            "model": model_name,
            "lookback_steps": window_size,
            "lookback_minutes": window_size * 15,
            "split": split_name,
            "MAE": float(mean_absolute_error(actual, predicted)),
            "MSE": mse,
            "RMSE": rmse,
            "R2": float(r2_score(actual, predicted)),
            "NRMSE": nrmse,
        }
        output_rows.append(row)
        if split_name == "Test":
            prediction_frame = pd.DataFrame(
                {
                    "scats_id": split["scats_id"],
                    "timestamp": split["timestamp"].astype(str),
                    "actual_vehicles_per_15min": actual,
                    "predicted_vehicles_per_15min": predicted,
                }
            )
            prediction_frame.to_csv(
                model_folder / f"test_predictions_w{window_size}.csv", index=False
            )
        print(
            f"{model_name} lookback={window_size} {split_name}: "
            f"RMSE={row['RMSE']:.3f}, MAE={row['MAE']:.3f}, "
            f"MSE={row['MSE']:.3f}, R2={row['R2']:.4f}"
        )
    print(f"Saved model: {model_path}")
    return output_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--lookbacks", nargs="+", type=int, choices=WINDOW_SIZES, default=list(WINDOW_SIZES))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    all_results = []
    for model_name in args.models:
        for window_size in args.lookbacks:
            all_results.extend(
                _fit_one(model_name, window_size, args.epochs, args.batch_size)
            )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output = MODEL_DIR / "evaluation_results.csv"
    pd.DataFrame(all_results).to_csv(output, index=False)
    print(f"Consolidated evaluation saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
