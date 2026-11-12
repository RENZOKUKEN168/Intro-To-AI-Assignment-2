
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "lstm"

SEED = 42
EPOCHS = 20
BATCH_SIZE = 256

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_sequences(name):
    """Load a prepared sequence dataset."""

    path = DATA_DIR / f"{name}_sequences.npz"

    with np.load(path) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

    if len(X) == 0:
        raise ValueError(f"{name} dataset is empty.")

    return X, y


def inverse_scale(values, params):
    """Convert scaled values back to vehicle counts."""

    scale = float(params["scale"][0])
    offset = float(params["min_offset"][0])

    return (np.asarray(values).reshape(-1) - offset) / scale


def build_lstm_model(input_shape):
    """Create the LSTM neural network."""

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(32),

        tf.keras.layers.Dense(16, activation="relu"),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


def evaluate_model(model, X, y, params, name):
    """Evaluate predictions in original vehicle units."""

    prediction = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0
    ).reshape(-1)

    actual = inverse_scale(y, params)
    predicted = inverse_scale(prediction, params)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name} Evaluation:")
    print(f"MAE: {mae:.4f} vehicles/15 min")
    print(f"RMSE: {rmse:.4f} vehicles/15 min")
    print(f"R2 Score: {r2:.4f}")

    return {
        "dataset": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def main():
    print("COS30019 Assignment 2B - LSTM Training")
    print("=" * 55)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_sequences("train")
    X_val, y_val = load_sequences("val")
    X_test, y_test = load_sequences("test")

    params = np.load(DATA_DIR / "scaler_params.npz")

    print("\nTraining shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Testing shape:", X_test.shape)

    model = build_lstm_model(X_train.shape[1:])

    print("\nModel Architecture:")
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    print("\nStarting LSTM Training...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )

    model_path = MODEL_DIR / "lstm_model.keras"
    model.save(model_path)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        MODEL_DIR / "training_history.csv",
        index=False
    )

    results = [
        evaluate_model(
            model, X_val, y_val, params, "Validation"
        ),
        evaluate_model(
            model, X_test, y_test, params, "Testing"
        )
    ]

    pd.DataFrame(results).to_csv(
        MODEL_DIR / "evaluation_results.csv",
        index=False
    )

    print(f"\nModel saved: {model_path}")
    print("LSTM Training completed.")


if __name__ == "__main__":
    main()
