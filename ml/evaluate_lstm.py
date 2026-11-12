
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models" / "lstm"
DATA_DIR = BASE_DIR / "data" / "processed"

GRAPH_DIR = MODEL_DIR / "graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# Graph 1: Training vs Validation Loss
# =====================================

def plot_training_loss():

    history_path = MODEL_DIR / "training_history.csv"

    history = pd.read_csv(history_path)

    epochs = range(1, len(history) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        history["loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.title("LSTM Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (Scaled)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = GRAPH_DIR / "training_validation_loss.png"

    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Graph 1 saved: {output}")


# =====================================
# Graph 2: Actual vs Predicted Traffic
# =====================================

def plot_actual_vs_predicted():

    model_path = MODEL_DIR / "lstm_model.keras"

    model = tf.keras.models.load_model(model_path)

    with np.load(DATA_DIR / "test_sequences.npz") as data:
        X_test = data["X"]
        y_test = data["y"]
        sites = data["scats_id"]
        timestamps = pd.to_datetime(data["timestamp"])

    # Predict using the trained LSTM
    predictions = model.predict(
        X_test,
        batch_size=256,
        verbose=0
    ).reshape(-1)

    # Reverse MinMax scaling
    with np.load(DATA_DIR / "scaler_params.npz") as params:
        scale = float(params["scale"][0])
        offset = float(params["min_offset"][0])

    actual = (y_test - offset) / scale
    predicted = (predictions - offset) / scale

    # Use a single SCATS site and consecutive time period
    result = pd.DataFrame({
        "scats_id": sites,
        "timestamp": timestamps,
        "actual": actual,
        "predicted": predicted
    })

    result = result.sort_values(
        ["scats_id", "timestamp"]
    )

    selected = None

    for site, group in result.groupby("scats_id"):

        group = group.reset_index(drop=True)

        # Identify consecutive 15-minute timestamps
        segment_id = (
            group["timestamp"].diff()
            .ne(pd.Timedelta(minutes=15))
            .cumsum()
        )

        for _, segment in group.groupby(segment_id):
            if len(segment) >= 96:
                selected = segment.head(96).copy()
                break

        if selected is not None:
            break

    if selected is None:
        raise ValueError(
            "No continuous 96-interval test segment found."
        )

    selected_site = int(selected["scats_id"].iloc[0])

    plt.figure(figsize=(12, 5))

    plt.plot(
        selected["timestamp"],
        selected["actual"],
        label="Actual Traffic Flow",
        linewidth=2
    )

    plt.plot(
        selected["timestamp"],
        selected["predicted"],
        label="Predicted Traffic Flow",
        linewidth=2,
        linestyle="--"
    )

    plt.title(
        f"LSTM Actual vs Predicted Traffic - SCATS {selected_site}"
    )

    plt.xlabel("Date and Time")
    plt.ylabel("Vehicles per 15 Minutes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    graph_path = GRAPH_DIR / "actual_vs_predicted.png"

    plt.savefig(graph_path, dpi=300)
    plt.close()

    # Save actual and predicted values for report evidence
    csv_path = GRAPH_DIR / "actual_vs_predicted.csv"

    selected.to_csv(csv_path, index=False)

    print(f"Graph 2 saved: {graph_path}")
    print(f"Prediction data saved: {csv_path}")
    print(f"Selected SCATS Site: {selected_site}")


# =====================================
# Main
# =====================================

def main():

    print("COS30019 Assignment 2B - LSTM Evaluation")
    print("=" * 55)

    plot_training_loss()
    plot_actual_vs_predicted()

    print("\nBoth graphs generated successfully.")


if __name__ == "__main__":
    main()
