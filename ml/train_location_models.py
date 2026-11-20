
"""Train global location-level traffic forecasting models.

Models:
    LSTM, GRU, BiLSTM, RandomForest

Datasets:
    data/processed/location_level/

All outputs are saved to a unique directory under:
    models/location_level/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from models import (
    BiLSTMRegressor,
    GRURegressor,
    RandomForestTrafficRegressor,
)


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed" / "location_level"
OUTPUT_ROOT = BASE_DIR / "models" / "location_level"

WINDOW_SIZES = (4, 8, 12)

MODEL_NAMES = (
    "LSTM",
    "GRU",
    "BiLSTM",
    "RandomForest",
)

SEED = 42

EPOCHS = 20
BATCH_SIZE = 256
PATIENCE = 4

RF_N_ESTIMATORS = 50
RF_MAX_DEPTH = 16
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = 1.0


def _load_split(
    split_name: str,
    window_size: int,
) -> dict[str, np.ndarray]:
    """Load and validate one location-level dataset split."""

    path = DATA_DIR / f"{split_name}_sequences_w{window_size}.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path}"
        )

    required = {
        "X",
        "y",
        "scats_id",
        "location_id",
        "timestamp",
    }

    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)

        if missing:
            raise ValueError(
                f"{path.name} is missing arrays: {sorted(missing)}"
            )

        arrays = {
            key: data[key]
            for key in required
        }

    arrays["y"] = arrays["y"].reshape(-1)
    arrays["scats_id"] = arrays["scats_id"].reshape(-1)
    arrays["location_id"] = arrays["location_id"].reshape(-1)
    arrays["timestamp"] = arrays["timestamp"].reshape(-1)

    X = arrays["X"]
    sample_count = len(X)

    if X.ndim != 3:
        raise ValueError(
            f"{path.name}: X must have 3 dimensions. Got {X.shape}"
        )

    if X.shape[1:] != (window_size, 1):
        raise ValueError(
            f"{path.name}: unexpected input shape {X.shape}"
        )

    for key in required:
        if len(arrays[key]) != sample_count:
            raise ValueError(
                f"{path.name}: array {key} has a different sample count."
            )

    if sample_count == 0:
        raise ValueError(f"{path.name} contains no samples.")

    if not np.isfinite(X).all() or not np.isfinite(arrays["y"]).all():
        raise ValueError(
            f"{path.name} contains NaN or infinite values."
        )

    return arrays


def _load_scaler() -> tuple[float, float]:
    """Load the training-fitted MinMaxScaler parameters."""

    path = DATA_DIR / "scaler_params.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing scaler parameters: {path}"
        )

    with np.load(path, allow_pickle=False) as params:
        scale = float(params["scale"][0])
        offset = float(params["min_offset"][0])

    if scale <= 0:
        raise ValueError("Scaler must have a positive scale.")

    return scale, offset


def _inverse_scale(
    values: np.ndarray,
    scale: float,
    offset: float,
) -> np.ndarray:
    """Convert predictions back to vehicles per 15 minutes."""

    return (
        np.asarray(values).reshape(-1) - offset
    ) / scale


def _build_lstm(
    input_shape: tuple[int, int],
) -> tf.keras.Model:
    """Build the LSTM architecture used in the existing trainer."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),

            tf.keras.layers.LSTM(
                64,
                return_sequences=True,
            ),

            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.LSTM(32),

            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),

            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def _predict(
    model_name: str,
    model: object,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Generate predictions using the appropriate model interface."""

    if model_name == "LSTM":
        predictions = model.predict(
            X,
            batch_size=batch_size,
            verbose=0,
        )

    elif model_name in ("GRU", "BiLSTM"):
        predictions = model.predict(
            X,
            batch_size=batch_size,
        )

    else:
        predictions = model.predict(X)

    return np.asarray(predictions).reshape(-1)


def _calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics in original traffic units."""

    mse = float(mean_squared_error(actual, predicted))
    rmse = float(np.sqrt(mse))

    value_range = float(np.max(actual) - np.min(actual))

    nrmse = (
        rmse / value_range
        if value_range > 0
        else float("nan")
    )

    r2 = (
        float(r2_score(actual, predicted))
        if len(actual) > 1 and np.unique(actual).size > 1
        else float("nan")
    )

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "NRMSE": nrmse,
    }


def _evaluate_split(
    model_name: str,
    model: object,
    split_name: str,
    split: dict[str, np.ndarray],
    window_size: int,
    scale: float,
    offset: float,
    batch_size: int,
    model_folder: Path,
) -> tuple[dict, list[dict]]:
    """Evaluate a split overall and separately for each location."""

    predicted_scaled = _predict(
        model_name,
        model,
        split["X"],
        batch_size,
    )

    actual = _inverse_scale(
        split["y"],
        scale,
        offset,
    )

    predicted = _inverse_scale(
        predicted_scaled,
        scale,
        offset,
    )

    # Traffic flow cannot be negative.
    predicted = np.maximum(predicted, 0.0)

    overall_metrics = _calculate_metrics(
        actual,
        predicted,
    )

    overall_row = {
        "model": model_name,
        "lookback_steps": window_size,
        "lookback_minutes": window_size * 15,
        "split": split_name,
        "n_samples": len(actual),
        **overall_metrics,
    }

    prediction_frame = pd.DataFrame(
        {
            "scats_id": split["scats_id"],
            "location_id": split["location_id"],
            "timestamp": split["timestamp"].astype(str),
            "actual_vehicles_per_15min": actual,
            "predicted_vehicles_per_15min": predicted,
        }
    )

    prediction_path = (
        model_folder
        / f"{split_name.lower()}_predictions_w{window_size}.csv"
    )

    prediction_frame.to_csv(
        prediction_path,
        index=False,
    )

    location_rows = []

    for (site_id, location_id), group in prediction_frame.groupby(
        ["scats_id", "location_id"],
        sort=True,
    ):
        location_actual = group[
            "actual_vehicles_per_15min"
        ].to_numpy()

        location_predicted = group[
            "predicted_vehicles_per_15min"
        ].to_numpy()

        metrics = _calculate_metrics(
            location_actual,
            location_predicted,
        )

        location_rows.append(
            {
                "model": model_name,
                "lookback_steps": window_size,
                "lookback_minutes": window_size * 15,
                "split": split_name,
                "scats_id": int(site_id),
                "location_id": int(location_id),
                "n_samples": len(group),
                **metrics,
            }
        )

    print(
        f"{model_name} lookback={window_size} {split_name}: "
        f"RMSE={overall_row['RMSE']:.3f}, "
        f"MAE={overall_row['MAE']:.3f}, "
        f"R2={overall_row['R2']:.4f}"
    )

    return overall_row, location_rows


def _train_one(
    model_name: str,
    window_size: int,
    datasets: dict[str, dict[str, np.ndarray]],
    scale: float,
    offset: float,
    run_dir: Path,
    epochs: int,
    batch_size: int,
    rf_n_estimators: int,
    rf_max_depth: int,
    rf_n_jobs: int,
) -> tuple[list[dict], list[dict]]:
    """Train, save and evaluate one model."""

    train = datasets["train"]
    validation = datasets["val"]
    test = datasets["test"]

    model_folder = run_dir / model_name.lower()
    model_folder.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(
        f"TRAINING {model_name} | LOOKBACK {window_size}"
    )
    print("=" * 65)

    history_values = None

    if model_name == "LSTM":
        tf.keras.utils.set_random_seed(SEED)

        model = _build_lstm(
            tuple(train["X"].shape[1:])
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
            )
        ]

        history = model.fit(
            train["X"],
            train["y"],
            validation_data=(
                validation["X"],
                validation["y"],
            ),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=callbacks,
            verbose=1,
        )

        history_values = history.history

    elif model_name == "GRU":
        tf.keras.utils.set_random_seed(SEED)

        model = GRURegressor(
            random_state=SEED
        )

        model.fit(
            train["X"],
            train["y"],
            validation_data=(
                validation["X"],
                validation["y"],
            ),
            epochs=epochs,
            batch_size=batch_size,
            patience=PATIENCE,
            verbose=1,
        )

        history_object = getattr(
            getattr(model, "_model", None),
            "history",
            None,
        )

        history_values = getattr(
            history_object,
            "history",
            None,
        )

    elif model_name == "BiLSTM":
        tf.keras.utils.set_random_seed(SEED)

        model = BiLSTMRegressor(
            random_state=SEED
        )

        model.fit(
            train["X"],
            train["y"],
            validation_data=(
                validation["X"],
                validation["y"],
            ),
            epochs=epochs,
            batch_size=batch_size,
            patience=PATIENCE,
            verbose=1,
        )

        history_object = getattr(
            getattr(model, "_model", None),
            "history",
            None,
        )

        history_values = getattr(
            history_object,
            "history",
            None,
        )

    else:
        model = RandomForestTrafficRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            random_state=SEED,
            n_jobs=rf_n_jobs,
        )

        model.fit(
            train["X"],
            train["y"],
        )

    # Save each model in its own algorithm directory.
    extension = (
        ".joblib"
        if model_name == "RandomForest"
        else ".keras"
    )

    model_path = (
        model_folder / f"window_{window_size}{extension}"
    )

    model.save(model_path)

    print(f"Saved model: {model_path}")

    if history_values:
        history_path = (
            model_folder
            / f"training_history_w{window_size}.csv"
        )

        pd.DataFrame(history_values).to_csv(
            history_path,
            index=False,
        )

    overall_rows = []
    location_rows = []

    for split_name, split in (
        ("Validation", validation),
        ("Test", test),
    ):
        overall, per_location = _evaluate_split(
            model_name=model_name,
            model=model,
            split_name=split_name,
            split=split,
            window_size=window_size,
            scale=scale,
            offset=offset,
            batch_size=batch_size,
            model_folder=model_folder,
        )

        overall_rows.append(overall)
        location_rows.extend(per_location)

    return overall_rows, location_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )

    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        choices=WINDOW_SIZES,
        default=list(WINDOW_SIZES),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=RF_N_ESTIMATORS,
    )

    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=RF_MAX_DEPTH,
    )

    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Unique name for this training run.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate datasets without training or saving.",
    )

    args = parser.parse_args(argv)

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")

    if args.rf_n_estimators < 1 or args.rf_max_depth < 1:
        parser.error("Random Forest settings must be positive.")

    run_name = args.run_name or datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )

    if (
        Path(run_name).name != run_name
        or run_name in {".", ".."}
    ):
        parser.error("--run-name must be a single directory name.")

    print("=" * 65)
    print("COS30019 LOCATION-LEVEL MODEL TRAINING")
    print("=" * 65)

    print(f"Models: {args.models}")
    print(f"Lookbacks: {args.lookbacks}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # Load and validate all required datasets before training.
    datasets = {}

    for window_size in args.lookbacks:
        print(
            f"\nChecking lookback {window_size}..."
        )

        datasets[window_size] = {}

        for split_name in ("train", "val", "test"):
            split = _load_split(
                split_name,
                window_size,
            )

            datasets[window_size][split_name] = split

            print(
                f"  {split_name}: "
                f"{split['X'].shape}"
            )

    scale, offset = _load_scaler()

    print(
        f"\nScaler: scale={scale:.8f}, offset={offset:.8f}"
    )

    if args.dry_run:
        print("\nDry run successful.")
        print("All selected datasets passed validation.")
        print("No models were trained and no files were written.")
        return 0

    # Every run receives a separate directory.
    run_dir = OUTPUT_ROOT / run_name

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dir.mkdir(
        exist_ok=False
    )

    manifest = {
        "run_name": run_name,
        "models": args.models,
        "lookbacks": args.lookbacks,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "random_seed": SEED,
        "dataset": "data/processed/location_level",
        "training_design": "Global models using pooled physical location series",
        "scaler": {
            "scale": scale,
            "offset": offset,
        },
    }

    with open(
        run_dir / "run_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    all_overall_rows = []
    all_location_rows = []

    for model_name in args.models:
        for window_size in args.lookbacks:
            overall_rows, location_rows = _train_one(
                model_name=model_name,
                window_size=window_size,
                datasets=datasets[window_size],
                scale=scale,
                offset=offset,
                run_dir=run_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                rf_n_estimators=args.rf_n_estimators,
                rf_max_depth=args.rf_max_depth,
                rf_n_jobs=args.rf_n_jobs,
            )

            all_overall_rows.extend(overall_rows)
            all_location_rows.extend(location_rows)

    pd.DataFrame(all_overall_rows).to_csv(
        run_dir / "evaluation_results.csv",
        index=False,
    )

    pd.DataFrame(all_location_rows).to_csv(
        run_dir / "per_location_evaluation.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("LOCATION-LEVEL TRAINING COMPLETED")
    print("=" * 65)

    print(f"Run directory: {run_dir}")

    print(
        f"Models trained: "
        f"{len(args.models) * len(args.lookbacks)}"
    )

    print(
        f"Overall evaluation: "
        f"{run_dir / 'evaluation_results.csv'}"
    )

    print(
        f"Per-location evaluation: "
        f"{run_dir / 'per_location_evaluation.csv'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
