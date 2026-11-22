"""Train Random Forest traffic-flow models on the shared SCATS sequences.

This script deliberately writes its artifacts and evaluation results under
models/random_forest/ so it does not overwrite the LSTM/GRU/BiLSTM outputs.
Run `python -m ml.dataset` first to prepare the shared chronological splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import RandomForestTrafficRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "random_forest"

WINDOW_SIZES = (4, 8, 12)
SEED = 42
N_ESTIMATORS = 50
MAX_DEPTH = 16


def _load_split(name: str, window_size: int) -> dict[str, np.ndarray]:
    """Load one preprocessed chronological split for a given lookback."""
    path = DATA_DIR / f"{name}_sequences_w{window_size}.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m ml.dataset` first."
        )

    with np.load(path) as data:
        required = {"X", "y", "scats_id", "timestamp"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"{path.name} is missing required arrays: {sorted(missing)}"
            )
        arrays = {key: data[key] for key in required}

    if not len(arrays["X"]):
        raise ValueError(
            f"{name} sequences for lookback {window_size} are empty."
        )

    if arrays["X"].ndim != 3:
        raise ValueError(
            f"{name} X must have shape (samples, timesteps, features); "
            f"got {arrays['X'].shape}."
        )

    sample_count = arrays["X"].shape[0]
    if arrays["y"].reshape(-1).shape[0] != sample_count:
        raise ValueError(f"{name} X and y contain different sample counts.")

    if len(arrays["scats_id"]) != sample_count:
        raise ValueError(f"{name} SCATS IDs do not match the sample count.")

    if len(arrays["timestamp"]) != sample_count:
        raise ValueError(f"{name} timestamps do not match the sample count.")

    if arrays["X"].shape[1] != window_size:
        raise ValueError(
            f"{name} sequences have {arrays['X'].shape[1]} time steps; "
            f"expected {window_size}."
        )

    return arrays


def _load_scaler() -> tuple[float, float]:
    """Read the MinMaxScaler parameters fitted by the shared data pipeline."""
    path = DATA_DIR / "scaler_params.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m ml.dataset` first."
        )

    with np.load(path) as params:
        scale = float(params["scale"][0])
        offset = float(params["min_offset"][0])

    if scale <= 0:
        raise ValueError("Invalid scaler: scale must be positive.")

    return scale, offset


def _inverse_scale(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Convert scaled flow values back to vehicles per 15-minute interval."""
    return (np.asarray(values).reshape(-1) - offset) / scale


def _evaluate_split(
    model: RandomForestTrafficRegressor,
    split_name: str,
    split: dict[str, np.ndarray],
    scale: float,
    offset: float,
    window_size: int,
) -> dict[str, float | int | str]:
    """Evaluate one split in the original traffic-flow units."""
    predicted_scaled = model.predict(split["X"])
    actual = _inverse_scale(split["y"], scale, offset)
    predicted = _inverse_scale(predicted_scaled, scale, offset)

    # Traffic flow cannot be negative. Apply the same constraint at inference.
    predicted = np.maximum(predicted, 0.0)

    mse = float(mean_squared_error(actual, predicted))
    rmse = float(np.sqrt(mse))
    value_range = float(np.max(actual) - np.min(actual))
    nrmse = rmse / value_range if value_range > 0 else float("nan")

    row: dict[str, float | int | str] = {
        "model": "RandomForest",
        "lookback_steps": window_size,
        "lookback_minutes": window_size * 15,
        "split": split_name,
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": mse,
        "RMSE": rmse,
        "R2": float(r2_score(actual, predicted)),
        "NRMSE": nrmse,
    }

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
            MODEL_DIR / f"test_predictions_w{window_size}.csv",
            index=False,
        )

    print(
        f"RandomForest lookback={window_size} {split_name}: "
        f"RMSE={row['RMSE']:.3f}, MAE={row['MAE']:.3f}, "
        f"MSE={row['MSE']:.3f}, R2={row['R2']:.4f}"
    )
    return row


def _fit_one(
    window_size: int,
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: float,
    n_jobs: int,
) -> list[dict[str, float | int | str]]:
    """Train, evaluate, and save one Random Forest lookback model."""
    train = _load_split("train", window_size)
    validation = _load_split("val", window_size)
    test = _load_split("test", window_size)
    scale, offset = _load_scaler()

    expected_shape = train["X"].shape[1:]
    for name, split in (("validation", validation), ("test", test)):
        if split["X"].shape[1:] != expected_shape:
            raise ValueError(
                f"{name} sequence shape {split['X'].shape[1:]} does not "
                f"match training shape {expected_shape}."
            )

    model = RandomForestTrafficRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=SEED,
        n_jobs=n_jobs,
    )

    print(
        f"Training RandomForest lookback={window_size} "
        f"with {n_estimators} trees..."
    )
    model.fit(train["X"], train["y"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"window_{window_size}.joblib"
    model.save(model_path)

    results = [
        _evaluate_split(
            model, "Validation", validation, scale, offset, window_size
        ),
        _evaluate_split(
            model, "Test", test, scale, offset, window_size
        ),
    ]

    print(f"Saved model: {model_path}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        choices=WINDOW_SIZES,
        default=list(WINDOW_SIZES),
        help="Lookback windows to train (4, 8, and/or 12 observations).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=N_ESTIMATORS,
        help=f"Number of trees per model (default: {N_ESTIMATORS}).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_DEPTH,
        help=f"Maximum tree depth (default: {MAX_DEPTH}).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="Minimum samples required at a leaf (default: 2).",
    )
    parser.add_argument(
        "--max-features",
        type=float,
        default=1.0,
        help="Fraction of input features considered per split (default: 1.0).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs. Use -1 for all available CPU cores.",
    )

    args = parser.parse_args(argv)

    if args.n_estimators < 1:
        parser.error("--n-estimators must be at least 1.")
    if args.max_depth is not None and args.max_depth < 1:
        parser.error("--max-depth must be positive.")
    if args.min_samples_leaf < 1:
        parser.error("--min-samples-leaf must be at least 1.")
    if not 0 < args.max_features <= 1:
        parser.error("--max-features must be greater than 0 and at most 1.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, float | int | str]] = []

    for window_size in args.lookbacks:
        all_results.extend(
            _fit_one(
                window_size,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                n_jobs=args.n_jobs,
            )
        )

    output = MODEL_DIR / "evaluation_results.csv"
    pd.DataFrame(all_results).to_csv(output, index=False)
    print(f"Random Forest evaluation saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
