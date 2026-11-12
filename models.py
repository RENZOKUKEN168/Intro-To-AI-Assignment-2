"""Traffic forecasting models for 15-minute SCATS data.

GRU, BiLSTM, and the regression-tree baseline use the same contract:
    X: (samples, timesteps, features)
    y: (samples,) or (samples, 1)

Data cleaning, chronological splitting, and any scaling belong to the shared
data pipeline. Fit scalers on training data only and pass transformed arrays to
both estimators so the comparison remains fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


def _as_3d_features(X: Any, *, name: str = "X") -> np.ndarray:
    """Validate sequence features and return a finite float32 3-D array."""
    array = np.asarray(X, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (samples, timesteps, features); got {array.shape}")
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must not contain an empty dimension")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _as_targets(y: Any, *, expected_samples: int | None = None) -> np.ndarray:
    """Validate scalar traffic-flow targets and return a 1-D float32 array."""
    target = np.asarray(y, dtype=np.float32)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 1:
        raise ValueError(f"y must have shape (samples,) or (samples, 1); got {target.shape}")
    if target.size == 0:
        raise ValueError("y must not be empty")
    if expected_samples is not None and target.shape[0] != expected_samples:
        raise ValueError(f"X has {expected_samples} samples but y has {target.shape[0]}")
    if not np.isfinite(target).all():
        raise ValueError("y contains NaN or infinite values")
    return target


def make_forecasting_windows(data: Any, lookback: int, *, target_column: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Create a one-step-ahead dataset from one location's time-ordered rows.

    Each row represents one 15-minute observation. Allowed lookbacks are 4, 8,
    or 12 rows (1, 2, or 3 hours). Sort by timestamp and call separately for
    each location before using this function so windows never cross locations.
    The target is the selected column in the row immediately after each window.
    """
    if lookback not in (4, 8, 12):
        raise ValueError("lookback must be 4, 8, or 12 observations (15-minute intervals)")
    rows = np.asarray(data, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 1)
    if rows.ndim != 2 or rows.shape[1] == 0:
        raise ValueError("data must have shape (observations, features)")
    if not np.isfinite(rows).all():
        raise ValueError("data contains NaN or infinite values")
    if rows.shape[0] <= lookback:
        raise ValueError(f"data needs at least {lookback + 1} observations for this lookback")
    if not -rows.shape[1] <= target_column < rows.shape[1]:
        raise ValueError("target_column is outside the available feature columns")

    target_index = target_column % rows.shape[1]
    sample_count = rows.shape[0] - lookback
    X = np.stack([rows[i : i + lookback] for i in range(sample_count)]).astype(np.float32)
    y = rows[lookback:, target_index].astype(np.float32)
    return X, y


def _prediction_sequences(X: Any, *, timesteps: int, features: int) -> np.ndarray:
    """Accept one sequence or a batch and validate it against a fitted model."""
    array = np.asarray(X, dtype=np.float32)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError("X must have shape (timesteps, features) or (samples, timesteps, features)")
    if array.shape[1:] != (timesteps, features):
        raise ValueError(
            f"Expected each sequence to have shape ({timesteps}, {features}); got {array.shape[1:]}"
        )
    if not np.isfinite(array).all():
        raise ValueError("X contains NaN or infinite values")
    return array


@dataclass(frozen=True)
class Evaluation:
    """Forecast metrics; NRMSE uses the held-out target range."""

    rmse: float
    nrmse: float
    mae: float
    mse: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mse": self.mse,
            "r2": self.r2,
            "nrmse": self.nrmse,
        }


def evaluate_predictions(y_true: Any, y_pred: Any) -> Evaluation:
    """Calculate RMSE, MAE, MSE, R², and range-normalized RMSE."""
    actual = _as_targets(y_true)
    predicted = _as_targets(y_pred, expected_samples=actual.size)
    if actual.size < 2:
        raise ValueError("At least two held-out samples are needed to calculate R² and NRMSE")
    error = predicted - actual
    mse = float(np.mean(np.square(error)))
    rmse = float(np.sqrt(mse))
    value_range = float(np.max(actual) - np.min(actual))
    if value_range == 0.0:
        raise ValueError("NRMSE is undefined when all held-out actual values are identical")
    nrmse = rmse / value_range
    mae = float(np.mean(np.abs(error)))
    r2 = float(r2_score(actual, predicted))
    return Evaluation(rmse=rmse, nrmse=nrmse, mae=mae, mse=mse, r2=r2)


class RegressionTreeRegressor:
    """Decision tree regressor using flattened traffic-history windows."""

    def __init__(
        self,
        *,
        max_depth: int | None = 12,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> None:
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be positive or None")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self._sequence_shape: tuple[int, int] | None = None

    def fit(self, X: Any, y: Any) -> "RegressionTreeRegressor":
        sequences = _as_3d_features(X)
        targets = _as_targets(y, expected_samples=sequences.shape[0])
        self._sequence_shape = (sequences.shape[1], sequences.shape[2])
        self._model.fit(sequences.reshape(sequences.shape[0], -1), targets)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self._sequence_shape is None:
            raise RuntimeError("Model is not fitted; call fit(X, y) before predict(X)")
        sequences = _prediction_sequences(
            X, timesteps=self._sequence_shape[0], features=self._sequence_shape[1]
        )
        return np.asarray(self._model.predict(sequences.reshape(sequences.shape[0], -1)), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        if self._sequence_shape is None:
            raise RuntimeError("Cannot save an unfitted model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "sequence_shape": self._sequence_shape,
                "config": {
                    "max_depth": self.max_depth,
                    "min_samples_leaf": self.min_samples_leaf,
                    "random_state": self.random_state,
                },
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RegressionTreeRegressor":
        artifact = joblib.load(Path(path))
        instance = cls(**artifact["config"])
        instance._model = artifact["model"]
        instance._sequence_shape = tuple(artifact["sequence_shape"])
        return instance


class GRURegressor:
    """Keras GRU regressor; TensorFlow is loaded only when this model is used."""

    def __init__(
        self,
        *,
        units: int = 64,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        if units < 1:
            raise ValueError("units must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model: Any | None = None
        self._sequence_shape: tuple[int, int] | None = None

    def _build(self, timesteps: int, features: int) -> Any:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "GRURegressor requires TensorFlow. Install the project's optional "
                "GRU dependencies with `pip install -r requirements-gru.txt`."
            ) from exc

        tf.keras.utils.set_random_seed(self.random_state)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(timesteps, features)),
                tf.keras.layers.GRU(self.units, dropout=self.dropout),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error",
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        return model

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        epochs: int = 50,
        batch_size: int = 32,
        validation_data: tuple[Any, Any] | None = None,
        patience: int = 6,
        verbose: int = 0,
    ) -> "GRURegressor":
        if epochs < 1 or batch_size < 1 or patience < 0:
            raise ValueError("epochs and batch_size must be positive; patience must be non-negative")
        sequences = _as_3d_features(X)
        targets = _as_targets(y, expected_samples=sequences.shape[0])
        self._sequence_shape = (sequences.shape[1], sequences.shape[2])
        self._model = self._build(*self._sequence_shape)

        callbacks: list[Any] = []
        try:
            import tensorflow as tf
        except ImportError as exc:  # _build already gives the detailed dependency message.
            raise ImportError("GRURegressor requires TensorFlow") from exc
        if validation_data is not None:
            valid_X = _prediction_sequences(
                validation_data[0], timesteps=self._sequence_shape[0], features=self._sequence_shape[1]
            )
            valid_y = _as_targets(validation_data[1], expected_samples=valid_X.shape[0])
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=patience, restore_best_weights=True
                )
            )
            validation_data = (valid_X, valid_y)

        self._model.fit(
            sequences,
            targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=False,
            verbose=verbose,
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self._model is None or self._sequence_shape is None:
            raise RuntimeError("Model is not fitted; call fit(X, y) before predict(X)")
        sequences = _prediction_sequences(
            X, timesteps=self._sequence_shape[0], features=self._sequence_shape[1]
        )
        return np.asarray(self._model.predict(sequences, verbose=0), dtype=np.float32).reshape(-1)

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(destination)

    @classmethod
    def load(cls, path: str | Path) -> "GRURegressor":
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("Loading a GRU model requires TensorFlow") from exc
        destination = Path(path)
        model = tf.keras.models.load_model(destination)
        instance = cls()
        instance._model = model
        input_shape = model.input_shape
        instance._sequence_shape = (int(input_shape[1]), int(input_shape[2]))
        return instance


class BiLSTMRegressor(GRURegressor):
    """Bidirectional LSTM with the same fit/predict/save contract as GRU."""

    def _build(self, timesteps: int, features: int) -> Any:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "BiLSTMRegressor requires TensorFlow. Install dependencies with "
                "`pip install -r requirements-gru.txt`."
            ) from exc

        tf.keras.utils.set_random_seed(self.random_state)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(timesteps, features)),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.units, dropout=self.dropout)
                ),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error",
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        return model

    @classmethod
    def load(cls, path: str | Path) -> "BiLSTMRegressor":
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("Loading a BiLSTM model requires TensorFlow") from exc
        model = tf.keras.models.load_model(Path(path))
        instance = cls()
        instance._model = model
        input_shape = model.input_shape
        instance._sequence_shape = (int(input_shape[1]), int(input_shape[2]))
        return instance
