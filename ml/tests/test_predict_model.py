"""Tests for model-specific inference loading, including Random Forest."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from models import RandomForestTrafficRegressor
import ml.predict_model as predictor


class RandomForestPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model_dir = self.root / "models"
        model_path = self.model_dir / "random_forest" / "window_4.joblib"

        # A small deterministic artifact is sufficient to verify the inference contract.
        X_train = np.array(
            [
                [[0.10], [0.20], [0.30], [0.40]],
                [[0.20], [0.30], [0.40], [0.50]],
                [[0.30], [0.40], [0.50], [0.60]],
                [[0.40], [0.50], [0.60], [0.70]],
                [[0.50], [0.60], [0.70], [0.80]],
            ],
            dtype=np.float32,
        )
        y_train = np.array([0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float32)
        model = RandomForestTrafficRegressor(n_estimators=5, n_jobs=1, random_state=7)
        model.fit(X_train, y_train)
        model.save(model_path)

        timestamps = pd.date_range("2006-10-01 00:00", periods=4, freq="15min")
        frame = pd.DataFrame(
            {
                "scats_id": [2000] * 4,
                "timestamp": timestamps,
                "traffic_flow": [10.0, 20.0, 30.0, 40.0],
            }
        )
        self.data_path = self.root / "scats_traffic_clean.csv"
        frame.to_csv(self.data_path, index=False)
        self.scaler_path = self.root / "scaler_params.npz"
        np.savez(self.scaler_path, scale=np.array([0.01]), min_offset=np.array([0.0]))

        self.patches = [
            patch.object(predictor, "DATA_PATH", self.data_path),
            patch.object(predictor, "SCALER_PATH", self.scaler_path),
            patch.object(predictor, "MODEL_DIR", self.model_dir),
        ]
        for item in self.patches:
            item.start()
        predictor._load_traffic_data.cache_clear()
        predictor._load_model.cache_clear()

    def tearDown(self):
        predictor._load_traffic_data.cache_clear()
        predictor._load_model.cache_clear()
        for item in self.patches:
            item.stop()
        self.temp_dir.cleanup()

    def test_random_forest_predicts_without_tensorflow(self):
        result = predictor.predict_traffic(
            2000,
            "2006-10-01 01:00",
            model_name="RandomForest",
            lookback=4,
        )
        self.assertEqual(result.scats_id, 2000)
        self.assertEqual(result.model_name, "RandomForest")
        self.assertEqual(result.unit, "vehicles_per_15_minutes")
        self.assertGreaterEqual(result.predicted_flow, 0.0)

    def test_random_forest_alias_is_accepted(self):
        result = predictor.predict_traffic(
            2000,
            "2006-10-01 01:00",
            model_name="random_forest",
            lookback=4,
        )
        self.assertEqual(result.model_name, "RandomForest")


if __name__ == "__main__":
    unittest.main(verbosity=2)
