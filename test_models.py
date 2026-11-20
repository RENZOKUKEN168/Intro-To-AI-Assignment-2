import tempfile
import unittest
from pathlib import Path

import numpy as np

from models import (
    BiLSTMRegressor,
    GRURegressor,
    RandomForestTrafficRegressor,
    RegressionTreeRegressor,
    evaluate_predictions,
    make_forecasting_windows,
)


class RegressionTreeTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [[1.0], [2.0], [3.0]],
                [[2.0], [3.0], [4.0]],
                [[3.0], [4.0], [5.0]],
                [[4.0], [5.0], [6.0]],
            ],
            dtype=np.float32,
        )
        self.y = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)

    def test_fit_predict_has_one_value_per_sample(self):
        model = RegressionTreeRegressor(max_depth=3, min_samples_leaf=1).fit(self.X, self.y)
        prediction = model.predict(self.X)
        self.assertEqual(prediction.shape, (4,))
        self.assertTrue(np.isfinite(prediction).all())

        # Each 15-minute series uses 4, 8, or 12 past observations to
        # predict the value immediately after the window.
        series = np.arange(20, dtype=np.float32)
        for lookback in (4, 8, 12):
            with self.subTest(lookback=lookback):
                X_window, y_next = make_forecasting_windows(series, lookback)
                self.assertEqual(X_window.shape, (20 - lookback, lookback, 1))
                np.testing.assert_array_equal(X_window[0, :, 0], np.arange(lookback))
                self.assertEqual(y_next[0], lookback)

    def test_predict_accepts_one_sequence(self):
        model = RegressionTreeRegressor(min_samples_leaf=1).fit(self.X, self.y)
        self.assertEqual(model.predict(self.X[0]).shape, (1,))

        rows = np.column_stack((np.arange(15), np.arange(100, 115)))
        X_window, y_next = make_forecasting_windows(rows, 4, target_column=1)
        self.assertEqual(X_window.shape, (11, 4, 2))
        self.assertEqual(y_next[0], 104)

    def test_save_and_load_preserves_predictions(self):
        model = RegressionTreeRegressor(min_samples_leaf=1).fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tree.joblib"
            model.save(path)
            loaded = RegressionTreeRegressor.load(path)
            np.testing.assert_allclose(loaded.predict(self.X), model.predict(self.X))

    def test_predict_before_fit_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            RegressionTreeRegressor().predict(self.X)

    def test_wrong_sequence_shape_is_rejected(self):
        model = RegressionTreeRegressor().fit(self.X, self.y)
        with self.assertRaisesRegex(ValueError, "Expected each sequence"):
            model.predict(np.zeros((2, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "4, 8, or 12"):
            make_forecasting_windows(np.arange(8), 6)
        with self.assertRaisesRegex(ValueError, "at least"):
            make_forecasting_windows(np.arange(4), 4)

    def test_mismatched_target_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "samples"):
            RegressionTreeRegressor().fit(self.X, self.y[:2])

    def test_nan_features_are_rejected(self):
        bad_X = self.X.copy()
        bad_X[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            RegressionTreeRegressor().fit(bad_X, self.y)
        bad_series = np.arange(8, dtype=np.float32)
        bad_series[2] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            make_forecasting_windows(bad_series, 4)

    def test_invalid_tree_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_depth"):
            RegressionTreeRegressor(max_depth=0)
        with self.assertRaisesRegex(ValueError, "min_samples_leaf"):
            RegressionTreeRegressor(min_samples_leaf=0)


class RandomForestTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [[1.0], [2.0], [3.0]],
                [[2.0], [3.0], [4.0]],
                [[3.0], [4.0], [5.0]],
                [[4.0], [5.0], [6.0]],
            ],
            dtype=np.float32,
        )
        self.y = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)

    def create_model(self):
        return RandomForestTrafficRegressor(
            n_estimators=20,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=1,
        )

    def test_fit_predict_has_one_value_per_sample(self):
        model = self.create_model().fit(self.X, self.y)
        prediction = model.predict(self.X)

        self.assertEqual(prediction.shape, (4,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_predict_accepts_one_sequence(self):
        model = self.create_model().fit(self.X, self.y)
        prediction = model.predict(self.X[0])

        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_save_and_load_preserves_predictions(self):
        model = self.create_model().fit(self.X, self.y)
        original_predictions = model.predict(self.X)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "random_forest.joblib"
            model.save(path)
            loaded = RandomForestTrafficRegressor.load(path)
            loaded_predictions = loaded.predict(self.X)

        np.testing.assert_allclose(loaded_predictions, original_predictions)

    def test_predict_before_fit_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            self.create_model().predict(self.X)

    def test_wrong_sequence_shape_is_rejected(self):
        model = self.create_model().fit(self.X, self.y)

        with self.assertRaisesRegex(ValueError, "Expected each sequence"):
            model.predict(np.zeros((2, 2), dtype=np.float32))

    def test_nan_features_are_rejected(self):
        bad_X = self.X.copy()
        bad_X[0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN"):
            self.create_model().fit(bad_X, self.y)

    def test_mismatched_target_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "samples"):
            self.create_model().fit(self.X, self.y[:2])

    def test_invalid_random_forest_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "n_estimators"):
            RandomForestTrafficRegressor(n_estimators=0)
        with self.assertRaisesRegex(ValueError, "max_depth"):
            RandomForestTrafficRegressor(max_depth=0)
        with self.assertRaisesRegex(ValueError, "min_samples_leaf"):
            RandomForestTrafficRegressor(min_samples_leaf=0)
        with self.assertRaisesRegex(ValueError, "max_features"):
            RandomForestTrafficRegressor(max_features=0)


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_range_normalized_rmse(self):
        result = evaluate_predictions([0.0, 2.0], [0.0, 1.0])
        self.assertAlmostEqual(result.rmse, np.sqrt(0.5))
        self.assertAlmostEqual(result.nrmse, np.sqrt(0.5) / 2.0)
        self.assertAlmostEqual(result.mae, 0.5)
        self.assertAlmostEqual(result.mse, 0.5)
        self.assertAlmostEqual(result.r2, 0.5)
        self.assertEqual(set(result.as_dict()), {"rmse", "mae", "mse", "r2", "nrmse"})

    def test_mismatched_prediction_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "samples"):
            evaluate_predictions([1.0, 2.0], [1.0])

    def test_constant_targets_reject_undefined_nrmse(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            evaluate_predictions([2.0, 2.0], [2.0, 3.0])

    def test_non_finite_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            evaluate_predictions([1.0, np.nan], [1.0, 2.0])


class GRUContractTests(unittest.TestCase):
    def test_invalid_gru_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "units"):
            GRURegressor(units=0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            GRURegressor(dropout=1.0)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            GRURegressor(learning_rate=0)
        with self.assertRaisesRegex(ValueError, "units"):
            BiLSTMRegressor(units=0)

    def test_predict_before_fit_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            GRURegressor().predict(np.ones((3, 1), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            BiLSTMRegressor().predict(np.ones((4, 1), dtype=np.float32))

    def test_fit_reports_missing_tensorflow(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            model = GRURegressor()
            with self.assertRaisesRegex(ImportError, "requires TensorFlow"):
                model.fit(np.ones((2, 3, 1)), np.ones(2))


if __name__ == "__main__":
    unittest.main()
