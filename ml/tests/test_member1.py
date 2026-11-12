
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocess import (
    load_scats_data,
    clean_scats_data,
    check_channel_completeness,
    aggregate_complete_days,
    reshape_time_series,
    TRAFFIC_COLUMNS
)

from ml.dataset import (
    split_data,
    create_sequences
)

from ml.predict_lstm import predict_traffic
from tbrgs.travel_time import TrafficPrediction


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def make_sample_data():
    """Create sample records for unit testing."""

    rows = []

    for day, channels in [
        ("2006-10-01", [1, 3]),
        ("2006-10-02", [1, 3]),
        ("2006-10-03", [1])
    ]:
        for channel in channels:
            row = {
                "SCATS Number": 970,
                "Date": pd.Timestamp(day),
                "HF VicRoads Internal": channel,
                "VR Internal Stat": 182,
                "VR Internal Loc": channel,
                "NB_TYPE_SURVEY": 1
            }

            for col in TRAFFIC_COLUMNS:
                row[col] = channel * 10

            rows.append(row)

    return pd.DataFrame(rows)


class TestDataValidation(unittest.TestCase):

    # TC01
    def test_01_load_original_dataset(self):
        df = load_scats_data()
        self.assertEqual(len(df), 4192)
        self.assertEqual(len(TRAFFIC_COLUMNS), 96)

    # TC02
    def test_02_invalid_scats_id(self):
        df = make_sample_data()

        # Convert column to object to allow invalid input
        df["SCATS Number"] = df["SCATS Number"].astype(object)

        df.loc[0, "SCATS Number"] = "invalid"

        with self.assertRaises(ValueError):
            clean_scats_data(df)

    # TC03
    def test_03_invalid_date(self):
        df = make_sample_data()

        # Convert column to object to allow invalid input
        df["Date"] = df["Date"].astype(object)

        df.loc[0, "Date"] = "invalid-date"

        with self.assertRaises(ValueError):
            clean_scats_data(df)

    # TC04
    def test_04_negative_traffic(self):
        df = make_sample_data()
        df.loc[0, "V00"] = -10

        with self.assertRaises(ValueError):
            clean_scats_data(df)

    # TC05
    def test_05_duplicate_channel_record(self):
        df = make_sample_data()
        df = pd.concat(
            [df, df.iloc[[0]]],
            ignore_index=True
        )

        with self.assertRaises(ValueError):
            clean_scats_data(df)


class TestDataProcessing(unittest.TestCase):

    # TC06
    def test_06_channel_completeness(self):
        df = make_sample_data()
        quality = check_channel_completeness(df)

        self.assertEqual(
            int(quality["is_complete"].sum()), 2
        )

        self.assertEqual(
            int((~quality["is_complete"]).sum()), 1
        )

    # TC07
    def test_07_traffic_aggregation(self):
        df = make_sample_data()
        quality = check_channel_completeness(df)

        daily = aggregate_complete_days(df, quality)

        self.assertEqual(len(daily), 2)

        # Channel 1 = 10, channel 3 = 30
        self.assertEqual(int(daily.iloc[0]["V00"]), 40)

    # TC08
    def test_08_time_series_reshaping(self):
        df = make_sample_data()
        quality = check_channel_completeness(df)

        daily = aggregate_complete_days(df, quality)
        result = reshape_time_series(daily)

        # 2 complete days * 96 intervals
        self.assertEqual(len(result), 192)

        self.assertEqual(
            list(result.columns),
            ["scats_id", "timestamp", "traffic_flow"]
        )

    # TC09
    def test_09_unique_timestamps(self):
        df = make_sample_data()
        quality = check_channel_completeness(df)

        daily = aggregate_complete_days(df, quality)
        result = reshape_time_series(daily)

        duplicates = result.duplicated(
            ["scats_id", "timestamp"]
        ).sum()

        self.assertEqual(duplicates, 0)


class TestDatasetSequences(unittest.TestCase):

    # TC10
    def test_10_chronological_splitting(self):
        df = pd.DataFrame({
            "scats_id": [970, 970, 970],
            "timestamp": pd.to_datetime([
                "2006-10-10",
                "2006-10-23",
                "2006-10-29"
            ]),
            "traffic_flow": [100, 200, 300]
        })

        train, val, test = split_data(df)

        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)

    # TC11
    def test_11_lstm_input_shape(self):
        with np.load(
            PROCESSED_DIR / "train_sequences.npz"
        ) as data:
            X = data["X"]
            y = data["y"]

        self.assertEqual(X.shape[1:], (8, 1))
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)

    # TC12
    def test_12_consecutive_time_windows(self):
        times = pd.date_range(
            "2006-10-01",
            periods=10,
            freq="15min"
        )

        df = pd.DataFrame({
            "scats_id": [970] * 10,
            "timestamp": times,
            "scaled_flow": np.linspace(0, 1, 10)
        })

        X, y, _, _ = create_sequences(df)

        self.assertEqual(X.shape, (2, 8, 1))
        self.assertEqual(len(y), 2)

    # TC13
    def test_13_missing_interval_rejected(self):
        times = list(pd.date_range(
            "2006-10-01",
            periods=10,
            freq="15min"
        ))

        # Create a gap in the middle
        times[5] = times[5] + pd.Timedelta(hours=2)

        df = pd.DataFrame({
            "scats_id": [970] * 10,
            "timestamp": times,
            "scaled_flow": np.linspace(0, 1, 10)
        })

        X, _, _, _ = create_sequences(df)

        self.assertEqual(len(X), 0)

    # TC14
    def test_14_separate_scats_sites(self):
        times = pd.date_range(
            "2006-10-01",
            periods=9,
            freq="15min"
        )

        df = pd.concat([
            pd.DataFrame({
                "scats_id": [970] * 9,
                "timestamp": times,
                "scaled_flow": np.linspace(0, 1, 9)
            }),
            pd.DataFrame({
                "scats_id": [2000] * 9,
                "timestamp": times,
                "scaled_flow": np.linspace(0, 1, 9)
            })
        ], ignore_index=True)

        X, _, sites, _ = create_sequences(df)

        self.assertEqual(len(X), 2)
        self.assertEqual(set(sites), {970, 2000})


class TestLSTMPrediction(unittest.TestCase):

    # TC15
    def test_15_valid_lstm_prediction(self):
        result = predict_traffic(
            970,
            "2006-10-27 08:00:00"
        )

        self.assertIsInstance(result, TrafficPrediction)
        self.assertEqual(result.scats_id, 970)
        self.assertEqual(result.model_name, "LSTM")
        self.assertEqual(
            result.unit,
            "vehicles_per_15_minutes"
        )
        self.assertTrue(
            np.isfinite(result.predicted_flow)
        )
        self.assertGreaterEqual(result.predicted_flow, 0)

    # TC16
    def test_16_unknown_scats_site(self):
        with self.assertRaises(ValueError):
            predict_traffic(
                999999,
                "2006-10-27 08:00:00"
            )

    # TC17
    def test_17_invalid_prediction_time(self):
        with self.assertRaises(ValueError):
            predict_traffic(
                970,
                "2006-10-27 08:07:00"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
