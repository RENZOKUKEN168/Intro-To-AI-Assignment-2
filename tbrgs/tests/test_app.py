"""Application-level tests for the A2A A* + Yen route integration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tbrgs.app import run_tbrgs
from tbrgs.travel_time import TrafficPrediction


class IntegratedApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.sites_csv = root / "scats_sites.csv"
        self.links_csv = root / "road_links.csv"

        self.sites_csv.write_text(
            "scats_id,site_name,latitude,longitude\n"
            "2000,Origin,-37.80,145.00\n"
            "2001,Route A,-37.801,145.01\n"
            "2002,Route B,-37.802,145.02\n"
            "2003,Route C,-37.803,145.03\n"
            "2004,Route D,-37.804,145.04\n"
            "2005,Route E,-37.805,145.05\n"
            "3002,Destination,-37.81,145.06\n",
            encoding="utf-8",
        )
        self.links_csv.write_text(
            "from_scats,to_scats,distance_km,controlled_intersections\n"
            "2000,2001,1.0,0\n2001,3002,1.0,0\n"
            "2000,2002,1.1,0\n2002,3002,1.1,0\n"
            "2000,2003,1.2,0\n2003,3002,1.2,0\n"
            "2000,2004,1.3,0\n2004,3002,1.3,0\n"
            "2000,2005,1.4,0\n2005,3002,1.4,0\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_tbrgs_returns_five_ranked_loopless_routes(self):
        def fake_prediction(site_id, prediction_time, *, model_name, lookback):
            return TrafficPrediction(
                scats_id=site_id,
                predicted_flow=100.0,
                unit="vehicles_per_15_minutes",
                model_name=model_name,
                prediction_time=prediction_time,
            )

        with patch("tbrgs.app.predict_traffic", side_effect=fake_prediction):
            astar_result, routes, estimates, config, skipped_sites = run_tbrgs(
                2000,
                3002,
                "2006-10-29 04:00",
                "RandomForest",
                4,
                self.sites_csv,
                self.links_csv,
                flow_source="destination",
                speed_branch="free_flow",
                k_routes=5,
            )

        self.assertEqual(astar_result.path[0], 2000)
        self.assertEqual(astar_result.path[-1], 3002)
        self.assertEqual(len(routes), 5)
        self.assertEqual(len({route.nodes for route in routes}), 5)
        self.assertEqual(
            [route.travel_time_seconds for route in routes],
            sorted(route.travel_time_seconds for route in routes),
        )
        for route in routes:
            self.assertEqual(len(route.nodes), len(set(route.nodes)))
        self.assertAlmostEqual(astar_result.path_cost, routes[0].travel_time_seconds)
        self.assertGreater(len(estimates), 0)
        self.assertEqual(config.k_routes, 5)
        self.assertEqual(skipped_sites, [])


    def test_skip_unavailable_sites_is_explicit_and_filters_incident_links(self):
        def fake_prediction(site_id, prediction_time, *, model_name, lookback):
            if site_id == 2005:
                raise ValueError("Not enough history for SCATS site 2005.")
            return TrafficPrediction(
                scats_id=site_id,
                predicted_flow=100.0,
                unit="vehicles_per_15_minutes",
                model_name=model_name,
                prediction_time=prediction_time,
            )

        with patch("tbrgs.app.predict_traffic", side_effect=fake_prediction):
            result, routes, estimates, config, skipped_sites = run_tbrgs(
                2000,
                3002,
                "2006-10-29 04:00",
                "RandomForest",
                12,
                self.sites_csv,
                self.links_csv,
                skip_unavailable_sites=True,
            )

        self.assertEqual(skipped_sites, [2005])
        self.assertEqual(len(routes), 4)
        self.assertTrue(all(2005 not in route.nodes for route in routes))
        self.assertTrue(all(2005 not in edge for edge in estimates))
        self.assertEqual(result.path[0], 2000)
        self.assertEqual(result.path[-1], 3002)

    def _run_with_method(self, method):
        def fake_prediction(site_id, prediction_time, *, model_name, lookback):
            return TrafficPrediction(
                scats_id=site_id,
                predicted_flow=100.0,
                unit="vehicles_per_15_minutes",
                model_name=model_name,
                prediction_time=prediction_time,
            )

        with patch("tbrgs.app.predict_traffic", side_effect=fake_prediction):
            return run_tbrgs(
                2000, 3002, "2006-10-29 04:00", "LSTM", 4,
                self.sites_csv, self.links_csv, search_method=method,
            )

    def test_every_a2a_search_method_finds_a_valid_route(self):
        for method in ("DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"):
            with self.subTest(method=method):
                result, routes, _, _, _ = self._run_with_method(method)
                self.assertEqual(result.path[0], 2000)
                self.assertEqual(result.path[-1], 3002)
                # A non-optimal search can never beat Yen's fastest route.
                self.assertGreaterEqual(
                    result.path_cost + 1e-6, routes[0].travel_time_seconds
                )

    def test_search_method_is_case_insensitive(self):
        result, routes, _, _, _ = self._run_with_method("as")
        self.assertAlmostEqual(result.path_cost, routes[0].travel_time_seconds)

    def test_unknown_search_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown search method"):
            self._run_with_method("DIJKSTRA")


if __name__ == "__main__":
    unittest.main(verbosity=2)
