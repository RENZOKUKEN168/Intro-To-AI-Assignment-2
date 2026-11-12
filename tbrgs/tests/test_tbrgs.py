"""Initial automated tests for the TBRGS integration scaffold.

Run from the repository root:
    python -m unittest discover -s tbrgs/tests -v
"""

import tempfile
import unittest
from pathlib import Path

from tbrgs.config import TBRGSConfig
from tbrgs.road_graph import (
    RoadLink,
    ScatsSite,
    build_road_network,
    load_road_network,
)
from tbrgs.travel_time import (
    TrafficPrediction,
    build_time_weighted_graph,
    estimate_link_travel_time,
    flow_to_speed_kmh,
    flow_to_vehicles_per_hour,
)
from tbrgs.top_k_routes import (
    dijkstra_shortest_path,
    k_shortest_loopless_paths,
)
from route_problem import RouteProblem
from inform_search import astar_search


class FlowConversionTests(unittest.TestCase):
    def test_15_minute_flow_is_multiplied_by_four(self):
        self.assertEqual(flow_to_vehicles_per_hour(80, "vehicles_per_15_minutes"), 320)

    def test_hourly_flow_is_not_multiplied(self):
        self.assertEqual(flow_to_vehicles_per_hour(320, "vehicles_per_hour"), 320)

    def test_unknown_flow_unit_is_rejected(self):
        with self.assertRaises(ValueError):
            flow_to_vehicles_per_hour(100, "vehicles_per_day")

    def test_negative_flow_is_rejected(self):
        with self.assertRaises(ValueError):
            flow_to_vehicles_per_hour(-1, "vehicles_per_hour")

    def test_speed_is_capped_at_sixty_for_low_flow(self):
        self.assertEqual(flow_to_speed_kmh(300), 60.0)

    def test_free_flow_branch_is_faster_than_congested_branch(self):
        free_speed = flow_to_speed_kmh(1000, branch="free_flow")
        congested_speed = flow_to_speed_kmh(1000, branch="congested")
        self.assertGreater(free_speed, congested_speed)

    def test_flow_above_capacity_raises_by_default(self):
        with self.assertRaises(ValueError):
            flow_to_speed_kmh(1600)

    def test_explicit_clamp_policy_caps_flow_at_capacity(self):
        at_capacity = flow_to_speed_kmh(1500, branch="free_flow")
        clamped = flow_to_speed_kmh(
            1600, branch="free_flow", oversaturation_policy="clamp"
        )
        self.assertAlmostEqual(at_capacity, clamped, places=6)


class RoadNetworkTests(unittest.TestCase):
    def setUp(self):
        self.sites = [ScatsSite(2000, "A"), ScatsSite(3002, "B")]

    def test_valid_directed_network_is_created(self):
        network = build_road_network(self.sites, [RoadLink(2000, 3002, 1.5)])
        self.assertEqual(network.node_ids, (2000, 3002))
        self.assertEqual(len(network.links), 1)

    def test_duplicate_site_id_is_rejected(self):
        with self.assertRaises(ValueError):
            build_road_network(
                [ScatsSite(2000), ScatsSite(2000)], [RoadLink(2000, 3002, 1.0)]
            )

    def test_link_to_unknown_site_is_rejected(self):
        with self.assertRaises(ValueError):
            build_road_network(self.sites, [RoadLink(2000, 9999, 1.0)])

    def test_duplicate_directed_link_is_rejected(self):
        with self.assertRaises(ValueError):
            build_road_network(
                self.sites,
                [RoadLink(2000, 3002, 1.0), RoadLink(2000, 3002, 1.2)],
            )

    def test_reverse_direction_must_be_explicit(self):
        network = build_road_network(self.sites, [RoadLink(2000, 3002, 1.0)])
        self.assertEqual(len(network.outgoing_links(3002)), 0)

    def test_invalid_road_distance_is_rejected(self):
        with self.assertRaises(ValueError):
            RoadLink(2000, 3002, 0)

    def test_csv_network_loader_uses_road_distance_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sites_file = root / "sites.csv"
            links_file = root / "links.csv"
            sites_file.write_text(
                "scats_id,site_name,latitude,longitude\n"
                "2000,Origin,,\n3002,Destination,,\n",
                encoding="utf-8",
            )
            links_file.write_text(
                "from_scats,to_scats,distance_km,controlled_intersections\n"
                "2000,3002,2.75,1\n",
                encoding="utf-8",
            )
            network = load_road_network(sites_file, links_file)
            self.assertEqual(network.links[0].distance_km, 2.75)
            self.assertEqual(network.links[0].controlled_intersections, 1)


class TravelTimeTests(unittest.TestCase):
    def test_travel_time_includes_signal_delay(self):
        result = estimate_link_travel_time(
            1.0,
            100,
            flow_unit="vehicles_per_15_minutes",
            controlled_intersections=2,
        )
        self.assertEqual(result.signal_delay_seconds, 60.0)
        self.assertAlmostEqual(result.total_seconds, result.driving_seconds + 60.0)

    def test_hourly_and_15_minute_units_produce_same_result_when_equivalent(self):
        per_15 = estimate_link_travel_time(
            2.0, 100, flow_unit="vehicles_per_15_minutes"
        )
        per_hour = estimate_link_travel_time(
            2.0, 400, flow_unit="vehicles_per_hour"
        )
        self.assertAlmostEqual(per_15.total_seconds, per_hour.total_seconds)

    def test_edge_flow_source_can_be_switched(self):
        network = build_road_network(
            [ScatsSite(2000), ScatsSite(3002)], [RoadLink(2000, 3002, 1.0)]
        )
        predictions = {
            2000: TrafficPrediction(2000, 50),
            3002: TrafficPrediction(3002, 100),
        }
        destination_graph, _ = build_time_weighted_graph(
            network, predictions, TBRGSConfig(flow_source="destination")
        )
        origin_graph, _ = build_time_weighted_graph(
            network, predictions, TBRGSConfig(flow_source="origin")
        )
        self.assertNotEqual(destination_graph[2000][3002], origin_graph[2000][3002])

    def test_missing_edge_prediction_is_reported(self):
        network = build_road_network(
            [ScatsSite(2000), ScatsSite(3002)], [RoadLink(2000, 3002, 1.0)]
        )
        with self.assertRaises(KeyError):
            build_time_weighted_graph(network, {2000: TrafficPrediction(2000, 50)})

    def test_prediction_mapping_key_must_match_prediction_id(self):
        network = build_road_network(
            [ScatsSite(2000), ScatsSite(3002)], [RoadLink(2000, 3002, 1.0)]
        )
        predictions = {3002: TrafficPrediction(2000, 50)}
        with self.assertRaises(ValueError):
            build_time_weighted_graph(network, predictions)


class RouteSearchTests(unittest.TestCase):
    def setUp(self):
        # Four distinct paths from 2000 to 3002, with one directed detour.
        self.graph = {
            2000: {2001: 10.0, 2002: 12.0, 2003: 15.0},
            2001: {3002: 10.0, 2002: 2.0},
            2002: {3002: 12.0},
            2003: {3002: 12.0},
            3002: {},
        }

    def test_dijkstra_returns_lowest_cost_path(self):
        route = dijkstra_shortest_path(self.graph, 2000, 3002)
        self.assertEqual(route.nodes, (2000, 2001, 3002))
        self.assertEqual(route.travel_time_seconds, 20.0)

        # The same traffic-derived seconds can be supplied to A2A's A* search.
        coordinates = {node: (0.0, 0.0) for node in self.graph}
        problem = RouteProblem(self.graph, 2000, [3002], coordinates)
        a2a_result = astar_search(problem)
        self.assertEqual(a2a_result.path, [2000, 2001, 3002])
        self.assertEqual(a2a_result.path_cost, 20.0)

    def test_yen_returns_routes_in_ascending_cost_order(self):
        routes = k_shortest_loopless_paths(self.graph, 2000, 3002, k=4)
        costs = [route.travel_time_seconds for route in routes]
        self.assertEqual(costs, sorted(costs))
        self.assertEqual(len(routes), 4)

    def test_routes_are_distinct(self):
        routes = k_shortest_loopless_paths(self.graph, 2000, 3002, k=5)
        paths = [route.nodes for route in routes]
        self.assertEqual(len(paths), len(set(paths)))

    def test_routes_are_loopless(self):
        routes = k_shortest_loopless_paths(self.graph, 2000, 3002, k=5)
        for route in routes:
            self.assertEqual(len(route.nodes), len(set(route.nodes)))

    def test_fewer_than_k_paths_are_returned_when_network_has_fewer(self):
        graph = {1: {2: 2.0}, 2: {3: 3.0}, 3: {}}
        routes = k_shortest_loopless_paths(graph, 1, 3, k=5)
        self.assertEqual(len(routes), 1)

    def test_unreachable_destination_returns_empty_list(self):
        graph = {1: {2: 1.0}, 2: {}, 3: {}}
        self.assertEqual(k_shortest_loopless_paths(graph, 1, 3, k=5), [])

    def test_same_origin_and_destination_returns_zero_cost_route(self):
        graph = {1: {2: 1.0}, 2: {}}
        route = k_shortest_loopless_paths(graph, 1, 1, k=5)
        self.assertEqual(len(route), 1)
        self.assertEqual(route[0].nodes, (1,))
        self.assertEqual(route[0].travel_time_seconds, 0.0)

    def test_invalid_k_is_rejected(self):
        with self.assertRaises(ValueError):
            k_shortest_loopless_paths(self.graph, 2000, 3002, k=0)

    def test_negative_edge_weight_is_rejected(self):
        graph = {1: {2: -1.0}, 2: {}}
        with self.assertRaises(ValueError):
            k_shortest_loopless_paths(graph, 1, 2, k=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
