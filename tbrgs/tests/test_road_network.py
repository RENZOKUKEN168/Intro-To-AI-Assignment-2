"""Checks on the generated Boroondara road network files."""

import csv
import unittest
from pathlib import Path

from tbrgs.build_road_network import ROAD_SEGMENTS, build_links, haversine_km, unreachable_sites
from tbrgs.road_graph import load_road_network

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "tbrgs"
SITES_CSV = DATA_DIR / "scats_sites.csv"
LINKS_CSV = DATA_DIR / "road_links.csv"


class RoadNetworkFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = load_road_network(SITES_CSV, LINKS_CSV)
        with LINKS_CSV.open(newline="", encoding="utf-8") as handle:
            cls.links = [
                (int(r["from_scats"]), int(r["to_scats"]), float(r["distance_km"]), int(r["controlled_intersections"]))
                for r in csv.DictReader(handle)
            ]

    def test_all_40_sites_have_coordinates(self):
        self.assertEqual(len(self.network.sites), 40)
        for site in self.network.sites.values():
            self.assertIsNotNone(site.latitude)
            self.assertIsNotNone(site.longitude)
            self.assertTrue(-38.0 < site.latitude < -37.7)
            self.assertTrue(144.9 < site.longitude < 145.2)

    def test_every_site_is_reachable(self):
        self.assertEqual(unreachable_sites(set(self.network.sites), self.links), set())

    def test_every_link_has_a_reverse_link_of_equal_length(self):
        lengths = {(a, b): d for a, b, d, _ in self.links}
        for (a, b), distance in lengths.items():
            self.assertEqual(lengths.get((b, a)), distance, f"{a}->{b} has no matching reverse link")

    def test_link_distances_are_plausible(self):
        for a, b, distance, controlled in self.links:
            self.assertGreater(distance, 0.1, f"{a}->{b}")
            self.assertLess(distance, 5.0, f"{a}->{b}")
            self.assertEqual(controlled, 1)

    def test_brief_example_sites_are_in_network(self):
        self.assertIn(2000, self.network.sites)
        self.assertIn(3002, self.network.sites)

    def test_haversine_known_distance(self):
        # One degree of latitude is about 111.2 km.
        self.assertAlmostEqual(haversine_km((-37.0, 145.0), (-38.0, 145.0)), 111.2, delta=0.2)

    def test_unknown_site_in_segment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no coordinates"):
            build_links({1: (-37.8, 145.0)}, {"Test Rd": [1, 2]})

    def test_segments_only_use_known_sites(self):
        used = {site for sites in ROAD_SEGMENTS.values() for site in sites}
        self.assertEqual(used - set(self.network.sites), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
