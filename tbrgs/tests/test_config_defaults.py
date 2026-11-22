"""Tests for loading GUI defaults from the JSON configuration file."""

import json
import tempfile
import unittest
from pathlib import Path

from tbrgs.config import BUILTIN_DEFAULTS, DEFAULTS_FILE, load_defaults


class ConfigDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "config.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_repository_config_file_loads(self):
        defaults = load_defaults(DEFAULTS_FILE)
        self.assertEqual(set(defaults), set(BUILTIN_DEFAULTS))
        self.assertTrue(defaults["sites_csv"].is_absolute())

    def test_missing_file_uses_builtin_defaults(self):
        defaults = load_defaults(self.path)
        self.assertEqual(defaults["origin"], BUILTIN_DEFAULTS["origin"])
        self.assertEqual(defaults["search_method"], "AS")

    def test_file_values_override_builtins(self):
        self.path.write_text(json.dumps({"model": "GRU", "k_routes": 3}), encoding="utf-8")
        defaults = load_defaults(self.path)
        self.assertEqual(defaults["model"], "GRU")
        self.assertEqual(defaults["k_routes"], 3)
        self.assertEqual(defaults["destination"], BUILTIN_DEFAULTS["destination"])

    def test_unknown_key_is_reported(self):
        self.path.write_text(json.dumps({"modle": "GRU"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unknown keys"):
            load_defaults(self.path)

    def test_invalid_json_is_reported(self):
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            load_defaults(self.path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
