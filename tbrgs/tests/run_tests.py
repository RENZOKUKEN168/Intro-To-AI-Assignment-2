"""Convenience test runner. Prefer `python -m unittest discover -s tbrgs/tests -v`."""

import unittest

suite = unittest.defaultTestLoader.discover("tbrgs/tests", pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
