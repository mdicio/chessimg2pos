"""
tests/run_all.py
----------------
Run the full test suite from the repository root:

    python tests/run_all.py            # normal output
    python tests/run_all.py -v         # verbose output

All tests use only stdlib + the project's own dependencies —
no internet access and no real chess images are needed.
"""

import sys
import os
import unittest

# Allow running from either the repo root or the tests/ folder
_here = os.path.dirname(__file__)
_src = os.path.join(_here, "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, os.path.abspath(_src))

loader = unittest.TestLoader()
suite = loader.discover(start_dir=_here, pattern="test_*.py")

verbosity = 2 if "-v" in sys.argv else 1
runner = unittest.TextTestRunner(verbosity=verbosity)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
