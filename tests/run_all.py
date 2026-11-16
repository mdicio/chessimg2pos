"""Run the test suite from anywhere:  python tests/run_all.py [-v]"""
import os
import sys

import pytest

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
sys.exit(pytest.main([here, *sys.argv[1:]]))
