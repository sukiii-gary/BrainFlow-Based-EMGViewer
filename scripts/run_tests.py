"""Run existing tests from any checkout path without touching physical boards."""
import os
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.chdir(root / 'app')
sys.path[:0] = [str(root / 'app'), str(root / 'emg_python')]
suite = unittest.defaultTestLoader.discover(str(root / 'app' / 'tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
