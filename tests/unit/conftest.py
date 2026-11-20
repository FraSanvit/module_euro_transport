"""Fixtures and import setup for unit tests of module scripts."""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[2] / "workflow" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
