"""Smoke tests for the unit-test environment."""

import importlib
import tomllib
from pathlib import Path

import pytest

PIXI_TOML = Path(__file__).parents[2] / "pixi.toml"
# conda package name -> python import name (only where they differ)
IMPORT_NAMES = {"curl": None, "odfpy": "odf"}


def _module_packages() -> list[str]:
    with PIXI_TOML.open("rb") as f:
        deps = tomllib.load(f)["feature"]["module"]["dependencies"]
    return [
        IMPORT_NAMES.get(name, name) for name in deps if IMPORT_NAMES.get(name, name)
    ]


@pytest.mark.parametrize("package", _module_packages())
def test_module_python_packages_importable(package):
    """Python packages used by rule scripts must also be importable in unit tests."""
    importlib.import_module(package)
