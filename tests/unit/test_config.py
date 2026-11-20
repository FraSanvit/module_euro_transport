"""Tests for the example configuration."""

from pathlib import Path

import jsonschema
import yaml

MODULE_ROOT = Path(__file__).parents[2]


def _load(path):
    return yaml.safe_load((MODULE_ROOT / path).read_text())


def test_example_config_is_valid():
    """The example config passes the module's config schema."""
    jsonschema.validate(
        _load("config/config.yaml"), _load("workflow/internal/config.schema.yaml")
    )


def test_example_config_covers_all_countries():
    """The example config runs the full case: every supported country."""
    countries = _load("config/config.yaml")["countries"]
    allowed = _load("workflow/internal/config.schema.yaml")["properties"]["countries"][
        "items"
    ]["enum"]
    assert sorted(countries) == sorted(allowed)
