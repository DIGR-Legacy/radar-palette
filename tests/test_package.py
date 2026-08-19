"""Packaging-level checks: the distribution installs and imports cleanly."""

from __future__ import annotations

import importlib

import pytest

SUBPACKAGES = [
    "radar_palette.advection",
    "radar_palette.gridding",
    "radar_palette.io",
    "radar_palette.testing",
    "radar_palette.util",
]


def test_version_is_a_string():
    import radar_palette

    assert isinstance(radar_palette.__version__, str)
    assert radar_palette.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} is missing a module docstring"
    assert isinstance(module.__all__, list)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_public_names_are_exported(name):
    """Everything advertised in ``__all__`` must actually be importable."""
    module = importlib.import_module(name)
    missing = [attr for attr in module.__all__ if not hasattr(module, attr)]
    assert not missing, f"{name}.__all__ advertises missing names: {missing}"


def test_top_level_all_is_importable():
    import radar_palette

    missing = [a for a in radar_palette.__all__ if not hasattr(radar_palette, a)]
    assert not missing
