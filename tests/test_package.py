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


def test_declared_xarray_floor_provides_datatree():
    """``radar_palette.io`` needs ``xarray.DataTree``, added in xarray 2024.10.0.

    Declaring the floor is not the same as it holding at install time, and a
    resolver free to pick an older xarray would fail deep inside a conversion with
    an AttributeError rather than at install. This asserts the capability directly.
    """
    xarray = pytest.importorskip("xarray")
    assert hasattr(xarray, "DataTree")


def test_third_party_imports_are_declared_dependencies():
    """Every third-party package the library imports must be a declared dependency.

    Guards against the failure mode this test was written for: a module importing
    something that happens to be installed transitively, so it works in a dev
    environment and breaks for a user whose resolver made a different choice.
    """
    import ast
    import pathlib
    import sys

    # tomllib is stdlib from 3.11, which is this package's floor (see
    # requires-python). An explicit skip rather than a bare ModuleNotFoundError so
    # that lowering the floor again produces a readable result instead of a crash.
    tomllib = pytest.importorskip("tomllib", reason="tomllib is stdlib on 3.11+")

    project_root = pathlib.Path(__file__).resolve().parent.parent
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.is_file():  # installed without the source tree
        pytest.skip("pyproject.toml not available next to the test suite")

    declared_requirements = tomllib.loads(pyproject_path.read_text())["project"][
        "dependencies"
    ]
    declared_names = {
        requirement.split(">")[0].split("=")[0].split("[")[0].strip().lower()
        for requirement in declared_requirements
    }
    # Distribution names differ from import names for these.
    import_name_to_distribution = {"pyart": "arm_pyart", "netCDF4": "netcdf4"}

    imported_top_level_modules = set()
    for source_path in (project_root / "src" / "radar_palette").rglob("*.py"):
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_top_level_modules.update(
                    alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_top_level_modules.add(node.module.split(".")[0])

    undeclared = set()
    for module_name in imported_top_level_modules:
        if module_name == "radar_palette" or module_name in sys.stdlib_module_names:
            continue
        distribution_name = import_name_to_distribution.get(module_name, module_name)
        if distribution_name.lower() not in declared_names:
            undeclared.add(f"{module_name} (as {distribution_name})")

    assert not undeclared, (
        f"imported but not declared in dependencies: {sorted(undeclared)}"
    )
