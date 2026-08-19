# Contributing to radar-palette

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
pytest
```

CI runs the same three on Python 3.10 through 3.13, plus a minimal-install job
that exercises the optional-dependency skip paths and a build job that runs
`twine check --strict`.

## Layout

`src/` layout, so the installed package is what gets tested — never the working
directory. Add a new capability as a module inside the subpackage it belongs to
(`advection`, `gridding`, `util`, `testing`) and export its public names from
that subpackage's `__init__.py` via `__all__`; `tests/test_package.py` asserts
that everything advertised there is importable.

## Conventions

- **Docstrings**: numpydoc, enforced by `ruff` (pydocstyle, `convention = "numpy"`).
- **Optional dependencies**: `scikit-image` and `finufft` are optional. Guard them
  at the call site with a clear error, and gate tests with the `requires_skimage`
  / `requires_finufft` markers in `tests/conftest.py` so a minimal install still
  collects a green suite.
- **Units in docstrings**: always. Metres, seconds, degrees, dBZ.
- **Reflectivity is interpolated in dBZ**, not linear Z.
- **Displacement is physical echo displacement** (first volume to second, metres);
  `velocity = displacement / dt`, no sign flip.
- **Validation**: an interpolation operator gets tested against a ground truth it
  did not see — either an analytic field from `radar_palette.testing`, or a
  held-out real volume. Report the baseline it is being compared against.
- **Tests that hit the network** get `@pytest.mark.network`; slow ones get
  `@pytest.mark.slow`. Neither runs in the default CI job.

## Commits and attribution

Work in this repository is collaborative between human and AI contributors, and
commits record that. Every commit and pull request carries trailers for all
contributors:

```
Co-authored-by: Scott Collis <scollis.acrf@gmail.com>
Co-authored-by: Claude <noreply@anthropic.com>
```

## Docs

```bash
make -C docs html          # builds with -W, so warnings are errors
```

`intersphinx` fetches inventories from python.org, numpy.org, scipy.org and the
Py-ART docs, which needs network access. To build offline without tripping `-W`:

```bash
RADAR_PALETTE_DOCS_OFFLINE=1 make -C docs html
```

## Releases

Versions come from git tags via `setuptools-scm`. Tag `vX.Y.Z` on `main`; there
is no version string to edit by hand.
