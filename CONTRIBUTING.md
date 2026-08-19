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

CI runs the same three on Python 3.11 through 3.13, plus a minimal-install job
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
- **Geometry**: use `radar_palette.gridding.antenna_to_cartesian_43` and
  `cartesian_to_antenna_43`, not the Py-ART equivalents, wherever a round trip
  matters — Py-ART's inverse omits the curvature term (see `CHANGELOG.md`). Note
  Py-ART's forward transform takes kilometres while its inverse returns metres.
- **Object flavours**: public entry points accept both Py-ART and xradar objects
  and return the caller's family by default (`output_flavor` overrides). Route
  conversions through `radar_palette.io` and Py-ART's own interoperability layer
  (`pyart.xradar`, `Grid.to_xarray`, xradar's cfradial readers) — do not hand-roll
  a mapping between the two data models.
- **Reflectivity is interpolated in dBZ**, not linear Z.
- **An interpolated volume carries the time it represents**, never the time of a
  volume it was derived from. Reconstruct times via
  `radar_palette.advection.timing`; do not let a `deepcopy` of a bracketing
  volume carry its clock through to the output.
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

If a local build reports `0.0.0`, `setuptools-scm` could not derive a version and
fell back to `fallback_version` — expected while no release tag exists, and also
seen in sandboxed or unusual checkouts where it declines to inspect the
repository. Check with:

```bash
python -c "import setuptools_scm, os; print(setuptools_scm.get_version(root=os.getcwd()))"
```

CI builds with `fetch-depth: 0` so tags are available there.
