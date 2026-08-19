# radar-palette

Advective interpolation and spectral (FFT) gridding for weather radar volumes.

Both capabilities began as research code developed against ARM C-SAPR and
NEXRAD volumes, and are staged here as a package while the APIs settle.  The
intent is eventual upstreaming into [Py-ART](https://github.com/ARM-DOE/pyart);
until then this package is the place they live and get tested.

## What is in here

| Subpackage | Purpose |
| --- | --- |
| `radar_palette.advection` | Time interpolation between two radar volumes that accounts for echo motion: dense optical flow for the motion field, then a semi-Lagrangian advect-and-blend on the radar's native geometry. |
| `radar_palette.gridding` | Spectral resampling of radar sweeps onto Cartesian lattices — exact Fourier series where the sweep geometry is uniform and periodic, non-uniform FFT otherwise, with a separate non-spectral vertical operator. |
| `radar_palette.util` | Shared beam-geometry and decibel-handling helpers. |
| `radar_palette.testing` | Synthetic radar objects and analytic fields with known ground truth. |

**Status: pre-alpha.** This commit is the packaging scaffold; the modules are
documented but not yet implemented. Nothing here is API-stable.

## Install

From a clone, for development:

```bash
git clone https://github.com/DIGR-Legacy/radar-palette.git
cd radar-palette
pip install -e ".[dev]"
pre-commit install
```

The core install pulls `numpy`, `scipy` and `arm_pyart`. The heavier numerical
machinery is behind extras so a minimal install stays light:

| Extra | Pulls | Needed for |
| --- | --- | --- |
| `advection` | `scikit-image` | optical-flow motion estimation |
| `spectral` | `finufft` | non-uniform FFT gridding |
| `all` | both of the above | |
| `test`, `docs`, `dev` | tooling | development |

## Development

```bash
pytest              # test suite
ruff check .        # lint (includes import sort and numpydoc style)
ruff format .       # format
python -m build     # build sdist + wheel
```

Versioning is handled by `setuptools-scm` from git tags — there is no version
string to edit by hand. No release has been tagged yet, so a build from a clone
reports a development version derived from the commit (`0.1.dev<N>+g<sha>`); a
build from a source archive with no git metadata falls back to `0.0.0`.

## Conventions

Two that are load-bearing and easy to get wrong:

- **Reflectivity is interpolated in dBZ, not linear Z.** Band-limited
  interpolation in linear Z produces large negative excursions on real sweeps.
- **Displacement is the physical echo displacement**, first volume to second,
  in metres — so `velocity = displacement / dt`, with no sign flip. Optical-flow
  backends that return a reference-to-moving warp have that inversion applied
  internally.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests are co-authored; commits
carry `Co-authored-by` trailers for all contributors, human and otherwise.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
