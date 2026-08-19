"""Radar-palette: advective interpolation and spectral gridding for weather radar.

The package is organised around two independent capabilities that share a
common geometry and decibel-handling layer:

``radar_palette.advection``
    Time interpolation of radar volumes that accounts for echo motion.
    Motion is estimated from a pair of bracketing volumes (dense optical
    flow), then each gate of the target volume is reconstructed by a
    semi-Lagrangian advect-and-blend on the radar's native geometry.

``radar_palette.gridding``
    Spectral (Fourier) resampling of radar sweeps onto Cartesian lattices,
    using an exact Fourier series where the sweep geometry is uniform and
    periodic, and a non-uniform FFT otherwise.  A separate, non-spectral
    vertical operator handles the elevation dimension.

Both are research code intended for eventual upstreaming into Py-ART; the
public API here is not yet stable.
"""

from __future__ import annotations

from radar_palette import advection, gridding, testing, util

try:  # pragma: no cover - generated at build time by setuptools-scm
    from radar_palette._version import version as __version__
except ImportError:  # pragma: no cover - running from a source tree
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("radar-palette")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "advection",
    "gridding",
    "testing",
    "util",
]
