"""Spectral (FFT-based) gridding of radar sweeps onto Cartesian lattices.

Scope
-----
A radar sweep is sampled on a polar (azimuth, range) lattice that is very
nearly uniform and, in azimuth, exactly periodic.  That structure admits a
band-limited resampling operator rather than a local weighted average:

1. **Geometry census** classifies a sweep as uniform-and-periodic,
   non-uniform, or degenerate, and reports the azimuth/range sampling
   statistics the choice of operator depends on.
2. **Horizontal operator** resamples each tilt with an exact Fourier series
   where the census permits it, and a non-uniform FFT (Kaiser-Bessel
   gridding kernel) otherwise.
3. **Vertical operator** combines tilts in elevation.  This is *not*
   spectral: tilt spacing is coarse, non-uniform and non-periodic, so a
   spectral operator in elevation is not defensible.

Conventions
-----------
Spectral interpolation of reflectivity is performed in dBZ.  Interpolating
band-limited fields in linear Z drives large negative excursions (Gibbs
ringing) on real sweeps and is not supported.

Status
------
Scaffolding only: the implementation lands in follow-up pull requests.
"""

from __future__ import annotations

__all__: list[str] = []
