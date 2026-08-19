"""Advection-aware time interpolation of radar volumes.

Scope
-----
Given two radar volumes bracketing a target time, estimate the echo motion
field between them and reconstruct the volume at the target time.  Two
pieces make that up, and they are deliberately separable:

1. **Motion estimation** on gridded reflectivity (dense optical flow,
   estimated per height level because storm motion is height dependent).
2. **Reconstruction** on the radar's *native* geometry: for each target
   gate, advect its Cartesian position back to the first volume and forward
   to the second, convert both to native (azimuth, slant range) for the same
   sweep, sample, and blend.

Conventions
-----------
Displacement is reported as the *physical* echo displacement in metres from
the first volume to the second, so ``velocity = displacement / dt`` with no
sign flip.  Optical-flow implementations that return a reference-to-moving
warp field have that sign inversion applied internally.

Reflectivity is interpolated in dBZ, not linear Z.

Status
------
Time reconstruction (:mod:`radar_palette.advection.timing`) is implemented. The
motion-estimation and reconstruction operators land in follow-up pull requests.
"""

from __future__ import annotations

from radar_palette.advection.timing import (
    apply_interpolated_time,
    interpolate_ray_times,
    volume_reference_time,
)

__all__ = [
    "apply_interpolated_time",
    "interpolate_ray_times",
    "volume_reference_time",
]
