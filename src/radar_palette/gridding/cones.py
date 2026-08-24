"""The cone surface of a fixed-elevation sweep, and the lattice it is gridded onto.

A sweep at a fixed elevation angle does not sample a horizontal plane, nor even a
tilted one: on a curved earth the beam climbs faster than a straight line, so the
locus of samples is a **curved cone**. Every consequence in this module follows from
that one fact.

Why it matters for assembling a volume
--------------------------------------
Because the surface is curved, the height of a tilt varies from column to column, and
so does the vertical distance to the tilt above it. On C-SAPR the 0.75-to-1.20 degree
gap is about 0.16 km at 20 km arc and about 0.9 km at 100 km. Any vertical
interpolation therefore has to work per column against locally-evaluated heights; a
single nominal height per tilt is wrong everywhere except at one range.

Geometry source
---------------
Heights and slant ranges come from :func:`cone_range_height`, which is the exact
inverse of :func:`radar_palette.gridding.antenna_to_cartesian_43` at fixed elevation.
Py-ART's own inverse is not used here: it omits the curvature term, which is the very
thing that makes a tilt a cone rather than a plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from radar_palette.gridding.evaluator import SweepSpectralEvaluator
from radar_palette.gridding.fill import fill_sweep
from radar_palette.gridding.geometry import EARTH_RADIUS_EFFECTIVE_M

__all__ = [
    "ConeStack",
    "TiltReport",
    "beam_footprint_crossover",
    "build_cones",
    "cone_range_height",
    "dedup_sweeps",
    "target_lattice",
]


def cone_range_height(arc_length_m, elevation_deg):
    """Slant range and height on a fixed-elevation cone at a given arc length.

    Parameters
    ----------
    arc_length_m : array_like
        Great-circle arc length from the radar, in metres.
    elevation_deg : float
        Beam elevation angle of the sweep, in degrees.

    Returns
    -------
    slant_range_m, height_m : numpy.ndarray
        Slant range along the beam and height above the radar, both in metres.

    Notes
    -----
    Exact inverse of :func:`radar_palette.gridding.antenna_to_cartesian_43` at fixed
    elevation, on the 4/3 effective earth. Note that even at 0 degrees elevation the
    height is **not** zero: the earth curves away beneath the beam, so a horizontal
    ray rises as ``arc**2 / (2 * R_effective)`` to first order.
    """
    arc_length_m = np.asarray(arc_length_m, dtype=np.float64)
    earth_radius = EARTH_RADIUS_EFFECTIVE_M
    central_angle = arc_length_m / earth_radius
    tangent_elevation = np.tan(np.deg2rad(elevation_deg))

    radial_distance = earth_radius / (
        np.cos(central_angle) - tangent_elevation * np.sin(central_angle)
    )
    slant_range_m = (
        radial_distance * np.sin(central_angle) / np.cos(np.deg2rad(elevation_deg))
    )
    return slant_range_m, radial_distance - earth_radius


def beam_footprint_crossover(beamwidth_deg, spacing_m):
    """Arc length at which the cross-beam footprint equals the grid spacing.

    Inside this distance the grid is finer than the measurement it represents;
    outside it, adjacent grid cells are not independent samples. Useful for stating
    honestly where a gridded product stops carrying independent information.

    Parameters
    ----------
    beamwidth_deg : float
        Antenna beamwidth in degrees.
    spacing_m : float
        Cartesian grid spacing in metres.

    Returns
    -------
    float
        Arc length in metres at which ``arc * beamwidth == spacing``.
    """
    return float(spacing_m / np.deg2rad(beamwidth_deg))


def target_lattice(half_width_m, spacing_m, elevation_deg, range_first_m, range_max_m):
    """Build the common horizontal lattice for one tilt, with its cone geometry.

    Parameters
    ----------
    half_width_m : float
        Half-width of the square domain, in metres.
    spacing_m : float
        Grid spacing in metres.
    elevation_deg : float
        Elevation of the tilt whose cone geometry is wanted.
    range_first_m, range_max_m : float
        First and last measured slant range of the sweep. Cells outside this span
        are marked invalid.

    Returns
    -------
    dict
        ``x``, ``y`` (1-D axes); ``X``, ``Y``, ``arc_m``, ``slant_range_m``,
        ``height_m``, ``azimuth_deg`` (2-D); ``valid`` (2-D bool); plus
        ``spacing_m``, ``n`` and ``elevation_deg``.

    Notes
    -----
    Axis order is **row is y, column is x**, matching the ``(nz, ny, nx)`` convention
    of :func:`pyart.map.map_gates_to_grid`. Transposing this does not announce itself
    on a roughly isotropic field: it produces an error equal to the field's own
    standard deviation rather than a visible flip.

    The ``valid`` mask excludes the near-radar hole for a reason beyond bookkeeping.
    The range axis is mirror-extended before the FFT, and the mirror is *even* about
    the first gate, so it reflects the echo into the hole and fabricates a plausible
    near-radar return. Masking is what keeps that reflection out of the product.
    """
    cell_count = int(round(2.0 * half_width_m / spacing_m)) + 1
    axis_m = np.linspace(-half_width_m, half_width_m, cell_count)
    east_m, north_m = np.meshgrid(axis_m, axis_m, indexing="xy")

    arc_m = np.hypot(east_m, north_m)
    slant_range_m, height_m = cone_range_height(arc_m, elevation_deg)
    azimuth_deg = np.rad2deg(np.arctan2(east_m, north_m)) % 360.0
    valid = (slant_range_m >= range_first_m) & (slant_range_m <= range_max_m)

    return {
        "x": axis_m,
        "y": axis_m,
        "X": east_m,
        "Y": north_m,
        "arc_m": arc_m,
        "slant_range_m": slant_range_m,
        "height_m": height_m,
        "azimuth_deg": azimuth_deg,
        "valid": valid,
        "spacing_m": float(spacing_m),
        "n": cell_count,
        "elevation_deg": float(elevation_deg),
    }


def dedup_sweeps(geometries, range_attribute="range_max_valid_m"):
    """Collapse split cuts so each unique elevation contributes exactly once.

    A WSR-88D split cut pairs a long surveillance sweep with a short Doppler sweep at
    the same fixed angle. For reflectivity the long one is wanted, so the member whose
    field reaches furthest in range is kept and ties fall back to the lower sweep
    index.

    Stating the rule in terms of *measured* range rather than cut naming means it
    needs no knowledge of any particular radar's scan-strategy vocabulary.

    Parameters
    ----------
    geometries : list of ~radar_palette.gridding.census.SweepGeometry
        Census of every sweep, from :func:`radar_palette.gridding.census_radar`.
    range_attribute : str, optional
        Attribute compared when choosing between members of a split-cut group.

    Returns
    -------
    list of ~radar_palette.gridding.census.SweepGeometry
        One entry per unique elevation, sorted by ascending fixed angle. Each keeps
        its ``split_cut_size``, so a caller can see that sweeps were dropped.
    """
    by_group = {}
    for geometry in geometries:
        by_group.setdefault(geometry.split_cut_group, []).append(geometry)

    kept = []
    for group in sorted(by_group, key=lambda key: (key is None, key)):
        members = sorted(
            by_group[group],
            key=lambda geometry: (
                -float(getattr(geometry, range_attribute)),
                geometry.sweep,
            ),
        )
        kept.append(members[0])
    return sorted(kept, key=lambda geometry: geometry.fixed_angle)


@dataclass
class TiltReport:
    """Diagnostics for one gridded tilt.

    Fields whose meaning is not obvious from the name:

    ``overshoot_db``
        How far the gridded field strays beyond the range of the data it came from,
        as the larger of the two one-sided excursions. A between-sample metric: a
        check on ray positions alone is blind to an artefact that appears *between*
        rays, which is where band-limited ringing lives.
    ``valid_fraction``
        Fraction of lattice cells inside the sweep's measured range span.
    """

    tilt_index: int
    sweep: int
    fixed_angle: float
    sweep_class: str
    azimuth_path: str
    nrays: int
    masked_input_fraction: float
    valid_fraction: float
    height_min_m: float
    height_max_m: float
    data_min_dbz: float
    data_max_dbz: float
    grid_min_dbz: float
    grid_max_dbz: float
    overshoot_db: float
    split_cut_size: int


@dataclass
class ConeStack:
    """A volume's sweeps gridded onto one shared horizontal lattice.

    ``reflectivity``, ``height_m`` and ``valid`` are all ``(n_tilts, ny, nx)`` and
    index-aligned: cone ``k`` has values ``reflectivity[k]`` on the surface
    ``height_m[k]``, meaningful only where ``valid[k]``. Tilts are ordered by
    ascending elevation, which the vertical search relies on.
    """

    reflectivity: np.ndarray
    height_m: np.ndarray
    valid: np.ndarray
    fixed_angle: np.ndarray
    sweep: np.ndarray
    x: np.ndarray
    y: np.ndarray
    reports: list = field(default_factory=list)


def build_cones(
    radar,
    geometries,
    half_width_m,
    spacing_m,
    field_name="reflectivity",
    band_frac=1.0,
    n_cg=None,
    az_engine=None,
    az_solver=None,
    az_ridge=None,
    fill_method="edge",
    up_az=4,
    up_r=4,
):
    """Spectrally grid every sweep onto one shared horizontal lattice.

    Parameters
    ----------
    radar : pyart.core.Radar
        Volume to grid.
    geometries : list of ~radar_palette.gridding.census.SweepGeometry
        Sweeps to grid, normally the output of :func:`dedup_sweeps`.
    half_width_m, spacing_m : float
        Domain half-width and grid spacing, in metres.
    field_name : str, optional
        Field to grid. Must be in dBZ; see
        :mod:`radar_palette.gridding.reflectivity`.
    az_engine, az_solver, az_ridge : optional
        Azimuth NUFFT backend, the solver on top of it, and that solver's
        regularisation, forwarded to each
        :class:`~radar_palette.gridding.evaluator.SweepSpectralEvaluator`. Left
        unset by default so the evaluator's own defaults remain the single source
        of truth. These are the consequential arguments for spectral gridding
        cost: one evaluator is built per sweep and that build dominates, so the
        total is nearly flat in output resolution and is set here.
    band_frac, n_cg : optional
        Passed to :class:`~radar_palette.gridding.SweepSpectralEvaluator`. ``n_cg``
        defaults to the evaluator's own default.
    fill_method : str, optional
        Gap-fill strategy applied before evaluation; see
        :func:`radar_palette.gridding.fill_sweep`.
    up_az, up_r : int, optional
        Upsampling factors for the evaluator's fast path.

    Returns
    -------
    ConeStack

    Notes
    -----
    Tilts are gridded one at a time and each evaluator is released before the next is
    built. The extended, upsampled lattice for a single wide sweep can reach the order
    of a gigabyte, so a volume's worth of evaluators cannot coexist in memory.
    """
    cones, heights, masks, reports = [], [], [], []
    axes = None

    for tilt_index, geometry in enumerate(geometries):
        start, end = radar.get_start_end(geometry.sweep)
        azimuths = np.asarray(radar.azimuth["data"][start : end + 1], dtype=np.float64)
        observed = radar.fields[field_name]["data"][start : end + 1]
        filled, input_mask = fill_sweep(observed, method=fill_method, return_mask=True)

        lattice = target_lattice(
            half_width_m,
            spacing_m,
            geometry.fixed_angle,
            geometry.range_first_m,
            geometry.range_max_valid_m,
        )
        if axes is None:
            axes = (lattice["x"], lattice["y"])

        evaluator_kwargs = {"band_frac": band_frac}
        if n_cg is not None:
            evaluator_kwargs["n_cg"] = n_cg
        # Left unset rather than defaulted here, so the evaluator's own defaults
        # stay the single source of truth for what an unconfigured call does.
        if az_engine is not None:
            evaluator_kwargs["az_engine"] = az_engine
        if az_solver is not None:
            evaluator_kwargs["az_solver"] = az_solver
        if az_ridge is not None:
            evaluator_kwargs["az_ridge"] = az_ridge
        evaluator = SweepSpectralEvaluator(
            filled, geometry, azimuths, **evaluator_kwargs
        )

        in_range = lattice["valid"]
        gridded = np.full(lattice["X"].shape, np.nan)
        gridded[in_range] = evaluator.evaluate_fast(
            lattice["slant_range_m"][in_range],
            lattice["azimuth_deg"][in_range],
            up_az=up_az,
            up_r=up_r,
        )

        cones.append(np.where(in_range, gridded, np.nan).astype(np.float32))
        heights.append(
            np.where(in_range, lattice["height_m"], np.nan).astype(np.float32)
        )
        masks.append(in_range.copy())

        measured = filled[~input_mask] if (~input_mask).any() else np.array([np.nan])
        data_min, data_max = float(np.nanmin(measured)), float(np.nanmax(measured))
        grid_min, grid_max = float(np.nanmin(gridded)), float(np.nanmax(gridded))
        reports.append(
            TiltReport(
                tilt_index=tilt_index,
                sweep=int(geometry.sweep),
                fixed_angle=float(geometry.fixed_angle),
                sweep_class=str(geometry.sweep_class),
                azimuth_path=evaluator.report.az_path,
                nrays=int(geometry.nrays),
                masked_input_fraction=float(input_mask.mean()),
                valid_fraction=float(in_range.mean()),
                height_min_m=float(np.nanmin(lattice["height_m"][in_range])),
                height_max_m=float(np.nanmax(lattice["height_m"][in_range])),
                data_min_dbz=data_min,
                data_max_dbz=data_max,
                grid_min_dbz=grid_min,
                grid_max_dbz=grid_max,
                overshoot_db=float(max(grid_max - data_max, data_min - grid_min)),
                split_cut_size=int(geometry.split_cut_size),
            )
        )
        del evaluator, gridded, lattice, filled, input_mask

    return ConeStack(
        reflectivity=np.stack(cones),
        height_m=np.stack(heights),
        valid=np.stack(masks),
        fixed_angle=np.array([g.fixed_angle for g in geometries], dtype=np.float64),
        sweep=np.array([g.sweep for g in geometries], dtype=int),
        x=axes[0],
        y=axes[1],
        reports=reports,
    )
