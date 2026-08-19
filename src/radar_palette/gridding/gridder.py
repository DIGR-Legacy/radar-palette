"""Flavour-aware entry point for gridding radar volumes onto a Cartesian lattice.

Accepts volumes in either supported object family and returns Cartesian output in
the family that matches: a Py-ART ``Radar`` grids to a :class:`pyart.core.Grid`,
an xradar ``DataTree`` grids to an :class:`xarray.Dataset`. Either default can be
overridden per call.

The spectral operator that will back this entry point
(:mod:`radar_palette.gridding`) is not yet ported. Until it is, gridding is
performed by :func:`pyart.map.grid_from_radars`, so the flavour contract and its
tests are settled before the operator lands and can be swapped in behind an
unchanged public signature.
"""

from __future__ import annotations

from radar_palette.io import (
    GridFlavor,
    RadarFlavor,
    detect_radar_flavor,
    resolve_output_flavor,
    to_grid_flavor,
    to_pyart_radar,
)

__all__ = ["grid_volume"]

_RADAR_TO_GRID_FLAVOR = {
    RadarFlavor.PYART: GridFlavor.PYART,
    RadarFlavor.XRADAR: GridFlavor.XARRAY,
}


def grid_volume(
    volumes, grid_shape, grid_limits, output_flavor=None, **gridding_kwargs
):
    """Grid one or more radar volumes onto a Cartesian lattice.

    Parameters
    ----------
    volumes : pyart.core.Radar or xarray.DataTree or sequence of these
        Volume, or volumes, to grid. A sequence may mix object families; the first
        entry determines the default output family.
    grid_shape : tuple of int
        Lattice shape as ``(nz, ny, nx)``.
    grid_limits : tuple of tuple of float
        Lattice extent in metres as ``((z_min, z_max), (y_min, y_max),
        (x_min, x_max))``.
    output_flavor : {'pyart', 'xarray'} or GridFlavor, optional
        Object family of the returned grid. Defaults to the family matching the
        input: Py-ART volumes give a ``Grid``, xradar volumes give a ``Dataset``.
        ``'xradar'`` is accepted as an alias for ``'xarray'``.
    **gridding_kwargs
        Passed through to the underlying gridding operator.

    Returns
    -------
    pyart.core.Grid or xarray.Dataset
        Gridded output in the requested family.

    Raises
    ------
    ValueError
        If ``volumes`` is empty, or ``output_flavor`` names an unknown flavour.

    Notes
    -----
    Currently gridded by :func:`pyart.map.grid_from_radars`. The spectral operator
    will replace that internally without changing this signature.
    """
    import pyart

    if not isinstance(volumes, (list, tuple)):
        volumes = (volumes,)
    if len(volumes) == 0:
        raise ValueError("grid_volume requires at least one radar volume")

    input_radar_flavor = detect_radar_flavor(volumes[0])
    output_flavor = resolve_output_flavor(
        output_flavor, _RADAR_TO_GRID_FLAVOR[input_radar_flavor]
    )

    radar_surfaces = tuple(to_pyart_radar(volume)[0] for volume in volumes)
    grid = pyart.map.grid_from_radars(
        radar_surfaces,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        **gridding_kwargs,
    )
    return to_grid_flavor(grid, output_flavor)
