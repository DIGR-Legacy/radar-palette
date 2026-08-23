"""Tests for method dispatch on the public gridding entry point.

:func:`radar_palette.gridding.grid_volume` now has two backings: Py-ART's Cressman
gridder and this package's spectral operator. The tests are grouped by the thing that
can independently be wrong:

``TestDefaultIsUnchanged``
    That adding the option did not change what existing callers get. This is the
    reason the default is ``"pyart"`` rather than ``"spectral"``.
``TestSpectralPath``
    That the spectral backing actually runs end to end and produces a grid of the
    right shape, in either object family.
``TestSpectralRefusals``
    That the spectral path fails loudly where it is undefined, rather than quietly
    producing something plausible.
``TestMethodsDiffer``
    That the two backings are genuinely different operators --- if they agreed
    everywhere, one of them would not be doing what it claims.
"""

from __future__ import annotations

import numpy as np
import pytest

pyart = pytest.importorskip("pyart")

from radar_palette.gridding import (  # noqa: E402
    DEFAULT_GRIDDING_METHOD,
    GRIDDING_METHODS,
    VerticalFlag,
    grid_volume,
)

GRID_SHAPE = (4, 41, 41)
GRID_LIMITS = ((500.0, 6000.0), (-50_000.0, 50_000.0), (-50_000.0, 50_000.0))


def make_volume(fixed_angles=(0.5, 1.0, 1.5, 2.5, 4.0), nrays=240, ngates=100):
    """A volume whose reflectivity is a smooth function of gate height."""
    fixed_angles = np.asarray(fixed_angles, dtype="float64")
    radar = pyart.testing.make_empty_ppi_radar(ngates, nrays, fixed_angles.size)
    radar.range["data"] = 1000.0 + 1000.0 * np.arange(ngates, dtype="float64")
    azimuths = (360.0 / nrays) * np.arange(nrays, dtype="float64")
    radar.azimuth["data"] = np.tile(azimuths, fixed_angles.size)
    radar.elevation["data"] = np.repeat(fixed_angles, nrays)
    radar.fixed_angle["data"] = fixed_angles
    radar.time["data"] = 0.5 * np.arange(radar.nrays, dtype="float64")
    radar.time["units"] = "seconds since 2011-05-20T11:27:34Z"
    values = 30.0 - 3.0e-3 * radar.gate_z["data"]
    radar.add_field(
        "reflectivity",
        {
            "data": np.ma.masked_invalid(values.astype("float32")),
            "units": "dBZ",
            "_FillValue": -9999.0,
        },
        replace_existing=True,
    )
    return radar


@pytest.fixture(scope="module")
def volume():
    return make_volume()


class TestMethodRegistry:
    def test_the_default_is_a_known_method(self):
        assert DEFAULT_GRIDDING_METHOD in GRIDDING_METHODS

    def test_both_backings_are_registered(self):
        assert set(GRIDDING_METHODS) == {"pyart", "spectral"}

    def test_the_default_preserves_previous_behaviour(self):
        """Deliberate: adding the spectral operator must not silently change results.

        The spectral path is opt-in because it is a different operator, not a
        strictly better one --- it resolves sharp gradients that Cressman weighting
        smooths away, at the cost of ringing at hard echo edges.
        """
        assert DEFAULT_GRIDDING_METHOD == "pyart"

    def test_rejects_an_unknown_method(self, volume):
        with pytest.raises(ValueError, match="unknown gridding method"):
            grid_volume(
                volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="magic"
            )


class TestDefaultIsUnchanged:
    def test_default_matches_an_explicit_pyart_request(self, volume):
        implicit = grid_volume(volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS)
        explicit = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="pyart"
        )
        np.testing.assert_allclose(
            np.ma.filled(implicit.fields["reflectivity"]["data"], np.nan),
            np.ma.filled(explicit.fields["reflectivity"]["data"], np.nan),
            equal_nan=True,
        )

    def test_default_matches_calling_pyart_directly(self, volume):
        """The entry point must add nothing of its own on the default path."""
        through_entry_point = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS
        )
        direct = pyart.map.grid_from_radars(
            (volume,), grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS
        )
        np.testing.assert_allclose(
            np.ma.filled(through_entry_point.fields["reflectivity"]["data"], np.nan),
            np.ma.filled(direct.fields["reflectivity"]["data"], np.nan),
            equal_nan=True,
        )


class TestSpectralPath:
    def test_returns_a_grid_of_the_requested_shape(self, volume):
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        assert grid.fields["reflectivity"]["data"].shape == GRID_SHAPE

    def test_axes_span_the_requested_limits(self, volume):
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        for axis, (lower, upper) in zip(
            (grid.z, grid.y, grid.x), GRID_LIMITS, strict=True
        ):
            assert axis["data"][0] == pytest.approx(lower)
            assert axis["data"][-1] == pytest.approx(upper)

    def test_carries_the_coverage_flag_as_a_field(self, volume):
        """The flag is the honest part of the product and must survive to the output.

        A spectral grid cell is NaN wherever the volume did not observe it, and the
        flag says which of the several reasons applies. Dropping it would leave a
        user unable to tell a data gap from a coverage limit.
        """
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        assert "coverage_flag" in grid.fields
        flags = np.ma.filled(grid.fields["coverage_flag"]["data"], 255).astype(int)
        assert set(np.unique(flags)) <= {int(f) for f in VerticalFlag} | {255}

    def test_finite_values_coincide_with_the_interpolated_flag(self, volume):
        """The same invariant the vertical layer guarantees, preserved end to end."""
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        values = np.ma.filled(grid.fields["reflectivity"]["data"], np.nan)
        flags = np.ma.filled(grid.fields["coverage_flag"]["data"], 255).astype(int)
        np.testing.assert_array_equal(
            np.isfinite(values), flags == int(VerticalFlag.INTERPOLATED)
        )

    def test_values_track_the_height_dependent_input(self, volume):
        """The test field falls 3 dB per km, so the grid must too."""
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        values = np.ma.filled(grid.fields["reflectivity"]["data"], np.nan)
        heights_m = grid.z["data"]
        level_means = [np.nanmean(values[k]) for k in range(values.shape[0])]
        finite = [m for m in level_means if np.isfinite(m)]
        assert len(finite) >= 2
        assert finite[0] > finite[-1]
        expected_drop = 3.0e-3 * (heights_m[-1] - heights_m[0])
        assert abs(finite[0] - finite[-1]) < 3.0 * expected_drop

    def test_accepts_a_vertical_scheme(self, volume):
        grid = grid_volume(
            volume,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            method="spectral",
            scheme="linear_z",
        )
        assert grid.fields["reflectivity"]["data"].shape == GRID_SHAPE

    def test_preserves_the_grid_origin_of_the_input(self, volume):
        spectral = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        cressman = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="pyart"
        )
        assert spectral.origin_latitude["data"] == pytest.approx(
            cressman.origin_latitude["data"]
        )
        assert spectral.origin_longitude["data"] == pytest.approx(
            cressman.origin_longitude["data"]
        )

    def test_records_the_method_in_the_metadata(self, volume):
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        assert "spectral" in grid.metadata.get("history", "")

    def test_returns_an_xarray_dataset_on_request(self, volume):
        import xarray as xr

        result = grid_volume(
            volume,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            method="spectral",
            output_flavor="xarray",
        )
        assert isinstance(result, xr.Dataset)

    def test_flavour_default_still_mirrors_the_input(self, volume):
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        assert isinstance(grid, pyart.core.Grid)


class TestSpectralRefusals:
    """Where the spectral path is undefined it must say so, not improvise."""

    def test_refuses_a_single_tilt_volume(self):
        """Vertical interpolation needs two cones to bracket a target.

        A single sweep cannot produce a 3-D field by this method at all, so the
        failure belongs at the entry point rather than as an all-NaN grid.
        """
        single = make_volume(fixed_angles=(0.5,))
        with pytest.raises(ValueError, match="at least two"):
            grid_volume(
                single,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
            )

    def test_refuses_multiple_volumes(self, volume):
        """Merging several volumes spectrally is not defined by this operator.

        The Cressman path accepts a sequence because its weighting composes across
        radars. The spectral path works per sweep of one volume, and inventing a
        merge rule here would be a research decision disguised as plumbing.
        """
        with pytest.raises(NotImplementedError, match="single volume"):
            grid_volume(
                (volume, volume),
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
            )

    def test_refuses_an_unknown_vertical_scheme(self, volume):
        with pytest.raises(ValueError, match="unknown"):
            grid_volume(
                volume,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
                scheme="quintic",
            )

    def test_still_refuses_an_empty_sequence(self, volume):
        with pytest.raises(ValueError, match="at least one"):
            grid_volume((), grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS)


class TestMethodsDiffer:
    def test_the_two_backings_do_not_agree_everywhere(self, volume):
        """If they matched, one would not be doing what it claims.

        Cressman weighting averages neighbouring gates into each cell; the spectral
        operator evaluates a band-limited interpolant. They answer differently by
        construction, and this test exists so that a mis-wired dispatch --- silently
        running Py-ART for both --- cannot pass.
        """
        spectral = np.ma.filled(
            grid_volume(
                volume,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
            ).fields["reflectivity"]["data"],
            np.nan,
        )
        cressman = np.ma.filled(
            grid_volume(
                volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="pyart"
            ).fields["reflectivity"]["data"],
            np.nan,
        )
        both_finite = np.isfinite(spectral) & np.isfinite(cressman)
        assert both_finite.sum() > 100
        assert not np.allclose(spectral[both_finite], cressman[both_finite])

    def test_but_they_broadly_agree_where_both_have_data(self, volume):
        """Different operators, same physics: the two must not disagree wildly.

        A large disagreement here would mean one of them is wrong, not merely
        different, so this is the sanity bound that makes the previous test
        meaningful rather than a licence for anything.
        """
        spectral = np.ma.filled(
            grid_volume(
                volume,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
            ).fields["reflectivity"]["data"],
            np.nan,
        )
        cressman = np.ma.filled(
            grid_volume(
                volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="pyart"
            ).fields["reflectivity"]["data"],
            np.nan,
        )
        both_finite = np.isfinite(spectral) & np.isfinite(cressman)
        assert np.nanmedian(np.abs(spectral[both_finite] - cressman[both_finite])) < 5.0


class TestWhyTheDefaultIsPyart:
    """The measured trade-off behind ``DEFAULT_GRIDDING_METHOD``.

    Neither operator dominates, which is the whole reason the older one stays the
    default. On a hard azimuthal echo edge (a 10 to 50 dBZ wedge) the spectral
    interpolant overshoots to 55.5 dBZ and undershoots to 4.7 --- about +5.5 and -5.3
    dB beyond the data --- while Cressman weighting stays exactly inside [10, 50]
    because averaging cannot exceed its inputs.

    Switching the default would hand every existing caller that ringing in exchange
    for sharpness they did not ask for. These tests make the trade-off a fact in the
    suite rather than a claim in a docstring.
    """

    WEDGE_LOW_DBZ = 10.0
    WEDGE_HIGH_DBZ = 50.0
    EDGE_GRID_SHAPE = (3, 81, 81)
    EDGE_GRID_LIMITS = ((500.0, 4000.0), (-40_000.0, 40_000.0), (-40_000.0, 40_000.0))

    @pytest.fixture(scope="class")
    def hard_edge_volume(self):
        """A wedge of strong echo with hard edges in azimuth."""
        radar = make_volume(nrays=360, ngates=100)
        azimuth_of_gate = (
            np.rad2deg(np.arctan2(radar.gate_x["data"], radar.gate_y["data"])) % 360.0
        )
        in_wedge = (azimuth_of_gate > 60.0) & (azimuth_of_gate < 120.0)
        values = np.where(in_wedge, self.WEDGE_HIGH_DBZ, self.WEDGE_LOW_DBZ)
        radar.add_field(
            "reflectivity",
            {
                "data": np.ma.masked_invalid(values.astype("float32")),
                "units": "dBZ",
                "_FillValue": -9999.0,
            },
            replace_existing=True,
        )
        return radar

    def gridded(self, radar, method):
        grid = grid_volume(
            radar,
            grid_shape=self.EDGE_GRID_SHAPE,
            grid_limits=self.EDGE_GRID_LIMITS,
            method=method,
        )
        return np.ma.filled(grid.fields["reflectivity"]["data"], np.nan)

    def test_the_spectral_path_rings_at_a_hard_edge(self, hard_edge_volume):
        values = self.gridded(hard_edge_volume, "spectral")
        assert np.nanmax(values) > self.WEDGE_HIGH_DBZ + 1.0
        assert np.nanmin(values) < self.WEDGE_LOW_DBZ - 1.0

    # Fields are stored float32, so an exact bound would test the storage dtype
    # rather than the operator: distance weighting overshoots by ~8e-6 dB on this
    # scene purely from rounding. A 0.01 dB tolerance is far below the ~5 dB ringing
    # being distinguished, so the test still separates the two operators cleanly.
    ROUNDING_TOLERANCE_DB = 0.01

    def test_the_default_path_does_not_ring(self, hard_edge_volume):
        """Distance weighting is an average, so it cannot exceed its inputs."""
        values = self.gridded(hard_edge_volume, "pyart")
        assert np.nanmax(values) <= self.WEDGE_HIGH_DBZ + self.ROUNDING_TOLERANCE_DB
        assert np.nanmin(values) >= self.WEDGE_LOW_DBZ - self.ROUNDING_TOLERANCE_DB

    def test_the_default_is_the_non_ringing_one(self, hard_edge_volume):
        """Ties the constant to the measurement rather than to a comment.

        If someone flips DEFAULT_GRIDDING_METHOD to "spectral", this fails and says
        why: the default would then overshoot the data range on a hard edge.
        """
        values = self.gridded(hard_edge_volume, DEFAULT_GRIDDING_METHOD)
        assert np.nanmax(values) <= self.WEDGE_HIGH_DBZ + self.ROUNDING_TOLERANCE_DB
        assert np.nanmin(values) >= self.WEDGE_LOW_DBZ - self.ROUNDING_TOLERANCE_DB


class TestDistanceWeightingLeavesInterConeGaps:
    """The other half of the trade-off, and the half that is easy to miss.

    :class:`TestWhyTheDefaultIsPyart` measures what the spectral path costs: it rings
    at a discontinuity. This class measures what the *default* path costs, which the
    module docstring described only as degrading "gracefully".

    It does not degrade gracefully. A radius of influence is a fixed length, while the
    vertical separation between adjacent tilts grows with range. Wherever the
    separation exceeds the radius no gate is within reach, and the cell is left empty
    --- an interior hole, with valid data both above and below it in the same column.

    This is not a mis-set parameter: Py-ART's own defaults for ``roi_func="dist_beam"``
    produce it, which is why it earns a test rather than a comment.
    """

    GRID_SHAPE = (24, 41, 41)
    GRID_LIMITS = ((500.0, 12_000.0), (-30_000.0, 30_000.0), (-30_000.0, 30_000.0))

    @staticmethod
    def _interior_holes(section):
        """Count empty cells that have data both above and below in the same column.

        A gap at the top or bottom of a column is a coverage limit and is expected. A
        gap *between* two valid samples is the operator failing to reach.
        """
        total = 0
        for column in range(section.shape[1]):
            valid = np.isfinite(section[:, column])
            if valid.sum() < 2:
                continue
            first = int(np.argmax(valid))
            last = len(valid) - 1 - int(np.argmax(valid[::-1]))
            total += int((~valid[first : last + 1]).sum())
        return total

    @pytest.fixture(scope="class")
    @classmethod
    def wide_gap_volume(cls):
        """Tilts spaced so the cone separation outruns a fixed radius."""
        return make_volume(fixed_angles=(0.5, 4.0, 10.0, 20.0), nrays=180, ngates=120)

    def _grid(self, volume, **roi_kwargs):
        grid = grid_volume(
            volume,
            grid_shape=self.GRID_SHAPE,
            grid_limits=self.GRID_LIMITS,
            method="pyart",
            fields=["reflectivity"],
            weighting_function="Barnes2",
            **roi_kwargs,
        )
        values = np.ma.filled(grid.fields["reflectivity"]["data"].astype(float), np.nan)
        return values

    def _mid_section(self, values):
        return values[:, values.shape[1] // 2, :]

    def test_beam_width_roi_leaves_interior_holes(self, wide_gap_volume):
        """The documented default leaves holes between cones."""
        section = self._mid_section(self._grid(wide_gap_volume, roi_func="dist_beam"))
        assert self._interior_holes(section) > 0

    def test_pyart_defaults_leave_them_too(self, wide_gap_volume):
        """Not a mis-set parameter: passing no ROI arguments does the same thing.

        Pinned because the natural reading of a hole-riddled section is that the
        caller chose badly. Py-ART's own defaults produce it, so the honest statement
        is about the operator, not about the arguments.
        """
        explicit = self._interior_holes(
            self._mid_section(self._grid(wide_gap_volume, roi_func="dist_beam"))
        )
        implicit = self._interior_holes(self._mid_section(self._grid(wide_gap_volume)))
        assert implicit > 0
        assert implicit == explicit

    def test_a_wider_radius_closes_them(self, wide_gap_volume):
        """A radius large enough to span the cone separation fills the gaps."""
        narrow = self._interior_holes(
            self._mid_section(self._grid(wide_gap_volume, roi_func="dist_beam"))
        )
        wide = self._interior_holes(
            self._mid_section(
                self._grid(wide_gap_volume, roi_func="constant", constant_roi=4000.0)
            )
        )
        assert wide < narrow

    def test_closing_the_holes_costs_peak_intensity(self, wide_gap_volume):
        """The trade-off itself, asserted so it cannot be quietly forgotten.

        Widening the radius averages over a larger neighbourhood: it fills more cells
        *and* attenuates the peak. Both directions are asserted, because a claim that
        one setting is simply better would be wrong.
        """
        narrow = self._grid(wide_gap_volume, roi_func="dist_beam")
        wide = self._grid(wide_gap_volume, roi_func="constant", constant_roi=4000.0)
        assert np.isfinite(wide).mean() > np.isfinite(narrow).mean()
        assert np.nanmax(wide) < np.nanmax(narrow)

    def test_the_spectral_path_has_no_interior_holes(self, wide_gap_volume):
        """The contrast that makes this a trade-off rather than a defect.

        Vertical assembly interpolates between whichever cones bracket the target
        height, so a separation wider than any fixed radius is not a problem for it.
        It declines to extrapolate *beyond* the outermost cones, which is a different
        thing and is what ``coverage_flag`` reports.
        """
        grid = grid_volume(
            wide_gap_volume,
            grid_shape=self.GRID_SHAPE,
            grid_limits=self.GRID_LIMITS,
            method="spectral",
            field_name="reflectivity",
        )
        values = np.ma.filled(grid.fields["reflectivity"]["data"].astype(float), np.nan)
        assert self._interior_holes(self._mid_section(values)) == 0


class TestFlavourAndMethodCompose:
    """Both axes of the public contract are independent and must stay that way.

    Object family and gridding method are orthogonal choices, so all four
    combinations are exercised here. The xradar-plus-spectral case is the one most
    likely to break silently: it is the only path where the volume is converted
    between object families *and* gridded by this package's own operator.
    """

    @pytest.fixture(scope="class")
    def datatree(self, tmp_path_factory):
        xradar = pytest.importorskip("xradar")
        path = tmp_path_factory.mktemp("flavour") / "volume.nc"
        pyart.io.write_cfradial(str(path), make_volume())
        return xradar.io.open_cfradial1_datatree(str(path))

    @pytest.mark.parametrize("method", GRIDDING_METHODS)
    def test_a_datatree_grids_to_a_dataset(self, datatree, method):
        import xarray as xr

        result = grid_volume(
            datatree, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method=method
        )
        assert isinstance(result, xr.Dataset)
        assert "reflectivity" in result

    @pytest.mark.parametrize("method", GRIDDING_METHODS)
    def test_a_radar_grids_to_a_grid(self, volume, method):
        result = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method=method
        )
        assert isinstance(result, pyart.core.Grid)

    def test_the_coverage_flag_survives_conversion_to_xarray(self, volume):
        """A flag dropped in conversion would be worse than never emitting one."""
        result = grid_volume(
            volume,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            method="spectral",
            output_flavor="xarray",
        )
        assert "coverage_flag" in result

    def test_both_families_give_the_same_spectral_answer(self, volume, datatree):
        """Converting the volume must not change the result.

        The same data arriving as a DataTree rather than a Radar is a representation
        difference, not a physical one, so any discrepancy here would be a defect in
        the conversion layer rather than in the operator.
        """
        from_radar = np.ma.filled(
            grid_volume(
                volume,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
            ).fields["reflectivity"]["data"],
            np.nan,
        )
        from_tree = grid_volume(
            datatree,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            method="spectral",
        )["reflectivity"].values
        np.testing.assert_allclose(
            from_radar, np.squeeze(from_tree), equal_nan=True, rtol=1e-5
        )
