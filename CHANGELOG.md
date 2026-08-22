# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from
git tags via `setuptools-scm`.

## [Unreleased]

### Added

- `advection_interpolate` gains `carry_fields`, warping several variables through a
  single motion estimate. The flow must come from a tracer, and the polarimetric
  variables are not tracers: differential reflectivity is a shape measure, copolar
  correlation is closer to a quality flag, and Doppler velocity is signed and folded
  at the Nyquist interval. One call now reconstructs reflectivity, velocity, ZDR and
  RhoHV from one flow estimate and one gridding pass, each field keeping its own
  metadata so units are not overwritten with the tracer's. `interp_field_name`
  renames only the tracer. Folded velocity carries a documented caveat: the blend is
  a weighted mean, so it flattens gates that cross a fold — 7.1% of echo gates above
  15 dBZ on the ARM C-SAPR2 pair used to exercise this, against 20.4% of all valid
  gates, the rest being noise. Dealias first if the interpolated velocity is to be
  read quantitatively.

- `grid_volume` gains `method={"pyart", "spectral"}`, wiring the spectral operator
  behind the public entry point. The FFT gridding capability is now complete end to
  end: sweep census, per-sweep spectral evaluation on its cone, and vertical
  assembly, reachable in one call from either object family.
- The spectral path emits a `coverage_flag` field alongside the data, carrying
  `VerticalFlag` per cell. A radar volume does not observe a box, and an unexplained
  NaN leaves a user unable to tell a data gap from a geometric coverage limit.

- `radar_palette.gridding.cones`: `cone_range_height`, `target_lattice`,
  `dedup_sweeps`, `build_cones`, `beam_footprint_crossover`, plus `ConeStack` and
  `TiltReport`. A sweep samples a curved cone, not a plane, so each tilt is gridded
  onto a shared horizontal lattice carrying its own height surface. `dedup_sweeps`
  collapses split cuts by measured range rather than by cut naming, so it needs no
  knowledge of any radar's scan-strategy vocabulary.
- `radar_palette.gridding.vertical`: `interp_column_stack`, `target_elevation`,
  `intercone_gap`, `VerticalFlag`, `VerticalProfile` and `DEFAULT_VERTICAL_SCHEME`.
  Interpolates a cone stack onto target heights per column, with **no
  extrapolation**: every finite value carries `VerticalFlag.INTERPOLATED` and every
  other cell is NaN with a flag saying why. `INTERIOR_GAP` covers the case that is
  easy to miss — maximum valid range is a property of the echo, not the geometry, so
  it can be non-monotonic in elevation and leave a hole in a column's tilt run that
  must not be interpolated across.

- `radar_palette.advection.flow`: `grid_optical_flow` estimates a dense,
  height-resolved echo motion field between two gridded volumes (TV-L1 optical flow
  per vertical level). Returns displacement in metres in the earlier-to-later sense,
  `(north, east)` order.
- `radar_palette.advection.morph`: `advection_interpolate` reconstructs a volume at
  a time between two observed volumes, advecting each bracketing volume to the target
  time along that field and blending. The morph runs per gate in native
  (azimuth, slant range, elevation) coordinates, so the output is an ordinary volume
  rather than a Cartesian grid; only the motion estimation goes through a grid.
  Accepts and returns both object flavours, and applies the acquisition-time
  correction from `radar_palette.advection.timing` at the source, so a reconstructed
  volume never inherits a bracketing volume's clock.

- `radar_palette.gridding.evaluator`: `SweepSpectralEvaluator` turns a sampled sweep
  into a continuous band-limited function, evaluable at arbitrary
  (range, azimuth). Three azimuth paths chosen by sweep class — exact FFT for a
  closing full-circle lattice, gap-closing extension for a sector, Kaiser-Bessel
  NUFFT for non-uniform rays. Range is mirror-extended, never treated as periodic.
- `radar_palette.gridding.nufft`: Kaiser-Bessel forward/adjoint operator pair with
  an adjoint-consistency test recorded in the evaluator's report, plus conjugate-
  gradient solution of the density-weighted normal equations.
- `radar_palette.gridding.reflectivity`: `to_dbz`, `to_linear`,
  `looks_like_linear_reflectivity` and `LinearReflectivityError`. The evaluator
  refuses input that looks like linear Z, because interpolating linear
  reflectivity across a hard echo edge drove 42.6% of evaluated samples negative in
  a measured reproduction.
- `radar_palette.gridding.fill`: `fill_sweep` makes a sweep gap-free by constant,
  edge-interpolating or band-limited (Gerchberg-Papoulis) fill. All three leave
  measured samples untouched.
- `radar_palette.gridding.geometry`: `antenna_to_cartesian_43` and
  `cartesian_to_antenna_43`, forward and exact analytic inverse radar geometry on
  the 4/3 effective earth, in metres. The inverse exists because
  `pyart.core.cartesian_to_antenna` is not the inverse of Py-ART's own forward
  transform; see below.
- `radar_palette.gridding.census`: `census_sweep`, `census_radar`, `SweepGeometry`
  and `SweepClass` measure a volume's sampling geometry and classify each sweep as
  `EXACT_UNIFORM_PERIODIC`, `UNIFORM_PARTIAL_SECTOR` or `NON_UNIFORM`, which
  determines whether a plain FFT is valid on the azimuth axis. Also tags split-cut
  groups, since NEXRAD VCPs repeat the low tilts and de-duplication is mandatory
  before vertical interpolation. Accepts either object flavour.
- Classifier tolerances are public and documented: `AZ_DEV_TOL_FRAC` (0.01),
  `DR_DEV_TOL_FRAC` (1e-3), `SPLIT_CUT_TOL_DEG` (0.05). The azimuth tolerance is
  fractional rather than absolute because the worst-case Nyquist error is `pi * f`,
  independent of ray count.

- `radar_palette.advection.timing`: reconstruction of per-ray acquisition times
  for advection-interpolated volumes (`volume_reference_time`,
  `interpolate_ray_times`, `apply_interpolated_time`). An interpolated volume now
  carries the time it represents rather than inheriting the earlier bracketing
  volume's clock. Intra-volume ray timing structure is preserved, since ray-to-ray
  spacing is set by the scan strategy and not by the interpolation.
- `radar_palette.testing`: `make_empty_ppi_volume` and `assign_scan_times` build
  bare PPI geometries with explicitly controlled acquisition times, so timing
  behaviour can be tested independently of any field data.
- **Python floor raised to 3.11** (`requires-python = ">=3.11"`). `xradar` 0.12.0 — the release whose cfradial readers this
  package depends on — requires 3.11, and Py-ART made the same move (`arm_pyart`
  2.2.1 dropped 3.10). A 3.10 environment could only resolve `arm_pyart` 2.2.0 /
  `xradar` 0.11.1, an older stack than the one tested here. The CI matrix entry for
  3.10 is removed in a companion change to `.github/workflows/ci.yml`, which a
  personal access token cannot push (GitHub gates workflow files on the classic
  `workflow` scope); until it lands the 3.10 job fails at install, correctly, since
  pip refuses a package whose `requires-python` excludes the interpreter.
- `xarray`, `xradar` and `netCDF4` are now declared dependencies. They were
  previously satisfied transitively through `arm_pyart` while being imported
  directly, which works only as long as a resolver happens to choose that way.
- Packaging tests assert that every third-party module the library imports is a
  declared dependency, and that the declared `xarray` floor actually provides
  `xarray.DataTree`.
- `radar_palette.io`: object-flavour interoperability between Py-ART
  (`Radar`/`Grid`) and xradar (`DataTree`/xarray `Dataset`). Conversions delegate
  to Py-ART's own layer (`pyart.xradar`, `Grid.to_xarray`, xradar's cfradial
  readers) rather than reimplementing the data-model mapping.
- `radar_palette.advection.retime_interpolated_volume` and
  `radar_palette.gridding.grid_volume`: flavour-aware entry points. Both accept
  either object family; by default the advection entry point returns the family it
  was given, and the gridding entry point maps Py-ART input to a `Grid` and xradar
  input to an xarray `Dataset`. `output_flavor` overrides either default.

### Changed

- `DEFAULT_GRIDDING_METHOD` is `"pyart"`, so **existing callers of `grid_volume` get
  byte-identical results** — asserted by a test comparing the default against a
  direct `pyart.map.grid_from_radars` call. The spectral operator is opt-in because
  the two are different rather than ranked: on a hard azimuthal echo edge (10 to
  50 dBZ wedge) the spectral interpolant reaches 55.5 and 4.7 dBZ (+5.5 / −5.3 dB
  beyond the data) while distance weighting stays inside the input range, since an
  average cannot exceed its inputs. Where both produce data on a smooth field they
  agree to 0.35 dB median. A test ties the constant to that measurement, so flipping
  the default fails with a reason rather than silently.

- Vertical interpolation defaults to `pchip_z` (monotone), which measured best of the
  four schemes and cannot introduce an extremum between two cones. The margin is
  small and that is the point: **tilt spacing, not scheme choice, is what limits
  vertical accuracy.** On a bright band at 2500 m with a 400 m standard deviation,
  going from 6 tilts to 23 (median inter-cone thickness 680 m to 213 m) improved RMSE
  from 4.086 dB to 0.295 dB — a factor of 14 — while the best-vs-worst scheme spread
  at 6 tilts was only 4.024 to 5.194 dB. Within one scan strategy the error tracks the
  local layer thickness: 0.42 dB median where adjacent cones are 400–700 m apart,
  3.81 dB where they are 1200–2500 m apart. Read a gridded value alongside its
  `gap_m`.

- `test_third_party_imports_are_declared_dependencies` now accepts a package declared
  in *either* `project.dependencies` or an extra. Previously it failed on any
  correctly-gated optional import, which would have pressured a contributor into
  declaring `scikit-image` as required or deleting the guard. A companion test,
  `test_optional_imports_are_not_required_dependencies`, closes the loophole from the
  other side by asserting `scikit-image` and `finufft` stay optional. Both were
  verified non-vacuous by deleting a declaration and confirming each fires.

- **Conjugate-gradient refinement of the NUFFT path is now on by default**
  (`DEFAULT_CG_ITERATIONS = 12`), departing from the research code it was ported
  from. Convolution gridding computes the adjoint of the sampling operator, not its
  inverse: on a constant field with ±0.3-spacing azimuth jitter, unrefined gridding
  reproduces the constant to only 60% relative deviation, scaling with jitter (3.6%
  at ±0.02, 19% at ±0.1). Twelve iterations bring it to 0.07% for roughly 50% more
  build time. Pass `n_cg=0` for the unrefined operator.
- The linear-Z guard tests **magnitude** rather than dynamic range. A max/min ratio
  is unreliable in both directions — a dBZ field spanning 0–50 has a ratio of 1e5
  and would be falsely flagged, while linear Z from a 6.8–31.8 dBZ scene has a ratio
  of only 316 and would be missed. Magnitude separates the cases cleanly. Known
  blind spot: a linear field peaking below ~23 dBZ stays under the ceiling and is
  not caught.

### Fixed

- `interpolate_ray_times` no longer requires the two volumes to have equal ray
  counts. Consecutive volumes running the *same* scan strategy do not repeat their
  ray count — measured 13955 / 13951 / 13954 on one ARM C-SAPR2 triple — so the
  guard rejected essentially every real pair. The arithmetic never needed it: the
  output carries the earlier volume's acquisition pattern shifted by a single scalar
  derived from the two reference times, so `radar_late` contributes no per-ray
  correspondence to violate.
- `target_time` now accepts a naive or `cftime` instant. `volume_reference_time`
  returns a UTC-aware time but `pyart.util.datetime_from_radar` returns a naive
  `cftime.DatetimeGregorian`, so building a target from a Py-ART time -- the most
  natural call a Py-ART user can make -- raised `TypeError: can't subtract
  offset-naive and offset-aware datetimes`. A naive target is now read as UTC, which
  CF times are by construction, matching the rule already applied to decoded ray
  times. The normalisation rebuilds the instant field by field rather than calling
  `replace(tzinfo=UTC)`, because `cftime.DatetimeGregorian` is not a
  `datetime.datetime` subclass and its `replace()` rejects `tzinfo`.
- `_sweep_sampler` now collapses repeated azimuths instead of raising. A real
  antenna records the same azimuth twice when it dwells — 3 of 195 sweeps across 13
  consecutive volumes — and the sorted-but-not-deduplicated axis made the wrap
  tiling in `_sample_sweep` non-strictly-ascending, which `RegularGridInterpolator`
  rejects outright. Duplicates are averaged rather than discarded: they are
  independent samples of the same beam position. Note the azimuths are otherwise
  well behaved, with zero non-monotonic sweeps once the 360° wrap is accounted for,
  so deduplication is the fix rather than resorting.

  Both were found by running the operator on a real volume sequence for the first
  time, and neither is reachable from synthetic volumes built on `arange`.

- Documented, with measured values, that `pyart.core.cartesian_to_antenna` is not
  the inverse of `pyart.core.antenna_to_cartesian`: the forward transform returns
  great-circle arc length on the 4/3 earth while the inverse computes flat
  straight-line distance with no curvature term. Range error reaches -313 m at
  118 km and -4649 m at 460 km, maximised over elevation (it peaks near 35 degrees,
  not at grazing incidence); elevation error reaches +1552 mdeg. Azimuth is
  unaffected. `radar_palette` ships its own inverse, closing to ~1e-10 m, and
  asserts the discrepancy in tests so an upstream fix surfaces as a failure.

- `Xradar.time` assignment does not propagate to the underlying `DataTree`, so
  correcting an interpolated volume's time through the Py-ART wrapper left an
  xradar caller's tree on its original clock. Time is now written back per sweep
  (`radar_palette.io.write_ray_times_to_datatree`). Note the asymmetry with
  `add_field`, which does write through.

- Advection-interpolated volumes previously reported the acquisition time of the
  earlier bracketing volume. The output looked valid and plotted without
  complaint, so any time-ordered downstream use (accumulation, cell tracking,
  matching against another instrument) was silently misaligned by up to the full
  inter-volume interval.

- Packaging scaffold: `src/` layout, `setuptools` + `setuptools-scm` build via
  `pyproject.toml`, optional-dependency extras (`advection`, `spectral`, `all`,
  `test`, `docs`, `dev`).
- Documented but empty subpackages `radar_palette.advection`,
  `radar_palette.gridding`, `radar_palette.util`, `radar_palette.testing`.
- Packaging-level test suite, `ruff` lint/format config, `pre-commit` hooks,
  Sphinx docs stub, `CONTRIBUTING.md`, `MANIFEST.in`, and a pull-request
  template.
- GitHub Actions CI (lint; tests on Python 3.10-3.13 plus a minimal-install
  job; build with `twine check --strict`) is added as a separate commit, since
  writing `.github/workflows/` requires a token permission the packaging push
  did not carry.
