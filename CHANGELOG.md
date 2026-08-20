# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from
git tags via `setuptools-scm`.

## [Unreleased]

### Added

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
