# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from
git tags via `setuptools-scm`.

## [Unreleased]

### Added

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

### Fixed

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
