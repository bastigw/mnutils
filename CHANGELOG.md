# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.3] - 2026-08-26

### Added

- `display_images` accepts `zooms=(row, col)` to render panels at physical voxel spacing instead of
  pixel count; `NiiBase.display()` passes it automatically
  ([#35](https://github.com/bastigw/mnutils/issues/35)).
- `utils.nifti.get_display_zooms`, and `get_display_affine` now also accepts `display_plane` directly.
- `docs/nifti/synthetic_dataset_gallery.md`, which checks every synthetic fixture's affine against its
  own ground truth.

### Fixed

- `get_all_dicom_series_ids` derives the gap-check range from the series that are actually `<= 99` and
  ends at the highest one present, so an exam containing a reformat series `>= 100` no longer reports
  every free ID up to 99 as missing. Also fixes an `IndexError` on an empty data folder
  ([#34](https://github.com/bastigw/mnutils/issues/34)).
- `create_MRSI_affine` applied a resize shift *and* an extra half-blocky-voxel shift; the interpolated
  pseudo image's affine was therefore offset by half a native voxel in x/y, visible as a posterior/right
  shift when overlaid on a T1.
- `NiiBase.with_new_data()` no longer force-reorients to `("P", "L", "S")`, which silently decoupled the
  array data from the affine it was paired with.
- Anisotropic voxels render stretched to their physical shape in the HTML image grid instead of square.

## [1.2.2] - 2026-08-25

### Fixed

- `get_all_dicom_series_ids` no longer treats reformat/resave series IDs (`>= 100`, e.g. 500/501,
  650/651, 40003) as gaps; only gaps within the dense protocol-step range are logged as errors
  ([#34](https://github.com/bastigw/mnutils/issues/34)).

## [1.2.1] - 2026-08-25

### Added

- Fast raster engine for `overlay_image_data_on_T1`: overlays are composited at pixel level
  instead of drawn as per-voxel patches, so large MRSI grids render in a fraction of the time
  ([#36](https://github.com/bastigw/mnutils/issues/36)).

### Changed

- README documents installing MNUtils from PyPI rather than from a git URL.
- Dev-tooling dependencies bumped (`dev-tools` group).

## [1.2.0] - 2026-08-20

First release published to PyPI.

### Added

- `display_images` HTML slice viewer and an interactive MRSI inspector, rendered self-contained via PIL.
- GitHub Actions CI: docs style check, lint, bare-install smoke test, matrix tests, and PyPI publish on tag.
- `[project.urls]` (Homepage, Documentation, Repository, Issues).

### Changed

- Spectra plotting now takes a `DataArray` as its primary input.
- Fitting delegated to [`xmris`](https://github.com/andrewendlinger/xmris) (`xmris[fitting]`, wrapping pyAMARES) instead of reimplemented locally.

### Fixed

- `display_images` HTML viewer slider staying inline past 100 slices.
- Plotting falls back to system sans-serif fonts when Arial is unavailable.
- Packaged sdist/wheel no longer ships the local `.ruff_cache` lint cache.

[Unreleased]: https://github.com/bastigw/mnutils/compare/v1.2.3...HEAD
[1.2.3]: https://github.com/bastigw/mnutils/releases/tag/v1.2.3
[1.2.2]: https://github.com/bastigw/mnutils/releases/tag/v1.2.2
[1.2.1]: https://github.com/bastigw/mnutils/releases/tag/v1.2.1
[1.2.0]: https://github.com/bastigw/mnutils/releases/tag/v1.2.0
