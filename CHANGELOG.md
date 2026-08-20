# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/bastigw/mnutils/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/bastigw/mnutils/releases/tag/v1.2.0
