(changelog)=
# Changelog

What each release changed for you. The [dev diary](#diary-about) records why.

(changelog-v1-2-1)=
## v1.2.1 — unreleased

**Fixed**

- Logging a path that lives on a different Windows drive from the working directory no longer
  raises — the log line falls back to the absolute path. —
  [#31](https://github.com/bastigw/mnutils/pull/31)

**Documentation**

- The README installs MNUtils from PyPI. — [#32](https://github.com/bastigw/mnutils/pull/32)

(changelog-v1-2-0)=
## v1.2.0 — 2026-08-20

First release published to PyPI.

**Added**

- `display_images` renders a self-contained HTML slice viewer, and MRSI spectra get an interactive
  inspector — both drawn with PIL so the page keeps working after the kernel stops. —
  [#12](https://github.com/bastigw/mnutils/pull/12) ·
  [#13](https://github.com/bastigw/mnutils/pull/13) ·
  [Displaying image arrays of any shape](#plotting-images) ·
  [Inspecting MRSI spectra interactively](#plotting-mrsi-inspector) ·
  [diary](#diary-anywidget-slice-viewer) · [diary](#diary-pil-image-grid)
- GitHub Actions CI: docs style check, lint, bare-install smoke test, matrix tests, and a PyPI
  publish on tag. — [#14](https://github.com/bastigw/mnutils/pull/14) ·
  [#26](https://github.com/bastigw/mnutils/pull/26)
- `[project.urls]` — Homepage, Documentation, Repository and Issues now show on the PyPI page. —
  [#29](https://github.com/bastigw/mnutils/pull/29)

**Changed**

- Spectra plotting takes a `DataArray` as its primary input. —
  [#7](https://github.com/bastigw/mnutils/pull/7) ·
  [Plotting FIDs and spectra from a DataArray](#plotting-spectra)
- Fitting is delegated to [`xmris`](https://github.com/andrewendlinger/xmris) (`xmris[fitting]`,
  which wraps pyAMARES) instead of being reimplemented here. — [AMARES fitting](#fitting-index)

**Fixed**

- The `display_images` HTML viewer keeps its slider inline past 100 slices. —
  [#23](https://github.com/bastigw/mnutils/pull/23)
- Plotting falls back to system sans-serif fonts when Arial is unavailable. —
  [#19](https://github.com/bastigw/mnutils/pull/19)
- The packaged sdist and wheel no longer ship the local `.ruff_cache` lint cache. —
  [#29](https://github.com/bastigw/mnutils/pull/29)

**Maintenance**

- Two dependency bumps. — [#16](https://github.com/bastigw/mnutils/pull/16) ·
  [#17](https://github.com/bastigw/mnutils/pull/17)

(changelog-earlier)=
## Earlier releases

`v1.1.2` and everything before it predate this changelog. Their contents are recorded only in the
[tag list](https://github.com/bastigw/mnutils/tags).
