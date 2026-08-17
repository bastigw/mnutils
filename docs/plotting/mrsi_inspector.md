---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

(plotting-mrsi-inspector)=

# Inspecting MRSI spectra interactively

> **I want to click around a T1 slice and see the MRSI spectrum under my cursor update live —
> without keeping a Python kernel running.**

[`inspect_MRSI_spectra()`](#mnutils.plotting.images.inspect_MRSI_spectra) shows a T1 image with an
MRSI overlay next to a spectrum plot. Click a voxel, use the arrow keys, or drag the slice slider,
and the spectrum panel updates to match. Every anatomical slice is pre-rendered once and every
voxel's spectrum is precomputed into a single buffer embedded in the widget, so all of that
interaction runs client-side — it keeps working after this page is built into a static site, the
same way the slider in [Displaying image arrays of any shape](#plotting-images-scrub) does.

| Function                                                                  | What it does here                             |
| ------------------------------------------------------------------------- | --------------------------------------------- |
| [`inspect_MRSI_spectra()`](#mnutils.plotting.images.inspect_MRSI_spectra) | builds and displays the interactive inspector |

```{code-cell} ipython3
:tags: [remove-cell]

from loguru import logger

logger.remove()
```

```{code-cell} ipython3
from pathlib import Path

from mnutils.GESeries import MRISeries, MRSISeries
from mnutils.plotting.images import inspect_MRSI_spectra


def _repo_root(start: Path = Path.cwd()) -> Path:
    # Anchor on pyproject.toml rather than a relative "../.." count: this page
    # is executed from two different working directories -- docs/plotting/ when
    # mystmd builds it, tests/autogen_notebooks/plotting/ when nbmake runs the
    # notebook `uv run test-gen` generates from it -- so no single relative
    # path satisfies both.
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root (no pyproject.toml found)")


hevo18_data = _repo_root() / "tests" / "datasets" / "HeVo-18" / "data"
```

(plotting-mrsi-inspector-load)=

## 1. Load a matching T1/MRSI pair

Same anonymized phantom dataset as
[Drawing an MRSI voxel on an anatomical slice](#nifti-voxel-overlay) — series 2 is the
T1-weighted anatomical scan, series 8 the MRSI acquisition covering it.

```{code-cell} ipython3
t1 = MRISeries(hevo18_data, 2)
mrsi = MRSISeries(hevo18_data, 8)
```

(plotting-mrsi-inspector-widget)=

## 2. Display the inspector

`blocky=True` (the default) resamples the raw, un-interpolated MRSI grid — one cell per acquired
voxel — rather than the smoothed NIfTI, so the overlay and the voxel picker line up with what was
actually measured. `autophase=True` (also the default) autophases each voxel's spectrum before
display; pass `magnitude=True` for the magnitude spectrum instead.

```{code-cell} ipython3
inspect_MRSI_spectra(t1, mrsi)
```

Click anywhere on the left image to move the selected voxel there, use the arrow keys to nudge it
by one voxel, or drag the slider below the image to change the anatomical slice — the MRSI slice
shown updates to whichever one that anatomical slice actually falls inside, and the spectrum panel
on the right always reflects the currently selected voxel.

:::{note}
`inspect_MRSI_spectra()` returns `None`, the same convention
[`display_images()`](#plotting-images-scrub) uses once it hands off to an interactive widget —
there's no `fig`/`axes` to return because the display is no longer a single static figure.
:::

(plotting-mrsi-inspector-options)=

## 3. Magnitude vs. autophased vs. raw real spectrum

The three display modes are mutually exclusive; `magnitude=True` takes priority; the spectrum
label shown in the widget's header names whichever one is active:

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: label reflects the chosen spectrum mode, for each mutually-exclusive option
from mnutils.plotting._widgets.mrsi_inspector import MRSIVoxelInspectorWidget

_kwargs = dict(
    left_frames=[b""],
    slice_titles=["s"],
    n_anat_slices=1,
    initial_slice=0,
    image_width=1,
    image_height=1,
    mrsi_to_display_affine=[0.0] * 16,
    display_to_mrsi_affine=[0.0] * 16,
    grid_shape=(1, 1, 1),
    mrsi_dims=(1, 1),
    initial_voxel=(0, 0, 0),
    ppm=[0.0],
    spectra_bytes=b"\x00" * 4,
    spectra_scale=1.0,
    npts=1,
)
for _label in ("Magnitude Spectrum", "Autophased Spectrum", "Real Spectrum"):
    _widget = MRSIVoxelInspectorWidget(spectrum_label=_label, **_kwargs)
    assert _widget._data["spectrum_label"] == _label
```

```{code-cell} ipython3
inspect_MRSI_spectra(t1, mrsi, magnitude=True)
```

```{code-cell} ipython3
inspect_MRSI_spectra(t1, mrsi, autophase=False)
```

:::{warning}
Passing both `magnitude=True` and `autophase=True` logs a warning and silently ignores
`autophase` — magnitude spectra have no phase left to correct. Leave `autophase` at its default
unless you're explicitly asking for the real-part spectrum.
:::

:::{seealso}
[Drawing an MRSI voxel on an anatomical slice](#nifti-voxel-overlay) covers the affine math this
widget's client-side voxel picker relies on — `mrsi_to_display_affine` and its inverse are built
the same way here as there. [The GESeries class hierarchy](#data-model-geseries) covers
`MRISeries`/`MRSISeries` generally.
:::
