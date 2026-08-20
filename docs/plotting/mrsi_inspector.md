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
from mnutils.GESeries import MRISeries, MRSISeries
from mnutils.plotting.images import inspect_MRSI_spectra
from mnutils.testing import build_fake_exam

nist_data = build_fake_exam("nist_phantom_exam") / "data"
```

(plotting-mrsi-inspector-load)=

## 1. Load a matching T1/MRSI pair

Same fake phantom exam as
[Drawing an MRSI voxel on an anatomical slice](#nifti-voxel-overlay) — series 2 is the
T1-weighted anatomical scan, series 8 the MRSI acquisition covering it.

```{code-cell} ipython3
t1 = MRISeries(nist_data, 2)
mrsi = MRSISeries(nist_data, 5)
```

```{code-cell} ipython3
t1.display()
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

The spectrum panel carries its own two range sliders: the horizontal one under the plot sets the
ppm window (the whole acquired sweep is embedded, so it reaches every sample), and the vertical one
beside it sets the y limits. Touching the vertical slider ticks _Fixed y-axis_ for you — an axis
you dialled in by hand only stays meaningful if it stops rescaling itself — and unticking the box
hands the axis back to the automatic per-voxel scaling. Leaving the box ticked without touching the
slider instead scales the axis to the loudest voxel anywhere in the grid, so amplitudes stay
comparable as you step from voxel to voxel.

:::{dropdown} What embedding the whole sweep costs
The widget used to crop the spectra to a ppm window before embedding them, on the assumption that a
wide sweep would make the payload unwieldy — a 2H acquisition at 3 T spans ~255 ppm, of which the
interesting ±20 ppm is a sixth. In practice the compressed buffer is small enough either way: for
the 16×16×16 grid at 700 points on this page it is ~6.7 MB embedded against ~1.1 MB cropped, and
the widget stays responsive. Not cropping is what lets the ppm slider reach every acquired sample
instead of a window fixed at call time, so the crop (and its `ppm_range` argument) is gone.
Grids large enough to matter still log a warning while encoding.
:::

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
    spectra_min=0.0,
    spectra_max=1.0,
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
