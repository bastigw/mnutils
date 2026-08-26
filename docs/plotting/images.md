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

(plotting-images)=

# Displaying image arrays of any shape

> **I have an array — could be one 2D slice, could be a volume, could be several volumes — and I
> just want to look at it.**

[`display_images()`](#mnutils.plotting.images.display_images) takes 2D, 3D or 4D input and
picks a sensible layout automatically.

```{code-cell} ipython3
:tags: [remove-cell]

from loguru import logger

logger.remove()
```

::: {dropdown} Imports & random number seed & 4D volume generation

```{code-cell} ipython3
import numpy as np

from mnutils.plotting.images import display_images

rng = np.random.default_rng(1337)

# 4 D volume creation
size_xy = 90
num_z = 100
num_t = 7  # 4th dimension length (e.g. 30 time steps)

# 4D coordinate grids: shapes (90,1,1,1), (1,90,1,1), (1,1,100,1), (1,1,1,30)
y, x, z, t = np.ogrid[:size_xy, :size_xy, :num_z, :num_t]

# 1. Dynamic center that orbits across dimension 4
x_center = (size_xy // 2) + 15 * np.sin(2 * np.pi * t / num_t)
z_center = 50 + 20 * np.cos(2 * np.pi * t / num_t)

# 2. Dynamic radius/scale that pulses along dimension 4
scale_xy = (size_xy // 2 - 5) + 5 * np.sin(4 * np.pi * t / num_t)

# Normalized radial distance squared in 4D space
r_sq = (
    ((x - x_center) / scale_xy) ** 2
    + ((y - size_xy // 2) / (size_xy // 2 - 5)) ** 2
    + ((z - z_center) / 25) ** 2
)

# Gaussian pattern + 4D noise
pattern = -10 + 28 * np.exp(-r_sq)
noise = rng.normal(0, 0.3, size=(size_xy, size_xy, num_z, num_t))

volume_4d = np.clip(pattern + noise, -10, 20)
```

:::

(plotting-images-rule)=

## The rule: the last axis is "how many images"; everything else is "one volume"

- **2D** `(H, W)` — one image.
- **3D** `(H, W, S)` — still **one image**: the _middle slice_ along the last axis, `S // 2`. Not
  a grid of `S` images.
- **4D** `(H, W, S, N)` — a **grid of `N` images**, laid out up to 5 per row. Each of the `N` still
  only shows _its own_ middle slice along axis 2 — the same "last axis of each sub-array is
  slices, not images" rule applies one level down.

```{code-cell} ipython3
single = volume_4d[:, :, num_z // 2, 1]  # single slice of the 4D volume
display_images(single)
```

A 4D array with one slice per volume is the grid case: seven panels, laid out five per row.

```{code-cell} ipython3
one_slice_grid = volume_4d[:, :, num_z // 2 : num_z // 2 + 1, :]
one_slice_grid.shape
```

```{code-cell} ipython3
display_images(one_slice_grid, titles=[f"t = {i}" for i in range(num_t)], colorbar=True)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: display_images axis semantics.
# `display_images` only *displays*, so the panel count is checked on the
# renderer it drives rather than on a return value.
from mnutils.plotting.images import _render_image_grid_frames

frames, bounds = _render_image_grid_frames(
    one_slice_grid, num_images=num_t, cmap="magma", vmin=0, vmax=1, per_panel_bounds=False
)
assert len(frames) == 1  # one slice
assert len(frames[0]) == num_t  # seven panels, one per 4th-dimension entry

frames_2d, _ = _render_image_grid_frames(
    single[:, :, np.newaxis, np.newaxis],
    num_images=1,
    cmap="magma",
    vmin=0,
    vmax=1,
    per_panel_bounds=False,
)
assert len(frames_2d) == 1 and len(frames_2d[0]) == 1
```

A **list** of same-shaped arrays works too — it's stacked along a new last axis before the same
rule applies (more on that [below](#plotting-images-scrub), since a list is often exactly how a
multi-slice case shows up).

(plotting-images-scrub)=

## But a volume has more than one slice — can I see the rest?

The examples above all used a single slice. Real volumes don't stop at one: a `(H, W, S)` array
with `S > 1` still shows only its middle slice by default — so what about the other `S - 1`?

`display_images()` shows them too, as a slider, right here on this page — not just in a live
Jupyter session. Every call displays an interactive widget and returns `None`; with more than one
slice that widget gains a slider, and with exactly one it simply doesn't.

```{code-cell} ipython3
display_images(volume_4d[:, :, :, 1])
```

The same rule applies one level down for a 4D grid: every panel shares a single slider, since
they're all still slices of the same underlying volume stack.

```{code-cell} ipython3
display_images(volume_4d)
```

:::{note}
The slider works after this page is built into a static site too, not only in a live kernel —
every slice is pre-rendered once, up front, and the slider just swaps between them client-side.
:::

This is also why a **list** of plain 2D arrays is a multi-slice case, not a static one: stacked
along a new last axis, four 2D arrays become one volume with four slices — the same "not a grid"
rule as above, just arrived at differently — so it engages the same interactive widget:

```{code-cell} ipython3
list_of_images = [np.full((10, 10), i, dtype=float) for i in range(4)]
display_images(list_of_images)
```

A list of _volumes_ (each already 3D) becomes 4D instead — a grid, one per volume:

```{code-cell} ipython3
list_of_single_slice_volumes = [np.full((10, 10, 1), i, dtype=float) for i in range(4)]
display_images(list_of_single_slice_volumes, titles=[f"volume {i}" for i in range(4)])
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: list input follows np.stack(..., axis=-1), then the same shape rule
stacked = np.stack(list_of_single_slice_volumes, axis=-1).astype(np.float32)
assert stacked.shape == (10, 10, 1, 4)  # stacked to 4D -> grid of 4
frames_list, _ = _render_image_grid_frames(
    stacked, num_images=4, cmap="magma", vmin=0, vmax=3, per_panel_bounds=False
)
assert len(frames_list) == 1 and len(frames_list[0]) == 4
```

:::{warning}
Every array in a list must share the same shape — `display_images` stacks them with
`np.stack(images, axis=-1)`, which raises `ValueError` on a mismatch. There's no fallback path
for comparing differently-shaped images side by side; resize or crop to a common shape first.
:::

(plotting-images-annotations)=

## Where the titles and the colorbar come from

The panels you see above are plain images, rendered straight from the array at its own pixel size
— no matplotlib figure is built at any point, which is why nothing is returned to post-process.
Everything drawn _around_ the pixels is HTML instead: `fig_title` becomes a heading, `titles`
become captions, and `colorbar=True` becomes a gradient strip labelled with the bounds the panels
were scaled to.

```{code-cell} ipython3
display_images(
    volume_4d[:, :, :, :3],
    fig_title="Three time points, one shared slider",
    titles=["t = 0", "t = 1", "t = 2"],
    colorbar=True,
)
```

Pass `colorbar_kws={"mode": "each"}` for one bar per panel instead of one for the grid. That mode
autoscales every panel to its own slice, so the numbers beside each bar change as you scrub:

```{code-cell} ipython3
display_images(
    volume_4d[:, :, :, :3],
    titles=["t = 0", "t = 1", "t = 2"],
    colorbar=True,
    colorbar_kws={"mode": "each"},
)
```

:::{warning}
Arguments that only made sense against an `Axes` — `aspect`, `xlabel`, `xticks`, `imshow_kws`,
`fig_kws` — are still accepted so old call sites don't break, but they no longer do anything. A
debug-level log line names them when you pass one.
:::

Panels are the array's own pixels, so a non-square array stays non-square — a `(30, 120)` slice is
drawn four times wider than it is tall, not stretched to fill its share of the row:

```{code-cell} ipython3
wide = np.stack([volume_4d[30:60, :, num_z // 2, i] for i in range(7)], axis=-1)[:, :, np.newaxis]
wide.shape  # (30, 90, 1, 3): three wide panels, one slice
```

```{code-cell} ipython3
display_images(wide, titles=["t = 0", "t = 1", "t = 2"], colorbar=True)
```

:::{note}
Panels fill their column and the column shrinks with the space available, so a narrow editor pane
produces smaller panels rather than clipped ones — down to a floor. `--mnu-panel-min-h` and
`--mnu-panel-max-h` bound a single panel's height, `--mnu-panel-min-w` its width, and because a
panel's height is its width times its aspect ratio the two floors combine: a row wraps once its
columns would fall below *either*, which is what keeps a row of wide frames from flattening into
strips. `--mnu-grid-max-h` then caps the *box* rather than the panels: once the rows outgrow it
the panel area scrolls, so a grid of many images keeps them readable instead of shrinking them all
to fit. Raising `--mnu-panel-max-h` is what makes the panels bigger on a wide screen.
:::

Once there are enough panels, the columns wrap at `--mnu-panel-min-w` and the rows that don't fit
go behind a scrollbar — the block stays the same size on the page whether it holds three images or
thirty:

```{code-cell} ipython3
many = rng.random((32, 32, 1, 24))
display_images(many, titles=[f"echo {i}" for i in range(24)])
```

(plotting-images-lowres)=

## Low-resolution arrays keep their pixels

A `(20, 20)` mask blown up to a few hundred screen pixels is a different problem from a `512²`
anatomical shrunk to fit: the first wants to look like the twenty-by-twenty grid of values it
actually is, the second wants smoothing. Both happen automatically — a panel scaled _up_ is drawn
with hard pixel edges, one scaled _down_ keeps the browser's smooth default — and small frames are
encoded losslessly, so no compression ringing creeps in around the blocks.

```{code-cell} ipython3
tiny = rng.integers(0, 4, size=(8, 8)).astype(float)
display_images(tiny, colorbar=True)
```

The same holds for a non-square low-resolution array, and for a grid of them:

```{code-cell} ipython3
display_images(rng.random((5, 20)))
```

```{code-cell} ipython3
masks = np.stack([(rng.random((16, 16)) > 0.5).astype(float) for _ in range(4)], axis=-1)
display_images(masks[:, :, np.newaxis, :], titles=[f"mask {i}" for i in range(4)])
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: small panels are encoded losslessly, big ones are not.
from mnutils.plotting.images import GRID_LOSSLESS_MAX_PIXELS

assert 8 * 8 <= GRID_LOSSLESS_MAX_PIXELS
tiny_frames, _ = _render_image_grid_frames(
    tiny[:, :, np.newaxis, np.newaxis],
    num_images=1,
    cmap="magma",
    vmin=0,
    vmax=3,
    per_panel_bounds=False,
)
import io as _io

import PIL.Image as _PILImage

with _PILImage.open(_io.BytesIO(tiny_frames[0][0])) as _im:
    assert _im.size == (8, 8)  # native resolution, no upscaling baked in
    # Lossless: every value in an 8x8 of four levels survives the round trip.
    assert len(set(_im.convert("RGBA").get_flattened_data())) == len(np.unique(tiny))
```

(plotting-images-zeros)=

## Zeros vs. real signal

Background voxels are often exactly `0`, not a small real value — treating them as data skews a
colorbar built from `vmin`/`vmax`/percentile clipping toward the background instead of the signal
you actually care about. Estimating bounds therefore always drops the zeros: that is
`fast_bounds(..., exclude_zeros=True)` under the hood, and any caller of `fast_bounds` can ask for
the same.

What you *see* is a separate decision, and it is the one `zeros_as_nan` makes — on by default:

```{code-cell} ipython3
mostly_zero = np.zeros((20, 20))
mostly_zero[8:12, 8:12] = rng.random((4, 4)) * 100

display_images(mostly_zero, zeros_as_nan=True, colorbar=True)
```

Masked-out voxels are _transparent_, not a colour: the panels are RGBA images, so the background
of the page shows through them in both light and dark themes. Pass `zeros_as_nan=False` when you
want the background painted with the colormap's low end instead — the bounds are unaffected either
way, which is why pinning `vmin`/`vmax` no longer silently changes whether the background shows
through.

:::{seealso}
Overlaying one image on another (anatomical + MRSI) is `GESeries`-level, not a bare-array
operation — see [The GESeries class hierarchy § NiiBase operations](#data-model-geseries-niibase-ops)
and [Drawing an MRSI voxel on an anatomical slice](#nifti-voxel-overlay).
:::
