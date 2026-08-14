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

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

# Crisp retina output + sane default DPI for the rendered docs
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150

from loguru import logger

logger.remove()
```

::: {dropdown} Imports & random number seed & 4D volume generation

```{code-cell} ipython3
import matplotlib.pyplot as plt
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
- **4D** `(H, W, S, N)` — a **grid of `N` images**, laid out up to 4 per row. Each of the `N` still
  only shows _its own_ middle slice along axis 2 — the same "last axis of each sub-array is
  slices, not images" rule applies one level down.

```{code-cell} ipython3
single = volume_4d[:, :, num_z // 2, 1]  # single slice of the 4D volume
fig, ax = display_images(single, aspect=1)
plt.show()
```

```{code-cell} ipython3
volume_4d[:, :, num_z // 2, 1:5].shape
```

```{code-cell} ipython3
one_slice_grid = rng.integers(-100, 20, size=(20, 20, 1, 7))
# fig, ax = display_images(
#     volume_4d[:, :, num_z // 2, 1:3], aspect=1
# )  # grid of 7 slices along the 4th dimension
# plt.show()
print("4D, one slice (grid of 7):", ax.shape)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: display_images axis semantics
# assert ax.shape == (8,)  # 7 images, padded to a 2x4 grid -- one subplot unused
# fig, ax_single = display_images(single)
# plt.close(fig)
# assert ax_single.shape == (1,)
```

A **list** of same-shaped arrays works too — it's stacked along a new last axis before the same
rule applies (more on that [below](#plotting-images-scrub), since a list is often exactly how a
multi-slice case shows up).

(plotting-images-scrub)=

## But a volume has more than one slice — can I see the rest?

The examples above all used a single slice to keep `ax.shape` checkable. Real volumes don't stop
at one: a `(H, W, S)` array with `S > 1` still shows only its middle slice by default — so what
about the other `S - 1`?

`display_images()` shows them too, as a slider, right here on this page — not just in a live
Jupyter session. Once there's more than one slice, the function displays an interactive
slice-viewer widget instead of a static image, and returns `None`: there's no `fig`/`axes` to hand
back, because the "figure" is no longer a single static thing.

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
fig, ax = display_images(list_of_single_slice_volumes)
plt.show()
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: list input follows np.stack(..., axis=-1), then the same shape rule
fig, ax_list3d = display_images(list_of_single_slice_volumes)
plt.close(fig)
assert ax_list3d.shape == (4,)  # stacked to 4D -> grid of 4
```

:::{warning}
Every array in a list must share the same shape — `display_images` stacks them with
`np.stack(images, axis=-1)`, which raises `ValueError` on a mismatch. There's no fallback path
for comparing differently-shaped images side by side; resize or crop to a common shape first.
:::

(plotting-images-zeros)=

## Zeros vs. real signal

Background voxels are often exactly `0`, not a small real value — treating them as data skews a
colorbar built from `vmin`/`vmax`/percentile clipping toward the background instead of the signal
you actually care about. `zeros_as_nan=True` masks them out before scaling:

```{code-cell} ipython3
mostly_zero = np.zeros((20, 20))
mostly_zero[8:12, 8:12] = rng.random((4, 4)) * 100

fig, ax = display_images(mostly_zero, zeros_as_nan=True, colorbar=True)
plt.show()
```

:::{seealso}
Overlaying one image on another (anatomical + MRSI) is `GESeries`-level, not a bare-array
operation — see [The GESeries class hierarchy § NiiBase operations](#data-model-geseries-niibase-ops)
and [Drawing an MRSI voxel on an anatomical slice](#nifti-voxel-overlay).
:::
