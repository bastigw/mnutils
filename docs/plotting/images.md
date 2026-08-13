---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (mnutils)
  language: python
  name: python3
---

(plotting-images)=
# Displaying image arrays of any shape

> **I have an array — could be one 2D slice, could be a volume, could be several volumes — and I
> just want to look at it. Do I need a different plotting call for each shape?**

No — [`display_images()`](#mnutils.plotting.images.display_images) takes 2D, 3D or 4D input and
picks a sensible layout automatically. What changes with each extra dimension isn't "more images
shown" the way you might expect, though — it's worth knowing the actual rule before it surprises
you.

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib

matplotlib.use("Agg")  # headless: this page never opens a window

from loguru import logger

logger.remove()
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

from mnutils.plotting.images import display_images
```

(plotting-images-rule)=
## The rule: the last axis is "how many images"; everything else is "one volume"

- **2D** `(H, W)` — one image.
- **3D** `(H, W, S)` — still **one image**: the *middle slice* along the last axis, `S // 2`. Not
  a grid of `S` images.
- **4D** `(H, W, S, N)` — a **grid of `N` images**, laid out up to 4 per row. Each of the `N` still
  only shows *its own* middle slice along axis 2 — the same "last axis of each sub-array is
  slices, not images" rule applies one level down.

```{code-cell} ipython3
single = np.random.randint(-10, 20, size=(20, 20))
fig, ax = display_images(single)
plt.close(fig)
print("2D:", ax.shape)

volume = np.random.randint(-10, 20, size=(20, 20, 6))
fig, ax = display_images(volume)
plt.close(fig)
print("3D (one volume, middle slice shown):", ax.shape)

grid = np.random.randint(-100, 20, size=(20, 20, 3, 7))
fig, ax = display_images(grid)
plt.close(fig)
print("4D (grid of 7):", ax.shape)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: display_images axis semantics
assert ax.shape == (8,)  # 7 images, padded to a 2x4 grid -- one subplot unused
fig, ax_single = display_images(single)
plt.close(fig)
assert ax_single.shape == (1,)
fig, ax_vol = display_images(volume)
plt.close(fig)
assert ax_vol.shape == (1,)  # still one image, not six
```

A **list** of same-shaped arrays works too — it's stacked along a new last axis before the same
rule applies. A list of 2D arrays becomes 3D (one image, the middle *array* of the list shown, not
a grid); a list of *volumes* (each already 3D) becomes 4D (a grid, one per volume):

```{code-cell} ipython3
list_of_images = [np.full((10, 10), i, dtype=float) for i in range(4)]
fig, ax = display_images(list_of_images)
plt.close(fig)
print("list of 2D arrays:", ax.shape)

list_of_volumes = [np.full((10, 10, 3), i, dtype=float) for i in range(4)]
fig, ax = display_images(list_of_volumes)
plt.close(fig)
print("list of volumes:", ax.shape)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: list input follows np.stack(..., axis=-1), then the same shape rule
fig, ax_list2d = display_images(list_of_images)
plt.close(fig)
assert ax_list2d.shape == (1,)  # stacked to 3D -> one image
fig, ax_list3d = display_images(list_of_volumes)
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
mostly_zero[8:12, 8:12] = np.random.rand(4, 4) * 100

fig, ax = display_images(mostly_zero, zeros_as_nan=True, colorbar=True)
plt.close(fig)
```

:::{seealso}
Overlaying one image on another (anatomical + MRSI) is `GESeries`-level, not a bare-array
operation — see [The GESeries class hierarchy § NiiBase operations](#data-model-geseries-niibase-ops)
and [Drawing an MRSI voxel on an anatomical slice](#nifti-voxel-overlay).
:::
