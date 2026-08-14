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

(nifti-voxel-overlay)=
# Drawing an MRSI voxel on an anatomical slice

> **I have one MRSI voxel's index `[i, j, k]`. Which pixel does it land on in a T1 slice, and
> how big a box do I draw around it?**

Both questions come down to one thing: a NIfTI affine maps a voxel *index* to the *world-space
center* of that voxel, in millimeters — never a corner. Every function on this page exists to get
that one mapping right, because the two ways to get it *subtly* wrong (an off-by-half-voxel
offset, a double-applied shift) don't crash — they draw a box next to the right answer, which is
much harder to notice.

| Function | What it does here |
|---|---|
| [`get_voxel_visible_slice_range()`](#mnutils.plotting.images.get_voxel_visible_slice_range) | which T1 slices a given MRSI voxel is visible on |
| [`draw_voxel_overlays_on_ax`](#mnutils.plotting.images) | draws the voxel box, given an MRSI→display affine |

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

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import affines

from mnutils.plotting.images import draw_voxel_overlays_on_ax, get_voxel_visible_slice_range
```

(nifti-voxel-overlay-mapping)=
## Mapping an MRSI voxel index to a T1 voxel index

`mrsi2anat = inv(T1.affine) @ MRSI.affine` composes the two affines into one: MRSI voxel index in,
T1 voxel index out. Applying it to voxel `[0, 0, 0]` should land exactly on the world-space center
of that voxel, converted into T1's index space — the same answer you'd get by hand from each
affine separately.

```{code-cell} ipython3
t1_affine = np.diag([0.5, 0.5, 3.0, 1.0])
t1_affine[:3, 3] = [-100.0, -80.0, -90.0]

mrsi_affine = np.diag([12.5, 12.5, 12.5, 1.0])
mrsi_affine[:3, 3] = [-50.0, -40.0, -30.0]  # voxel 0 center, in world mm

mrsi2anat = np.linalg.inv(t1_affine).dot(mrsi_affine)

voxel = (3, 5, 2)
result = affines.apply_affine(mrsi2anat, voxel)
print(result)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: mrsi2anat affine composition, checked independently per axis
world_center = mrsi_affine[:3, 3] + np.diag(mrsi_affine[:3, :3]) * voxel
expected_t1 = (world_center - t1_affine[:3, 3]) / np.diag(t1_affine[:3, :3])
np.testing.assert_allclose(result[:3], expected_t1, atol=1e-6)
```

(nifti-voxel-overlay-box)=
## Drawing the box: corner = center minus half-width

`matplotlib.patches.Rectangle` takes a *corner*, not a center — so the box's corner has to be the
voxel's world center, offset back by half the voxel's own display width. Get that offset wrong by
even a little and the box still *looks* plausible; it's just centered on the wrong point.

```{code-cell} ipython3
mrsi2anat_2d = np.diag([25.0, 25.0, 4.17, 1.0])
mrsi2anat_2d[:3, 3] = [68.39, 71.91, 9.83]

voxel_in_anat = affines.apply_affine(mrsi2anat_2d, (3, 4, 5))
correct_offset = mrsi2anat_2d[0, 0] / 2
correct_corner = voxel_in_anat[1] - correct_offset
print(correct_corner, "-> center", correct_corner + correct_offset, "vs", voxel_in_anat[1])
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: corner + half-width recovers the center exactly
assert correct_corner + mrsi2anat_2d[1, 1] / 2 == pytest.approx(voxel_in_anat[1])
```

❌ **The bug this guards against:** a hardcoded offset (`affine[0, 0] - 13`, tuned to look right
for one dataset's voxel size) instead of `affine[0, 0] / 2`. The two agree only by coincidence
for the voxel size they were tuned against — 12 vs. the correct 12.5 above, half a T1 voxel off
for every other dataset.

✅ **The rule:** the offset is always `voxel_size / 2`, derived from the affine itself, never a
constant.

`draw_voxel_overlays_on_ax` gets this right even when the display affine transposes the axes (a
common case: MRSI axis 0 maps to the display *column*, not the row) — the voxel extents then live
**off-diagonal**, so reading `affine[0, 0]`/`affine[1, 1]` directly for the box size would silently
give zero:

```{code-cell} ipython3
vs = 25.0
mrsi_to_display = np.array(
    [
        [0.0, -vs, 0.0, 440.0],
        [-vs, 0.0, 0.0, 444.0],
        [0.0, 0.0, 4.17, 9.83],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
assert mrsi_to_display[0, 0] == 0 and mrsi_to_display[1, 1] == 0  # off-diagonal, on purpose

fig, ax = plt.subplots()
polys = draw_voxel_overlays_on_ax(ax, [(3, 4, 5)], mrsi_to_display)
plt.show()

xy = polys[0].get_xy()[:4]
width = xy[:, 0].max() - xy[:, 0].min()
height = xy[:, 1].max() - xy[:, 1].min()
print(width, height)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: box size survives an anti-diagonal (transposed) display affine
assert width == pytest.approx(vs)
assert height == pytest.approx(vs)
```

(nifti-voxel-overlay-slices)=
## Which T1 slices a voxel is visible on

An MRSI voxel is thick — it spans several thin T1 slices along the through-plane axis.
`get_voxel_visible_slice_range` returns the `(min, max)` T1 slice indices the voxel covers,
always including the slice nearest its own center, and clamped to `[0, n_slices - 1]` so a voxel
near the edge of the volume never returns an out-of-bounds index.

```{code-cell} ipython3
mrsi2anat_z = np.diag([25.0, 25.0, 4.17, 1.0])

center_slice = get_voxel_visible_slice_range((5, 5, 5), mrsi2anat_z)
print("voxel [5,5,5] visible on T1 slices", center_slice)

edge_clamped = get_voxel_visible_slice_range((0, 0, 0), mrsi2anat_z, n_slices=76)
print("voxel [0,0,0] visible on T1 slices", edge_clamped, "(clamped to 76 slices)")
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: get_voxel_visible_slice_range
voxel_center_z = affines.apply_affine(mrsi2anat_z, (5, 5, 5))[2]
assert center_slice[0] <= int(voxel_center_z) <= center_slice[1]
assert edge_clamped[0] >= 0
assert edge_clamped[1] <= 75
```

(nifti-voxel-overlay-mrsi-affine)=
## `create_MRSI_affine`: why the shift has two terms

[`MRSISeries.create_MRSI_affine()`](#data-model-geseries-niibase-ops) builds the "blocky" grid
affine (one cell per acquired voxel) from the fine, interpolated `nii.affine` it started from.
Scaling the voxel size alone gets the field of view right but leaves the grid offset by half a
fine voxel — so the origin shift is applied in two parts: first to re-center for the size change,
then an *additional* half of the new (blocky) voxel size on top.

```{code-cell} ipython3
from pathlib import Path

from mnutils.GESeries import MRSISeries


def _repo_root(start: Path = Path.cwd()) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root (no pyproject.toml found)")


mrsi = MRSISeries(_repo_root() / "tests" / "datasets" / "20250408-NIST-Mag2" / "data", 5)

fine_affine = mrsi.nii.affine
blocky_affine = mrsi.create_MRSI_affine()

resize_shift = (np.diag(blocky_affine[:3]) - np.diag(fine_affine[:3])) / 2
half_voxel_shift = np.array([blocky_affine[0, 0], blocky_affine[1, 1], 0]) / 2
total_shift = resize_shift + half_voxel_shift

print(np.allclose(blocky_affine[:3, 3], fine_affine[:3, 3] + total_shift, atol=1e-3))
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: create_MRSI_affine's two-term shift, and RAW_exp uses the same affine
np.testing.assert_allclose(blocky_affine[:3, 3], fine_affine[:3, 3] + total_shift, atol=1e-3)
np.testing.assert_allclose(mrsi.RAW_exp.nii.affine, blocky_affine, atol=1e-4)
```

`MRSISeries.RAW_exp` (the blocky NIfTI used for overlay display) is built from exactly this
affine — the assertion above is what keeps them from drifting apart.

(nifti-voxel-overlay-cropping)=
## Cropping the display without breaking the voxel box

`draw_voxel_overlays_on_ax` always computes the box in **full, uncropped display coordinates** —
it only depends on `mrsi_to_display_affine`, never on the image array you actually plotted. Crop
the anatomical/MRSI images to a tighter figure and the box drawn by the un-cropped affine now
points at the wrong pixel *unless* you shift it back by the crop's own offset.

```{code-cell} ipython3
from mnutils.GESeries import MRISeries, MRSISeries
from mnutils.plotting.images import draw_voxel_overlays_on_ax, overlay_image_data_on_T1_on_ax
from mnutils.utils.nifti import get_display_affine, resample_and_orient_nifti

hevo18_data = _repo_root() / "tests" / "datasets" / "HeVo-18" / "data"
t1 = MRISeries(hevo18_data, 2)
mrsi_brain = MRSISeries(hevo18_data, 8)

t1_full, mrsi_full = resample_and_orient_nifti(source_nii=mrsi_brain.RAW_exp.nii, target_nii=t1.nii)
mrsi_to_display = np.linalg.inv(get_display_affine(t1.nii)).dot(
    np.asarray(mrsi_brain.RAW_exp.nii.affine)
)

demo_voxel = (5, 5, 8)
y_slice, x_slice = slice(150, 400), slice(150, 400)
x_offset, y_offset = x_slice.start, y_slice.start

fig, ax = plt.subplots(figsize=(4, 4))
overlay_image_data_on_T1_on_ax(
    t1_full[y_slice, x_slice, demo_voxel[2]],
    mrsi_full[y_slice, x_slice, demo_voxel[2]],
    ax=ax,
)
(patch,) = draw_voxel_overlays_on_ax(ax, demo_voxel, mrsi_to_display_affine=mrsi_to_display)
# Shift the box from full-image coordinates into the cropped axes' coordinates.
patch.set_xy(patch.get_xy() - np.array([x_offset, y_offset]))
plt.show()

xy = patch.get_xy()[:4]
print(xy.min(axis=0) >= 0, xy.max(axis=0) <= [x_slice.stop - x_offset, y_slice.stop - y_offset])
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: shifted patch lands inside the cropped axes' bounds
assert (xy.min(axis=0) >= 0).all()
assert (xy[:, 0].max() <= x_slice.stop - x_offset)
assert (xy[:, 1].max() <= y_slice.stop - y_offset)
```

The recipe, generalized:

1. Build `mrsi_to_display_affine = inv(get_display_affine(T1)) @ MRSI.RAW_exp.nii.affine`.
2. Crop both anatomical and (resampled) MRSI images with the same `(y_slice, x_slice, z)` bounds.
3. Plot the cropped images with `overlay_image_data_on_T1_on_ax`.
4. Draw the voxel box with `draw_voxel_overlays_on_ax(ax, voxel, mrsi_to_display_affine, ...)` —
   still in full-image coordinates.
5. Shift the patch: `patch.set_xy(patch.get_xy() - np.array([x_offset, y_offset]))`.

For a single voxel with no cropping, [`overlay_voxel_on_T1()`](#mnutils.plotting.images) wraps
steps 1, 3 and 4 into one call:

```{code-cell} ipython3
from mnutils.plotting.images import overlay_voxel_on_T1

fig, ax = overlay_voxel_on_T1(t1, mrsi_brain, demo_voxel, figsize=(4, 4))
plt.show()
```

:::{seealso}
[The GESeries class hierarchy](#data-model-geseries) covers `create_MRSI_affine` and `RAW_exp` as
part of `MRSISeries` generally; this page is only about the geometry math underneath them.
:::
