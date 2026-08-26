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

(nifti-synthetic-gallery)=
# Every fake dataset, checked against its own ground truth

> **Two independent bugs just silently misplaced every MRSI overlay this fixture pipeline
> produced — a shift-math error in the affine build, and an array reorder that decoupled data
> from its own header. Both were caught by chasing *one* dataset (`brain_mrs_mrsi_exam`) by hand.
> What about the other four?**

`mnutils.testing.build_fake_exam()` backs every docs page and test in this repo (see
[the dev-diary entry](#diary-synthetic-exam-fixtures) for why it's fake and how its *content* is
made real). That reach is exactly the risk: a silent regression in the generator doesn't fail
loudly in one place, it quietly drifts every page that reads from it. This page is the single
place that loads **all five** fixtures and checks each one against ground truth computed
*independently* of the code path being tested — not by eyeballing a plot, by comparing to a
position or ordering derived a different way.

| Function | What it does here |
|---|---|
| [`MRISeries`](#mnutils.GESeries.MRISeries) | loads the anatomical NIfTI for each exam |
| [`MRSISeries`](#mnutils.GESeries.MRSISeries) | loads the spectroscopic grid; `.RAW_exp` is the blocky display NIfTI checked below |
| [`overlay_nifti_data_on_T1()`](#mnutils.plotting.images.overlay_nifti_data_on_T1) | the visual half of each geometric check |
| [`get_mat_data_from_series()`](#mnutils.utils.file_helpers.get_mat_data_from_series) | series-ID lookup checked on the missing-series fixture |
| [`load_mat_file()`](#mnutils.utils.data_loaders.load_mat_file) | loads the legacy-format fixture |

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
from loguru import logger

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150
logger.remove()
```

```{code-cell} ipython3
import numpy as np
from nibabel import affines
from scipy.optimize import linear_sum_assignment

from mnutils.GESeries import MRISeries, MRSISeries
from mnutils.plotting.images import overlay_nifti_data_on_T1
from mnutils.testing import build_fake_exam
```

(nifti-synthetic-gallery-brain)=
## `brain_mrs_mrsi_exam`: a real, downsampled T1 as ground truth

The MRSI grid's per-voxel signal comes from the real T1 template, downsampled by
`template_grid_intensity()`. That gives an independent way to check `RAW_exp`'s affine: pick a
grid index, work out where `template_grid_intensity()`'s downsampling *actually* placed it by
going back to the full-resolution template directly (not through any of the code under test), and
compare to what `RAW_exp.nii.affine` says that index's world position is.

```{code-cell} ipython3
brain_data = build_fake_exam("brain_mrs_mrsi_exam") / "data"
t1_brain = MRISeries(brain_data, 2)
mrsi_brain = MRSISeries(brain_data, 8)
```

Overlay of `RAW_exp` on the T1 shows the grid correctly placed on the anatomy.

```{code-cell} ipython3
overlay_nifti_data_on_T1(t1_brain.nii, mrsi_brain.RAW_exp.nii)
```

Overlay of the interpolated MRSI grid 

```{code-cell} ipython3
overlay_nifti_data_on_T1(t1_brain.nii, mrsi_brain.nii)
```

```{code-cell} ipython3
from mnutils.testing._spectra import _cropped_template

template_data, template_affine = _cropped_template()
native_shape = np.array(template_data.shape)
grid_shape = np.array(mrsi_brain.dims)
raw_affine = np.asarray(mrsi_brain.RAW_exp.nii.affine)

# grid_mode=True zoom (used to build the grid) samples native index (idx+0.5)*(native/grid)-0.5
check_indices = [(0, 0, 0), (8, 4, 5), (15, 15, 15)]
errors_mm = []
for idx in check_indices:
    idx = np.array(idx)
    native_idx = (idx + 0.5) * (native_shape / grid_shape) - 0.5
    true_world = affines.apply_affine(template_affine, native_idx)
    raw_world = affines.apply_affine(raw_affine, idx)
    errors_mm.append(np.abs(raw_world - true_world).max())

print("max ground-truth error across sampled indices (mm):", max(errors_mm))
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: RAW_exp's affine matches the template's own downsampling, independently derived
assert max(errors_mm) < 0.01, f"brain_mrs_mrsi_exam RAW_exp geometry off by {max(errors_mm)} mm"
```

`mrsi_brain.nii` (the interpolated pseudo image plotted above, not `RAW_exp`) carries its *own*
affine straight from the fixture -- `RAW_exp`'s check above never exercises it. The interpolated
grid's index `j` corresponds to native template index `(j + 0.5) * (native/interp) - 0.5`, the
same `grid_mode=True` mapping as `RAW_exp`, just at the interpolated resolution instead of the
blocky one.

```{code-cell} ipython3
interp_shape = np.array(mrsi_brain.nii.shape)
interp_affine = np.asarray(mrsi_brain.nii.affine)

interp_errors_mm = []
for idx in check_indices:
    idx = np.array(idx)
    native_idx = (idx + 0.5) * (native_shape / interp_shape) - 0.5
    true_world = affines.apply_affine(template_affine, native_idx)
    interp_world = affines.apply_affine(interp_affine, idx)
    interp_errors_mm.append(np.abs(interp_world - true_world).max())

print("max ground-truth error, interpolated affine (mm):", max(interp_errors_mm))
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: the interpolated nii's own affine, independent of RAW_exp/create_MRSI_affine
assert max(interp_errors_mm) < 0.01, (
    f"brain_mrs_mrsi_exam interpolated geometry off by {max(interp_errors_mm)} mm"
)
```

(nifti-synthetic-gallery-nist)=
## `nist_phantom_exam`: eight spheres, exact analytic positions

The NIST phantom places its spheres analytically — `SphereRing.spheres()` returns each sphere's
*exact* world-mm center and intensity, strictly increasing around the ring. That's ground truth
with no downsampling involved at all: every bright `RAW_exp` voxel should sit inside its sphere's
10&nbsp;mm radius, and brightness should rank in the same order as the assigned intensities.

```{code-cell} ipython3
from mnutils.testing._spectra import DEFAULT_SPHERE_RINGS

nist_data = build_fake_exam("nist_phantom_exam") / "data"
t1_nist = MRISeries(nist_data, 2)
mrsi_nist = MRSISeries(nist_data, 5)

overlay_nifti_data_on_T1(t1_nist.nii, mrsi_nist.RAW_exp.nii)
```

```{code-cell} ipython3
mrsi_nist.RAW_exp.display(colorbar=True)
```

```{code-cell} ipython3
ring = DEFAULT_SPHERE_RINGS[0]
spheres = ring.spheres()
centers = np.array([c for c, _ in spheres])
intensities = np.array([i for _, i in spheres])

raw_affine = np.asarray(mrsi_nist.RAW_exp.nii.affine)
raw_data = np.asarray(mrsi_nist.RAW_exp.nii.get_fdata())
bright_idx = np.array(np.where(raw_data > 0)).T
bright_world = affines.apply_affine(raw_affine, bright_idx)
bright_val = raw_data[tuple(bright_idx.T)]

# Optimal one-to-one sphere <-> bright-voxel assignment (nearest-neighbour alone breaks down
# when two spheres are each other's second-nearest voxel).
dist_matrix = np.linalg.norm(centers[:, None, :] - bright_world[None, :, :], axis=-1)
sphere_order, voxel_order = linear_sum_assignment(dist_matrix)
matched_dist = dist_matrix[sphere_order, voxel_order]
matched_val = bright_val[voxel_order]

print("n bright voxels:", len(bright_idx), "n spheres:", len(spheres))
print(
    "max sphere-to-voxel distance (mm):",
    matched_dist.max(),
    "/ sphere radius:",
    ring.sphere_radius_mm,
)
print(
    "brightness order matches intensity order:",
    np.array_equal(np.argsort(intensities), np.argsort(matched_val)),
)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: every sphere has exactly one bright voxel, geometrically inside it and correctly ranked
assert len(bright_idx) == len(spheres)
assert matched_dist.max() < ring.sphere_radius_mm
assert np.array_equal(np.argsort(intensities), np.argsort(matched_val))
```

(nifti-synthetic-gallery-shape-only)=
## The other three: shape-only fixtures

`brain_extraction_exam`, `mrsi_missing_series_exam`, and `legacy_matlab73_example` don't carry
MRSI geometry to check — they exist to exercise a *loading* code path, not a spatial one. Their
actual teaching pages are [Loading a series and an exam](#basics-loading-data),
[Finding files in a data folder](#basics-file-discovery), and
[Loading a `.mat` reconstruction](#basics-mat-files); this section only confirms each one still
produces the shape those pages (and `tests/test_brain_extract.py`, for the HD-BET path) expect.

```{code-cell} ipython3
brain_extraction_data = build_fake_exam("brain_extraction_exam") / "data"
t1_body = MRISeries(brain_extraction_data, 2)
print("brain_extraction_exam T1 body shape:", t1_body.nii.shape)

# mrsi_missing_series_exam has no "own" reference NIfTI at all -- it exists purely to exercise
# get_exam_folder/get_mat_data_from_series's series-ID lookup (issue #34), not full MRSISeries
# loading. Series5's FileNotFoundError case is already covered on file_discovery.md; here it's
# just Series7's lookup, confirming the fixture itself still resolves.
from mnutils.utils import file_helpers as fh

missing_series_root = build_fake_exam("mrsi_missing_series_exam")
series7_mat = fh.get_mat_data_from_series(missing_series_root, 7)
print("mrsi_missing_series_exam Series7 .mat:", series7_mat.name)

from mnutils.utils import data_loaders as dl

legacy_path = build_fake_exam("legacy_matlab73_example")
legacy_data = dl.load_mat_file(legacy_path)
print("legacy_matlab73_example keys:", list(legacy_data.keys()))
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: the three shape-only fixtures still load the way their own pages expect
assert t1_body.nii.shape[0] > 0 and len(t1_body.nii.shape) == 3
assert series7_mat.suffix == ".mat"
assert "pyr" in legacy_data
```

:::{seealso}
[The dev-diary entry](#diary-synthetic-exam-fixtures) covers why each fixture is fake (or not) and
how its content is built; [`create_MRSI_affine`: why the shift has two terms](#nifti-voxel-overlay-mrsi-affine)
covers the affine math this page's `brain_mrs_mrsi_exam`/`nist_phantom_exam` checks exercise.
:::
