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

(data-model-geseries)=

# The GESeries class hierarchy

A GE series folder can hold very different shapes of data depending on what was acquired: a 3D
anatomical volume, a single-voxel spectrum averaged over time, a spectroscopic-imaging grid, or a
dynamic wash-in series sampled every few seconds. `GESeries` isn't one class trying to cover all
of that — it's a small hierarchy, and [`ExamBase`](#basics-loading-data) picks the right subclass
for a folder so you don't have to guess it yourself.

| Class                                                  | What it adds                                                         |
| ------------------------------------------------------ | -------------------------------------------------------------------- |
| [`NiiBase`](#mnutils.GESeries.NiiBase)                 | wraps a NIfTI image: affine, `images()`, `display()`, brain masking  |
| [`MRISeries`](#mnutils.GESeries.MRISeries)             | + converts a DICOM series to NIfTI (anatomical scans)                |
| [`RawMRISeries`](#mnutils.GESeries.RawMRISeries)       | + the raw `.mat` reconstruction, FIDs, and scanner header properties |
| [`MRSSeries`](#mnutils.GESeries.MRSSeries)             | + single-voxel spectrum, `averages × chemical_shift`                 |
| [`MRSWashinSeries`](#mnutils.GESeries.MRSWashinSeries) | + grouped, time-resolved fitting for a wash-in acquisition           |
| [`MRSISeries`](#mnutils.GESeries.MRSISeries)           | + spectroscopic-imaging grid, `chemical_shift × i × j × k`           |

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart TD
    N["NiiBase<br>(NIfTI wrapper)"] --> M["MRISeries<br>(+ DICOM-to-NIfTI)"]
    M --> R["RawMRISeries<br>(+ raw .mat, FIDs, header)"]
    R --> S["MRSSeries<br>(+ single-voxel spectrum)"]
    R --> I["MRSISeries<br>(+ spectroscopic-imaging grid)"]
    S --> W["MRSWashinSeries<br>(+ time-resolved fitting)"]
```

The examples below load fixtures generated on the fly by `mnutils.testing.build_fake_exam` --
see [the diary entry](#diary-synthetic-exam-fixtures) for why nothing here is a real scan.

```{code-cell} ipython3
from loguru import logger

logger.remove()
```

```{code-cell} ipython3
from mnutils.GESeries import MRISeries, MRSISeries, MRSSeries, MRSWashinSeries, NiiBase
from mnutils.plotting.spectra import plot_fid, plot_spectra
from mnutils.testing import build_fake_exam
```

```{code-cell} ipython3
DATASETS = build_fake_exam("brain_mrs_mrsi_exam")
DATA_FOLDER = DATASETS / "data"
```

(data-model-geseries-niibase)=

## `NiiBase` — a NIfTI image, oriented and displayable

Every series that has DICOM data ends up with a `NiiBase` (or subclass) wrapping the converted
NIfTI: `nii`, `affine`, and helpers like `images()`/`display()` that apply a display orientation
consistently across the toolbox, so plotting code never re-derives it.

(data-model-geseries-mriseries)=

## `MRISeries` — DICOM series, converted automatically

`MRISeries` is what you get for a series that has DICOM but no raw scanner exam data — typically
an anatomical scan. It converts the DICOMs to NIfTI on init if no NIfTI exists yet.

:::{note}
The anatomical volumes in these fake exams are cropped from a real T1 scan — Chris Rorden's
`chris_t1`, from the [niivue-images](https://github.com/neurolabusc/niivue-images) sample set,
licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use only).
Downloaded and cached on first use by `mnutils.testing.build_fake_exam()` — see
[the diary entry](#diary-synthetic-exam-fixtures) for how everything else here is simulated.
:::

```{code-cell} ipython3
t1 = MRISeries(DATA_FOLDER, 2)  # 002_3D_Ax_T1_BRAVO
t1.display(cmap="gray")
print(t1.nii.shape)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: MRISeries
assert isinstance(t1, NiiBase)
assert t1.nii.ndim == 3
```

(data-model-geseries-mrsseries)=

## `MRSSeries` — a single-voxel spectrum over time

A series with both raw exam data and DICOM (or no DICOM at all, for a pure spectroscopy
acquisition) loads as one of the `RawMRISeries` subclasses instead. `MRSSeries` holds a
single-voxel spectrum with an `averages` dimension — one spectrum per repetition, not yet
combined.

```{code-cell} ipython3
mrs = MRSSeries(DATA_FOLDER, 7)  # 006_MRS_unloc
mrs.spec
```

```{code-cell} ipython3
plot_spectra(mrs.avg_spec)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: MRSSeries
assert mrs.spec.dims == ("averages", "chemical_shift")
assert mrs.spec.sizes["averages"] == mrs.averages
assert mrs.nucleus == 2
```

`avg_fid`/`avg_spec` collapse the `averages` dimension; see
[Loading a series and an exam](#basics-loading-data) for a worked example.

(data-model-geseries-washin)=

## `MRSWashinSeries` — the same spectrum, resolved over time

A wash-in acquisition is structurally an `MRSSeries` — same `averages × chemical_shift` shape —
but its averages are meant to be grouped into time bins and fit independently, tracking a
metabolite as it washes in. `MRSWashinSeries` adds `fit_grouped_by_duration()` and
`plot_washin()` on top of everything `MRSSeries` already gives you.

```{code-cell} ipython3
washin = MRSWashinSeries(DATA_FOLDER, 7)  # 007_MRS_washin
washin.spec
```

```{code-cell} ipython3
washin.plot_washin(group_duration=60)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: MRSWashinSeries
assert isinstance(washin, MRSSeries)
assert washin.spec.dims == ("averages", "chemical_shift")
assert washin.fit_results == []  # nothing fit yet
```

(data-model-geseries-mrsiseries)=

## `MRSISeries` — a spectroscopic-imaging grid

`MRSISeries` is the spatial counterpart: a spectrum _per voxel_, laid out on a 3D grid
(`chemical_shift × i × j × k`). It carries per-voxel fitting (`fit_all_voxels`,
`fit_single_voxel`) and an SNR mask used to skip voxels with no real signal.

```{code-cell} ipython3
mrsi = MRSISeries(DATA_FOLDER, 8)  # 008_MRSI_pseudo_S700_X10_Y10_Z10_T1_C1
print(mrsi.spec.dims, mrsi.spec.shape)
print(mrsi.dims)  # spatial grid (i, j, k)
mrsi.spec
```

```{code-cell} ipython3
from mnutils.plotting.images import inspect_MRSI_spectra

inspect_MRSI_spectra(t1, mrsi)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: MRSISeries
assert mrsi.spec.dims == ("chemical_shift", "i", "j", "k")
assert mrsi.dims == mrsi.spec.shape[1:4]
assert isinstance(mrsi, NiiBase)  # create_MRSI_nii() gives it a displayable volume too
```

(data-model-geseries-niibase-ops)=

## `NiiBase` operations: resampling and overlays

Two `NiiBase` images at different resolutions come up constantly — an anatomical scan and an
MRSI grid, say — and need to be compared voxel-for-voxel. `resample_self_to` reslices one onto
another's grid; `overlay_nifti_data_on_T1` (from `mnutils.plotting.images`) plots the result on
top of an anatomical volume. These examples use the fake `20250408-NIST-Mag2` exam, a phantom
scan where series 2 is the anatomical T1 and series 5 is an MRSI grid.

```{code-cell} ipython3
resampled = mrsi.resample_self_to(t1.nii, order=0)
print(isinstance(resampled, NiiBase), resampled.nii.shape == t1.nii.shape)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: resample_self_to
assert isinstance(resampled, NiiBase)
assert resampled.nii.shape == t1.nii.shape
```

`images()` applies the display orientation and returns a plain array — what `display()` and the
overlay helpers plot under the hood:

```{code-cell} ipython3
from mnutils.plotting.images import overlay_nifti_data_on_T1

overlay_nifti_data_on_T1(t1.nii, mrsi.nii)
```

```{code-cell} ipython3
overlay_nifti_data_on_T1(t1.nii, mrsi.RAW_exp.nii)
```

`MRSISeries` carries a few more properties worth knowing about beyond `spec`: an SNR-based mask
for skipping low-signal voxels, a computed voxel size, and per-voxel spectrum access.

```{code-cell} ipython3
print(mrsi.voxel_size)  # mm, derived from field_of_view / matrix size
print(mrsi.SNR_map.dims)
voxel_spectrum = mrsi.get_voxel_spectrum(10, 10, 8)
plot_spectra(voxel_spectrum)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: MRSISeries extras
assert mrsi.SNR_map.dims == ("i", "j", "k")
assert mrsi.create_MRSI_affine().shape == (4, 4)
assert voxel_spectrum.dims == ("chemical_shift",)
```

:::{seealso}
Which subclass a given series folder resolves to is decided by [`ExamBase`](#basics-loading-data),
not by you calling one of these constructors directly in normal use — see
[Loading a series and an exam](#basics-loading-data) for the usual entry point.
:::
