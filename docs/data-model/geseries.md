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

| Class | What it adds |
|---|---|
| [`NiiBase`](#mnutils.GESeries.NiiBase) | wraps a NIfTI image: affine, `images()`, `display()`, brain masking |
| [`MRISeries`](#mnutils.GESeries.MRISeries) | + converts a DICOM series to NIfTI (anatomical scans) |
| [`RawMRISeries`](#mnutils.GESeries.RawMRISeries) | + the raw `.mat` reconstruction, FIDs, and scanner header properties |
| [`MRSSeries`](#mnutils.GESeries.MRSSeries) | + single-voxel spectrum, `averages × chemical_shift` |
| [`MRSWashinSeries`](#mnutils.GESeries.MRSWashinSeries) | + grouped, time-resolved fitting for a wash-in acquisition |
| [`MRSISeries`](#mnutils.GESeries.MRSISeries) | + spectroscopic-imaging grid, `chemical_shift × i × j × k` |

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart TD
    N["NiiBase<br>(NIfTI wrapper)"] --> M["MRISeries<br>(+ DICOM-to-NIfTI)"]
    M --> R["RawMRISeries<br>(+ raw .mat, FIDs, header)"]
    R --> S["MRSSeries<br>(+ single-voxel spectrum)"]
    R --> I["MRSISeries<br>(+ spectroscopic-imaging grid)"]
    S --> W["MRSWashinSeries<br>(+ time-resolved fitting)"]
```

The examples below load real fixtures from `tests/datasets/`.

```{code-cell} ipython3
from pathlib import Path

from mnutils.GESeries import MRISeries, MRSISeries, MRSSeries, MRSWashinSeries, NiiBase


def _repo_root(start: Path = Path.cwd()) -> Path:
    # Anchor on pyproject.toml rather than a relative "../.." count: this page
    # is executed from two different working directories -- docs/data-model/
    # when mystmd builds it, tests/autogen_notebooks/data-model/ when nbmake
    # runs the notebook `uv run test-gen` generates from it.
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root (no pyproject.toml found)")


DATASETS = _repo_root() / "tests" / "datasets"
DATA_FOLDER = DATASETS / "HeVo-18" / "data"
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

```{code-cell} ipython3
t1 = MRISeries(DATA_FOLDER, 2)  # 002_3D_Ax_T1_BRAVO
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
mrs = MRSSeries(DATA_FOLDER, 6)  # 006_MRS_unloc
print(mrs.spec.dims, mrs.spec.shape)
print(mrs.nucleus, mrs.centre_frequency)  # 2 => deuterium (2H)
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
print(washin.spec.dims, washin.spec.shape)
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

`MRSISeries` is the spatial counterpart: a spectrum *per voxel*, laid out on a 3D grid
(`chemical_shift × i × j × k`). It carries per-voxel fitting (`fit_all_voxels`,
`fit_single_voxel`) and an SNR mask used to skip voxels with no real signal.

```{code-cell} ipython3
mrsi = MRSISeries(DATA_FOLDER, 8)  # 008_MRSI_pseudo_S700_X10_Y10_Z10_T1_C1
print(mrsi.spec.dims, mrsi.spec.shape)
print(mrsi.dims)  # spatial grid (i, j, k)
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
top of an anatomical volume. These examples use `tests/datasets/20250408-NIST-Mag2/`, a phantom
scan where series 2 is the anatomical T1 and series 5 is an MRSI grid.

```{code-cell} ipython3
from mnutils.plotting.images import overlay_nifti_data_on_T1

phantom_folder = DATASETS / "20250408-NIST-Mag2" / "data"
t1 = MRISeries(phantom_folder, 2)
mrsi_phantom = MRSISeries(phantom_folder, 5)

resampled = mrsi_phantom.resample_self_to(t1.nii, order=0)
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
:tags: [remove-cell]

images = t1.images()
assert images.shape == t1.nii.shape

overlay_nifti_data_on_T1(t1.nii, mrsi_phantom.nii)  # STRICT TEST: runs without error
```

`MRSISeries` carries a few more properties worth knowing about beyond `spec`: an SNR-based mask
for skipping low-signal voxels, a computed voxel size, and per-voxel spectrum access.

```{code-cell} ipython3
print(mrsi_phantom.voxel_size)  # mm, derived from field_of_view / matrix size
print(mrsi_phantom.SNR_map.dims)
voxel_spectrum = mrsi_phantom.get_voxel_spectrum(0, 0, 0)
print(voxel_spectrum.dims)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: MRSISeries extras
assert mrsi_phantom.SNR_map.dims == ("i", "j", "k")
assert mrsi_phantom.create_MRSI_affine().shape == (4, 4)
assert voxel_spectrum.dims == ("chemical_shift",)
```

:::{seealso}
Which subclass a given series folder resolves to is decided by [`ExamBase`](#basics-loading-data),
not by you calling one of these constructors directly in normal use — see
[Loading a series and an exam](#basics-loading-data) for the usual entry point.
:::
