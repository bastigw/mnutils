(diary-synthetic-exam-fixtures)=
# CI can't see a scanner, so the test exam has to be born on disk

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-19</span>

CI (#3) ran green locally and failed for real: `tests/datasets/` is `.gitignore`d, so `HeVo-18`,
`HeVo-23`, `LG_D19`, and `20250408-NIST-Mag2` didn't exist on the runner, and 8 tests/notebooks
that hard-coded paths into them died on `ValueError: The base folder does not exist`. Uploading
them wasn't a small fix either — they're real scanner acquisitions with unclear redistribution
provenance, several MB each, and brittle to extend (issue #10). But the first pass at faking them
filled every `.mat`/`.nii` with plain random noise, which only gets a docs page to *run* — a
spectrum plot of white noise or a brain-shaped blob of static doesn't teach a reader anything about
what `mrs.spec` or `t1.nii` actually look like, and every "which voxel is brighter" example had
nothing real to point at.

:::{important}
`mnutils.testing.build_fake_exam()` fabricates every on-disk shape the real datasets used to
provide, but the *content* is real where it's cheap to make real: spectra come from
`xmris.simulate_fid` (HDO/Glucose/Glx peaks, not noise), brain anatomy is cropped from a real
downloaded T1 template, and the phantom dataset is a ring of 8 spheres with strictly increasing
signal intensity that the spectral simulation reads directly off.
:::

(diary-synthetic-exam-fixtures-shape)=
## What has to be faked, and what doesn't

The loaders need far less real-looking *structure* than the dataset names suggest — but the
*content* is worth getting right, since these fixtures are what every docs-page reader actually
sees:

| Real thing | What the code actually reads | Fake used |
|---|---|---|
| DICOM series | `get_dicom_folder` only checks the folder name (`002_...`); `MRISeries` loads any `.nii.gz` already sitting in that folder before ever trying DICOM conversion | a real anatomical NIfTI (see below), zero real DICOM bytes |
| `ScanArchive*.mat` | `RawMRISeries` parses `h.rdb_hdr`/`h.series`/`h.exam`/`h.image`/`h.mrconfig` plus `spec`/`bb` out of a nested MATLAB struct | a plain nested `dict` through `scipy.io.savemat` — round-trips into the `mat_struct` shape `load_mat_file`/`_todict` expect, `spec` filled by `xmris.simulate_fid` |
| `ScanArchive*.h5` (raw P-file) | `load_raw_fids` only needs the file to *exist* — a `Series{id}_raw_fids.h5` cache sitting next to it is read directly and the archive's contents are never touched | an empty stub `.h5` + a pre-built cache `.h5` (still random noise — nothing reads these FIDs for their physics, only their shape) |
| Legacy v7.3 `.mat` (`mat_files.md`'s `mat73` fallback case) | `mat73.loadmat` reads any HDF5 file whose datasets carry a `MATLAB_class` attribute | a bare `h5py.File` with one dataset and that one attribute set |

```python
# the call site every doc page and test ends up with
from mnutils.testing import build_fake_exam

DATASETS = build_fake_exam()  # generated once per process, cleaned up at exit
t1 = MRISeries(DATASETS / "HeVo-18" / "data", 2)  # real, cropped T1 template
mrsi = MRSISeries(DATASETS / "HeVo-18" / "data", 8)  # HDO/Glucose/Glx, sharper inside the brain
```

(diary-synthetic-exam-fixtures-spectra)=
## Spectra: `xmris.simulate_fid`, not noise

`mnutils.testing._spectra.simulate_spectrum` places four peaks from one `DMI_TRUTH` dict — HDO
(4.70 ppm), Glucose (3.80 ppm), Glx (2.40 ppm), and a broad `Baseline` component (1.9 ppm, 70 Hz
linewidth) standing in for the macromolecular background every real spectrum sits on top of — each
with its own illustrative relative amplitude and linewidth (HDO dominant, as in real 2H-MRSI), via
`xmris.simulate_fid` + `xmris.to_spectrum`. Every voxel/average takes two knobs: `intensity` (peak
amplitude and target SNR) and `clarity` (a linewidth multiplier on top of each peak's own
linewidth — higher clarity means a narrower, cleaner peak). Both feed a spatial map instead of
being constant:

- **`HeVo-18`/`HeVo-23`'s brain series**: the MRSI series' own NIfTI is the real T1 template
  downsampled onto the spec grid — that same downsampled array *is* the per-voxel
  intensity/clarity map, so "inside the brain" literally means "where the real template has
  signal."
- **`20250408-NIST-Mag2`**: a `SphereRing` (8 spheres, evenly spaced around a 60 mm-radius circle,
  offset 20 mm off the z=0 plane) assigns each sphere a distinct intensity, linearly increasing
  around the ring; grid cells inside a sphere take that sphere's intensity, everything else is
  near-silent.

Every repetition/voxel also gets a small, independently-seeded B0 (`b0_ppm`) and zero-order phase
(`phase_deg`) jitter — real MRS drifts shot-to-shot, real MRSI drifts voxel-to-voxel, and a stack of
spectra that all sit at exactly 0 ppm/0° would look more consistent than any real acquisition. SNR
realism is split by how many transients actually back a spectrum: `simulate_averages` (the
`MRS_unloc`/`MRS_washin` series) scales its target SNR by `sqrt(averages / _REFERENCE_TRANSIENTS)`,
so `MRS_unloc`'s real 64-transient acquisition count lands well above a single-shot voxel;
`simulate_grid` (MRSI) has no such boost — each voxel is effectively single-transient, so it draws
from a separately calibrated, ~10x lower `_BASE_SNR_MRSI` noise floor instead of the unloc-side
`_BASE_SNR_UNLOC`.

```python
# adding a second range of spheres with a different spectral profile -- SphereRing's job
from mnutils.testing._spectra import PEAKS_PPM, SphereRing

second_ring = SphereRing(
    name="lactate_ring",
    ring_radius_mm=30.0,
    z_offset_mm=-15.0,
    peaks_ppm={"Lactate": 1.3, **PEAKS_PPM},
)
```

Both the anatomical image and the MRSI grid place spheres from the same `world_xyz -> intensity`
function (`sphere_intensity`), evaluated at each one's own resolution — so a sphere lands in
roughly the same place in the T1-like image and in the spectral grid without needing the two to
share a coordinate system exactly. The alignment is good to about half a voxel (`create_MRSI_affine`
adds a half-voxel shift in x/y that the shared placement function doesn't replicate) — negligible
next to a 10 mm sphere radius.

:::{dropdown} Why not upload the NIST phantom and fake the rest?
Half-real, half-fake fixtures mean two code paths stay alive in every doc page that touches both,
and the repo still carries a binary blob whose provenance someone has to keep vouching for. One
generator, zero checked-in data, is simpler to reason about and is what issue #10 asks for anyway.
:::

(diary-synthetic-exam-fixtures-template)=
## Anatomy: a real, downloaded template

`HeVo-18`'s and `HeVo-23`'s anatomical/MRSI-seed NIfTIs are cropped from **Chris Rorden's
`chris_t1`**, part of the [niivue-images](https://github.com/neurolabusc/niivue-images) sample set,
licensed [**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/) — **non-commercial use
only**. `_fetch_t1_template_path()` downloads it once per machine into the system temp directory
(`mnutils_template_cache/`) and reuses that cache on every later call; nothing is checked into the
repo, and every CI run either downloads or reuses its own runner-local copy.

:::{dropdown} Why not skip the download and use a procedural phantom instead?
A layered-ellipsoid/Shepp-Logan-style skull-brain model would need no network access and no
attribution, but it isn't real anatomy — the whole point here was to stop looking at noise. The
template is small (~4 MB), cached after the first fetch, and CC BY-NC 4.0 is easy to honor for a
non-commercial open-source test fixture (attribution above, and in
[the GESeries class hierarchy](#data-model-geseries) page where it's first loaded).
:::

(diary-synthetic-exam-fixtures-cleanup)=
## Cleanup: `atexit`, not a fixture

`build_fake_exam()` is called with no arguments from both pytest and plain notebook cells (MyST's
`--execute` runs docs pages outside pytest entirely), so there's no shared fixture scope to hang
teardown on. It writes to a fresh `tempfile.mkdtemp()` directory the first time it's called per
process, caches that path (`functools.cache`) so every call within one test run or one notebook
reuses the same tree, and registers `atexit.register(shutil.rmtree, ...)` once — so the tree is
removed when the pytest process or the notebook kernel exits, whichever called it. The downloaded
template lives separately, in a machine-wide cache directory rather than the per-process tree, so
it survives across runs instead of being re-fetched every time.

`test_brain_extract.py::test_hevo23_bet` also gained a `pytest.importorskip("torch")` guard and an
explicit `device="cpu"` on `extract_brain()` — the `bet` extra (torch + HD-BET) isn't installed by
the CI `test` job's plain `uv sync`, and `extract_brain`'s `device` default is `"cuda"`, which
would fail on a GPU-less runner even with torch present. Both were latent gaps the missing dataset
had been masking — the test never got far enough to hit either.

:::{seealso}
[The GESeries class hierarchy](#data-model-geseries) and
[Loading a series and an exam](#basics-loading-data) are what these fixtures actually feed; this
entry only covers why they're fake (or not) and how they're built.
:::
