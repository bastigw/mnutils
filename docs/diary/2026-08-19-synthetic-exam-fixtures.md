(diary-synthetic-exam-fixtures)=
# CI can't see a scanner, so the test exam has to be born on disk

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-19</span>

CI (#3) ran green locally and failed for real: `tests/datasets/` is `.gitignore`d, so `HeVo-18`,
`HeVo-23`, `LG_D19`, and `20250408-NIST-Mag2` didn't exist on the runner, and 8 tests/notebooks
that hard-coded paths into them died on `ValueError: The base folder does not exist`. Uploading
them wasn't a small fix either — they're real scanner acquisitions with unclear redistribution
provenance, several MB each, and brittle to extend (issue #10). The NIST phantom is anonymized and
small enough that uploading it would have been fine, but it only covers one of four dataset
shapes docs pages rely on — the rest would still need faking, so faking all four once turned out
to be less work than faking three and uploading one.

:::{important}
`mnutils.testing.build_fake_exam()` fabricates, at test/build time, every on-disk shape the real
datasets used to provide — GE folder-naming, `.mat` recon headers, raw-FID `.h5` caches, and NIfTI
volumes — and every doc page and test now points at it instead of `tests/datasets/`.
:::

(diary-synthetic-exam-fixtures-shape)=
## What has to be faked, and what doesn't

The loaders need far less real-looking data than the dataset names suggest:

| Real thing | What the code actually reads | Fake used |
|---|---|---|
| DICOM series | `get_dicom_folder` only checks the folder name (`002_...`); `MRISeries` loads any `.nii.gz` already sitting in that folder before ever trying DICOM conversion | a small `nibabel.Nifti1Image` with a diagonal affine, zero real DICOM |
| `ScanArchive*.mat` | `RawMRISeries` parses `h.rdb_hdr`/`h.series`/`h.exam`/`h.image`/`h.mrconfig` plus `spec`/`bb` out of a nested MATLAB struct | a plain nested `dict` through `scipy.io.savemat` — it round-trips into the `mat_struct` shape `load_mat_file`/`_todict` expect with no special handling needed |
| `ScanArchive*.h5` (raw P-file) | `load_raw_fids` only needs the file to *exist* to resolve the series folder — a `Series{id}_raw_fids.h5` cache sitting next to it is read directly and the archive's contents are never touched | an empty stub `.h5` + a pre-built cache `.h5` |
| Legacy v7.3 `.mat` (`mat_files.md`'s `mat73` fallback case) | `mat73.loadmat` reads any HDF5 file whose datasets carry a `MATLAB_class` attribute | a bare `h5py.File` with one dataset and that one attribute set — no real MATLAB writer involved |

```python
# the call site every doc page and test ends up with
from mnutils.testing import build_fake_exam

DATASETS = build_fake_exam()  # generated once per process, cleaned up at exit
t1 = MRISeries(DATASETS / "HeVo-18" / "data", 2)
mrs = MRSSeries(DATASETS / "HeVo-18" / "data", 6)
mrsi = MRSISeries(DATASETS / "HeVo-18" / "data", 8)
```

`build_fake_exam()` builds four named sub-trees — `HeVo-18/`, `HeVo-23/`, `LG_D19/`,
`20250408-NIST-Mag2/` — plus a loose legacy `.mat` file, keeping the original dataset names and
directory shapes (`LG_D19` has no `data/` subfolder, `HeVo-23` uses new-style `SeriesNNNN_...`
DICOM naming, `LG_D19`'s exam folder is literally named `Exam4873anon`) so every doc page's
existing code needed only its `DATASETS = ...` line swapped, not a rewrite. Series numbers and
narrative gaps are picked to match what each page already asserts: `HeVo-18` series 6/7/8 for
MRS/washin/MRSI, series 13 with exam data but no DICOM folder (`file_discovery.md`'s missing-series
case), `LG_D19` series 5 with no `.mat` file (the `FileNotFoundError` case) and series 7 shaped
`(700, ...)` to match `mat_files.md`'s assertion.

:::{dropdown} Why not upload the NIST phantom and fake the rest?
Half-real, half-fake fixtures mean two code paths stay alive in every doc page that touches both
(`DATASETS / "20250408-NIST-Mag2"` next to a generator call), and the repo still carries a binary
blob whose provenance someone has to keep vouching for. One generator, zero checked-in data, is
simpler to reason about and is what issue #10 asks for anyway.
:::

:::{dropdown} Why not real DICOM/`.mat` bytes via a synthesis library?
`pydicom`/pyAMARES-style FID synthesis would make the fixtures look more like real acquisitions,
but nothing under test actually parses pixel data or FID physics — `_get_header_value` and
`get_nifti_file`/`get_h5_data_from_series` only care about struct keys and file existence. Matching
real bytes buys realism the assertions can't see.
:::

(diary-synthetic-exam-fixtures-cleanup)=
## Cleanup: `atexit`, not a fixture

`build_fake_exam()` is called with no arguments from both pytest and plain notebook cells (MyST's
`--execute` runs docs pages outside pytest entirely), so there's no shared fixture scope to hang
teardown on. It writes to a fresh `tempfile.mkdtemp()` directory the first time it's called per
process, caches that path (`functools.cache`) so every call within one test run or one notebook
reuses the same tree, and registers `atexit.register(shutil.rmtree, ...)` once — so the tree is
removed when the pytest process or the notebook kernel exits, whichever called it. Nothing
accumulates across runs because each process gets its own unique temp directory.

`test_brain_extract.py::test_hevo23_bet` also gained a `pytest.importorskip("torch")` guard and an
explicit `device="cpu"` on `extract_brain()` — the `bet` extra (torch + HD-BET) isn't installed by
the CI `test` job's plain `uv sync`, and `extract_brain`'s `device` default is `"cuda"`, which
would fail on a GPU-less runner even with torch present. Both were latent gaps the missing dataset
had been masking — the test never got far enough to hit either.

:::{seealso}
[The GESeries class hierarchy](#data-model-geseries) and
[Loading a series and an exam](#basics-loading-data) are what these fixtures actually feed; this
entry only covers why they're fake and how they're built.
:::
