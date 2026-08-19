(diary-synthetic-exam-fixtures)=
# CI can't see a scanner, so the test exam has to be born on disk

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-19</span>

CI (#3) runs green locally-faked but fails for real: `tests/datasets/` is `.gitignore`d, so
`HeVo-18`, `HeVo-23`, `LG_D19`, `20250408-NIST-Mag2`, and `TestDownSampling` don't exist on the
runner, and 8 tests/notebooks that hard-code paths into them die on `ValueError: The base folder
does not exist`. Uploading them isn't a small fix either — they're real scanner acquisitions with
unclear redistribution provenance, several MB each, and brittle to extend (issue #10). The NIST
phantom is anonymized and small enough that uploading it would have been fine, but it only covers
one of five dataset shapes docs pages rely on — the rest would still need faking, so faking all
five once is less work than faking four and uploading one.

:::{important}
Build one synthetic-exam generator that fabricates, at test time, every on-disk shape the real
datasets currently provide — GE folder-naming, `.mat` recon headers, raw-FID `.h5` caches, and
NIfTI volumes — and point every doc page and test at it instead of `tests/datasets/`.
:::

(diary-synthetic-exam-fixtures-shape)=
## What has to be faked, and what doesn't

The loaders turn out to need far less real-looking data than the dataset names suggest:

| Real thing | What the code actually reads | Fake needed |
|---|---|---|
| DICOM series | `get_dicom_folder` only checks the folder name (`002_...`); `MRISeries` loads any `.nii`/`.nii.gz` already sitting in that folder before ever trying DICOM conversion | a tiny NIfTI, zero real DICOM |
| `ScanArchive*.mat` | `RawMRISeries` parses `h.rdb_hdr`/`h.series`/`h.exam`/`h.image`/`h.mrconfig` plus `spec`/`bb`/`bbabs` out of a nested MATLAB struct | a `scipy.io.savemat` dict with those exact keys |
| `ScanArchive*.h5` (raw P-file) | `load_raw_fids` only needs the file to *exist* to resolve the series folder — if a `Series{id}_raw_fids.h5` cache already sits next to it, its `fids` dataset is read directly and the archive's contents are never touched | an empty stub `.h5` + a pre-built cache `.h5` |

```python
# the call site every doc page and test should end up with
from mnutils.testing import build_fake_exam

DATA_FOLDER = build_fake_exam(tmp_path)  # one exam, every series shape below
t1 = MRISeries(DATA_FOLDER, 2)
mrs = MRSSeries(DATA_FOLDER, 6)
mrsi = MRSISeries(DATA_FOLDER, 8)
```

One consolidated exam replaces all five real datasets: series numbers and folder-naming choices
are picked to satisfy every doc page currently keyed to `HeVo-18`/`HeVo-23`/`LG_D19`/
`20250408-NIST-Mag2` (T1 + MRS + washin + MRSI + a series missing its `.mat`, a series with exam
data but no DICOM — `file_discovery.md`'s presence/absence narrative — plus a second, differently
laid out phantom exam so `geseries.md`'s resample/overlay walkthrough still crosses two exams).
`TestDownSampling` has no doc-page reference left to trace — dropped rather than faked.

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

:::{attention} Assumptions to verify
- `scipy.io.savemat` round-trips a nested Python dict into the `mat_struct` shape
  `load_mat_file`/`_todict` expects (untested — this is the generator's main risk).
- hd-bet actually runs end-to-end on a synthetic (non-brain-shaped) NIfTI volume without erroring;
  `test_brain_extract.py` also needs a `pytest.importorskip("torch")` guard to match the `bet`
  extra being optional in the CI `test` job, independent of the fixture question.
- One consolidated exam can satisfy every doc page's series-numbering assumptions without
  colliding (e.g. `geseries.md` series 2/6/7/8 vs. `file_discovery.md` series 5/10/13) — the exact
  layout still needs to be worked out per page during implementation.
:::
