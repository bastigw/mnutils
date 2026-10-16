(diary-nifti-only)=
# MNUtils reads NIfTI, and asks the files — not the folders — what they are

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-10-16</span>

MNUtils only ever reads NIfTI. DICOM is involved in just two places, and both cause trouble.
First, `MRISeries` quietly runs dcm2niix the first time a series is loaded and writes the result
into the raw data folder. That makes `dcm2niiw` a core dependency of a package that never reads a
DICOM byte. Second, `GEExam` decides what each series *is* from its DICOM folder name:
`get_dicom_folder`'s `(?:Series)?(\d{1,5})_` pattern for the ID, and substrings like `mrs_washin`,
`mrsi_pseudo` or `localizer` for the type. If someone renames a folder, or an export tool spells it
differently, that series is silently skipped or loaded as the wrong class.

:::{important}
MNUtils drops DICOM support entirely: it expects NIfTI files (converted beforehand, outside the
package, with JSON sidecars) and identifies every series from metadata inside the files, never
from folder names.
:::

(diary-nifti-only-shape)=
## Where a series' identity comes from

| Question | Today | After |
|---|---|---|
| Which image series exist? | DICOM folders matching `NNN_…` | every `*.nii*` under the data folder that has a JSON sidecar |
| Series ID of an image | number parsed from the folder name | sidecar `SeriesNumber` |
| Skip localizers, prescans | folder-name substring | sidecar `SeriesDescription` (same skip list) |
| MRS, washin or MRSI? | DICOM folder-name substring | sidecar `SeriesDescription` if the series has an image, else `.mat` `h.series.se_desc`; matched case-insensitively with spaces read as underscores |
| No NIfTI found | runs dcm2niix and writes into the data folder | `FileNotFoundError` that names the dcm2niix command to run |

```{mermaid}
flowchart LR
  N["<span style='font-family:monospace'>*.nii.gz + .json</span>"] -->|"SeriesNumber,<br>SeriesDescription"| I["exam index<br>(series ID → files, type)"]
  M["<span style='font-family:monospace'>Exam/SeriesN/ScanArchive*.mat</span>"] -->|"<span style='font-family:monospace'>h.series.se_desc</span><br>(fallback)"| I
  I --> G["GEExam picks<br>MRISeries / MRSSeries / MRSISeries …"]
```

The two sources are merged by series ID, so an MRSI series still finds its own pseudo-image
NIfTI. Folder names become free-form labels that nothing reads. The public call sites stay the
same:

```python
exam = GEExam(BASE_FOLDER)          # index built from sidecars + .mat headers
t1 = MRISeries(DATA_FOLDER, 2)      # resolved by SeriesNumber, not by "002_..."
mask = NiiBase("anywhere/brain.nii.gz")   # a bare NIfTI path never needed an exam
```

:::{warning}
This is a breaking change. A data folder that holds only DICOMs stops loading. Convert it once
with `dcm2niix -b y -z y -f %s_%d -o <series folder> <series folder>` before upgrading, and
archive the DICOMs (they are the only lossless copy). The `get_dicom_folder` and
`get_all_dicom_series_ids` names, the `convert_dicoms=` argument and the "Has DICOM" overview
column all go away, without deprecation aliases. It ships as a minor version bump, flagged in the
CHANGELOG.
:::

:::{dropdown} Why not keep conversion as an explicit `convert_exam()` inside MNUtils?
That would stop loading from writing into data folders, but MNUtils would still own a DICOM
dependency and the bugs that come with its edge cases. dcm2niix is already the standard tool and
the scanner workflow runs it anyway. Leaving it outside keeps the package to what it actually
reads.
:::

:::{dropdown} Why not keep folder names as a fallback when there is no sidecar?
A fallback keeps the fragile path alive, and it is the path that fails silently. A missing sidecar
is a loud, fixable error (re-run dcm2niix with `-b y`). A misnamed folder is a series that just
doesn't appear.
:::

:::{note} Checked against the real test exams
Every point below was run against the exams in `tests/datasets/` (NIST phantom, HeVo-18, LG_D19,
TestDownSampling).

- **Sidecars carry the identity.** All 30 sidecars have `SeriesNumber` and `SeriesDescription`, and
  `SeriesNumber` equals the folder number every time, including `Series0002_…` and `;`-separated
  folder names.
- **`se_desc` is not the pseudo-image name for MRSI.** For MRS and washin it equals the sidecar
  description (`MRS unloc`, `MRS washin`). For MRSI it is the *sequence* name (`MRSI 3D 2 NEX`,
  `2H bssfp16`, …); the `MRSI pseudo:S700;…` text exists only in the NIfTI sidecar. So the type comes
  from the sidecar when the series has an image, and from `se_desc` otherwise. Either source is
  lower-cased with spaces read as underscores before the existing `mrs_washin` / `mrsi_pseudo` /
  `mrs_` rules run.
- **Main image = the NIfTI that has a sidecar.** "Sidecar without a dcm2niix suffix" fails: NIST
  series 12 only has `…_e6.json`. Derived files (`_bet`, `_bet_bet`, `*_resampled_lowres*`, `.OLD`)
  have no sidecar and fall out. With several sidecars (FGRE `i00001/06/11`), take the lowest suffix.
- **Series with a `.mat` but no NIfTI still load** (NIST series 4), as MRS/MRSI without an image.
- **The raw archive is the `.mat` with the shortest suffix after `ScanArchive_<id>`.** The plain
  `ScanArchive_<id>.mat` wins; when only suffixed ones exist (NIST series 13: `_csi.mat`,
  `_snr_map.mat`), the shorter suffix wins (`_csi`). Any choice among several candidates logs a
  warning naming the file picked.
- **Series with only an `.h5` are not indexed.** Their `Header.xml` has exam and series numbers but
  no description, so they cannot be typed. They stay loadable by explicit series ID.
:::

:::{important} Scope: the whole package moves in this change
The DICOM names do not survive as deprecated shims. This change also:

- reworks `mnutils.testing.synthetic` so the fake exams ship sidecars (`SeriesNumber`,
  `SeriesDescription`) and `se_desc` values shaped like the real ones above, instead of folder
  names that `get_dicom_folder` parses;
- rewrites every caller of `get_dicom_folder`, `get_all_dicom_series_ids` and `convert_dicoms=`
  (`file_helpers.py`, `images.py`, `GEExam.py`, `GESeries.py`) onto the sidecar/`.mat` index, and
  deletes those functions and the dcm2niix call;
- rewrites `docs/basics/file_discovery.md`, the only docs page that depends on folder parsing.
:::
