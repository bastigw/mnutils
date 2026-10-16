(diary-nifti-only)=
# MNUtils reads NIfTI, and asks the files — not the folders — what they are

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-10-16</span>

MNUtils only ever reads NIfTI, yet DICOM used to leak in at two points, and both caused trouble.
First, `MRISeries` quietly ran dcm2niix the first time a series was loaded and wrote the result
into the raw data folder, which made `dcm2niiw` a core dependency of a package that never read a
DICOM byte. Second, `GEExam` decided what each series *was* from its DICOM folder name: a
`(?:Series)?(\d{1,5})_` pattern for the ID, and substrings like `mrs_washin`, `mrsi_pseudo` or
`localizer` for the type. Rename a folder, or export with a tool that spells it differently, and
that series was silently skipped or loaded as the wrong class.

:::{important}
MNUtils drops DICOM support entirely: it expects NIfTI files (converted beforehand, outside the
package, with JSON sidecars) and identifies every series from metadata inside the files, never
from folder names.
:::

(diary-nifti-only-shape)=
## Where a series' identity comes from

| Question | Before | Now |
|---|---|---|
| Which image series exist? | DICOM folders matching `NNN_…` | every `*.nii*` under the data folder that has a same-named JSON sidecar with a `SeriesNumber` |
| Series ID of an image | number parsed from the folder name | sidecar `SeriesNumber` |
| Skip localizers, FGRE | folder-name substring | sidecar `SeriesDescription` (same skip list) |
| MRS, washin or MRSI? | folder-name substring | sidecar `SeriesDescription` if the series has an image, else the `.mat` header's `h.series.se_desc` |
| Which `.mat` is the raw archive? | first glob hit | the plain `ScanArchive_*.mat`, else the shortest suffix; a warning names the pick |
| No NIfTI found | ran dcm2niix, wrote into the data folder | `FileNotFoundError` that names the dcm2niix command |

```{mermaid}
flowchart LR
  N["<span style='font-family:monospace'>*.nii.gz + .json</span>"] -->|"SeriesNumber,<br>SeriesDescription"| I["exam overview<br>(series ID → description)"]
  M["<span style='font-family:monospace'>Exam/SeriesN/ScanArchive*.mat</span>"] -->|"<span style='font-family:monospace'>h.series.se_desc</span><br>(only if no sidecar)"| I
  I --> G["ExamBase picks<br>MRISeries / MRSSeries / MRSISeries …"]
```

The two sources are merged by series ID, so an MRSI series still finds its own pseudo-image NIfTI.
Descriptions are lower-cased with spaces read as underscores (`MRS washin` → `mrs_washin`) before
the type rules run, in `GEExam._create_series_dict`: `mrs_washin`, then `mrsi`, then `mrs_`;
`bs_prescan` is skipped, and an exam series with no description at all is assumed to be MRS. Folder
names are now free-form labels that nothing reads. The public call sites stay the same:

```python
exam = ExamBase(BASE_FOLDER)  # overview built from sidecars + .mat headers
t1 = MRISeries(DATA_FOLDER, 2)  # resolved by SeriesNumber, not by "002_..."
mask = NiiBase("anywhere/brain.nii.gz")  # a bare NIfTI path never needed an exam
```

:::{warning}
This is a breaking change. A data folder that holds only DICOMs stops loading. Convert it once
with `dcm2niix -b y -z y -f %s_%d -o <series folder> <series folder>` before upgrading, and
archive the DICOMs (they are the only lossless copy). `get_dicom_folder`,
`get_all_dicom_series_ids` and the `convert_dicoms=` argument are deleted, without deprecation
aliases, and the overview's `Has DICOM` / `Folder Name` columns become `Has NIfTI` /
`Description`. It is flagged in the CHANGELOG and ships as a minor bump.
:::

(diary-nifti-only-real-data)=
## What the real exams said

The first draft of this entry rested on five assumptions about GE exports. They were checked
against four real datasets (a phantom and three subject sessions, about 30 sidecars and 25 `.mat`
headers) before any code was written, and two of them broke:

- **`se_desc` is not the pseudo-image name for MRSI.** The draft planned to read the series type
  from the `.mat` header and keep the existing `mrsi_pseudo` rule. For MRS and washin the header
  does match the sidecar (`MRS unloc`, `MRS washin`), but for MRSI it holds the *sequence* name
  (`MRSI 3D 2 NEX`, `2H bssfp16`); the `MRSI pseudo:S700;…` text exists only in the NIfTI sidecar.
  So the sidecar decides when a series has an image and `se_desc` is the fallback, and the rule
  matches `mrsi` rather than `mrsi_pseudo`.
- **"The sidecar without a dcm2niix suffix" does not pick the main image.** NIST series 12 only has
  `…_e6.json`, and FGRE has three numbered sidecars. A NIfTI counts when it has a sidecar of its
  own, which also drops the derived `_bet`, `_resampled_*` and `.OLD` files without special cases;
  among several, the lowest file name wins and a warning names it.

The other three held: sidecars always carry `SeriesNumber` and `SeriesDescription` and agree with
the folder number; raw archives next to derived ones (`…_csi.mat`, `…_snr_map.mat`) are resolved by
suffix length; and a series with a `.mat` but no NIfTI (NIST series 4) still loads, as MRS or MRSI
without an image. Series with only a raw `.h5` have nothing to type them by (`Header.xml` carries
the exam and series numbers but no description), so they are not in the overview; they stay
loadable by explicit series ID.

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

(diary-nifti-only-fixtures)=
## The fake exams moved with it

`mnutils.testing.build_fake_exam` used to fabricate folders that `get_dicom_folder` parsed. It now
writes a sidecar next to every NIfTI and `se_desc` values shaped like the real ones (`MRS unloc`,
`MRS washin`, `MRSI 3D 2 NEX`, with the `MRSI pseudo:…` text on the pseudo-image's sidecar). Series
13 of the brain exam became an MRS `.mat` with no NIfTI so the docs keep one example of raw data
without an image, and `docs/basics/file_discovery.md`, the only docs page that depended on folder
parsing, was rewritten around `get_niftis_from_series` and `get_all_nifti_series_ids`.
