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

(basics-file-discovery)=
# Finding files in a data folder

`ExamBase` and `GESeries` never hardcode a path to a series's DICOM folder or `.mat` file — a GE
export names folders differently across scanner software versions (`010_axial_localizer` vs.
`Series0010_BS_prescan`), so every load goes through the same handful of discovery functions in
`utils.file_helpers`. Reaching for them directly is useful any time you want one file without
loading a whole series object.

| Function | What it does here |
|---|---|
| [`get_exam_folder()`](#mnutils.utils.file_helpers.get_exam_folder) | finds the folder containing raw scanner exam data |
| [`get_mat_data_from_series()`](#mnutils.utils.file_helpers.get_mat_data_from_series) | finds a series's reconstructed `.mat` file |
| [`get_dicom_folder()`](#mnutils.utils.file_helpers.get_dicom_folder) | finds a series's DICOM folder, across naming conventions |
| [`get_all_dicom_series_ids()`](#mnutils.utils.file_helpers.get_all_dicom_series_ids) | lists every series ID with a DICOM folder, and any gaps |
| [`move_files_with_glob()`](#mnutils.utils.file_helpers.move_files_with_glob) | archives matching files into an `old/` subfolder, timestamped |

```{code-cell} ipython3
:tags: [remove-cell]

from loguru import logger

logger.remove()
```

```{code-cell} ipython3
from pathlib import Path

from mnutils.utils import file_helpers as fh
from mnutils.testing import build_fake_exam

lg_folder = build_fake_exam("mrsi_missing_series_exam")
```

(basics-file-discovery-exam-mat)=
## The exam folder and a series's `.mat` file

`get_exam_folder` looks for the one subfolder with "Exam" in its name — that's where raw scanner
exam data (`.mat` reconstructions) lives, separate from the DICOM export next to it.

```{code-cell} ipython3
exam_folder = fh.get_exam_folder(lg_folder)
mat_path = fh.get_mat_data_from_series(lg_folder, 7)
print(exam_folder)
print(mat_path)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: exam folder + mat file discovery
assert exam_folder.name == "Exam4873anon"
assert mat_path.parent == exam_folder / "Series7"
assert mat_path.suffix == ".mat"
```

Both raise rather than guess when the folder structure doesn't match what's expected:

```{code-cell} ipython3
:tags: [remove-cell]

import pytest

# No "Exam*" folder at this level -- base_path is already inside one.
with pytest.raises(ValueError, match="No folder with"):
    fh.get_exam_folder(lg_folder / "Exam4873anon")

# Series 5 has no .mat file in this dataset.
with pytest.raises(FileNotFoundError, match="No .mat files found"):
    fh.get_mat_data_from_series(lg_folder, 5)
```

(basics-file-discovery-dicom)=
## DICOM folders, across naming conventions

Older exports name a series folder `NNN_<description>`; newer ones use `SeriesNNNN_<description>`.
`get_dicom_folder` matches either, so callers never branch on which convention a dataset uses.

```{code-cell} ipython3
old_style = fh.get_dicom_folder(lg_folder, 10)
new_style = fh.get_dicom_folder(build_fake_exam("brain_extraction_exam") / "data", 10)
print(old_style.name)
print(new_style.name)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: get_dicom_folder across naming conventions
assert old_style.name.startswith("010")
assert "BS_prescan" in new_style.name
with pytest.raises(ValueError, match="No DICOM folder matching"):
    fh.get_dicom_folder(lg_folder, 99)
```

(basics-file-discovery-series-ids)=
## Which series IDs actually have DICOM data

`get_all_dicom_series_ids` returns `(found, missing)` — every series ID with a DICOM folder, and
any integers skipped in that range (a genuinely missing/corrupt series, not just "series ID 1
doesn't exist because numbering starts at 1"). `ExamBase` uses this internally to build its
overview.

Only IDs ≤ 99 are checked for gaps, and only up to the highest one that exists. Those are the
protocol steps, numbered densely as the exam runs, so a hole there means something really is
absent. Everything from 100 up — reformats, resaves, late additions like `500`/`501` or `40003` —
is sparse by design, and neither gets flagged nor stretches the checked range.

```{code-cell} ipython3
found, missing = fh.get_all_dicom_series_ids(build_fake_exam("brain_mrs_mrsi_exam") / "data")
print(f"{len(found)} series found, gaps at {missing}")
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: get_all_dicom_series_ids
# Series 13 has raw exam data but no DICOM in this dataset (see docs/data-model/geseries.md) --
# it shows up as a gap here, exactly what ExamBase's classification depends on.
assert 13 in missing
assert 13 not in found
assert found == sorted(found)
```

(basics-file-discovery-archiving)=
## Archiving old output files

`move_files_with_glob` is unrelated to loading — it's what `RawMRISeries`/`MRSISeries` use
internally before writing new processed-data output, so a re-run doesn't silently overwrite a
previous result. It moves every match into an `old/` subfolder, prepending a timestamp so repeat
runs don't collide.

```{code-cell} ipython3
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    (folder / "result_a.h5").touch()
    (folder / "result_b.h5").touch()

    fh.move_files_with_glob(folder, "*.h5")

    moved = sorted(p.name for p in (folder / "old").glob("*.h5"))
    print(moved)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: move_files_with_glob
import re

with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    (folder / "a.h5").touch()
    fh.move_files_with_glob(folder, "*.h5")
    moved_files = list((folder / "old").glob("*.h5"))
    assert len(moved_files) == 1
    assert re.match(r"\d{8}_\d{6}_a\.h5", moved_files[0].name)
    assert not (folder / "a.h5").exists()  # moved, not copied
```

:::{seealso}
[Loading a series and an exam](#basics-loading-data) is where `ExamBase` puts these functions to
use — `get_exam_overview` (called on init) is built from `get_all_dicom_series_ids` and
`get_all_exam_series_ids` together.
:::
